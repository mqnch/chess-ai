import math
import queue
import threading
import time
import numpy as np
import torch
import chess
from encode import board_to_tensor, move_to_action, action_to_move, QUEEN_MOVE_OFFSET, KNIGHT_MOVE_OFFSET, UNDERPROMOTION_OFFSET

class BatchInference:
    """
    Manages batched inference for multiple MCTS workers.
    Collects states from workers, batches them, runs the neural network,
    and distributes results back to the waiting workers.
    """
    def __init__(self, model, batch_size=8, device='cpu'):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.queue = queue.Queue()
        self.running = True
        self.model.to(self.device)
        self.model.eval()
        
        # Start the inference thread
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _inference_loop(self):
        """
        Continuous loop that pulls from the queue, batches requests,
        runs inference, and sets results.
        """
        while self.running:
            requests = []
            
            # Get first item (blocking)
            try:
                req = self.queue.get(timeout=0.1)
                requests.append(req)
            except queue.Empty:
                continue
            
            # Get more items to fill batch (non-blocking)
            while len(requests) < self.batch_size:
                try:
                    req = self.queue.get_nowait()
                    requests.append(req)
                except queue.Empty:
                    break
            
            if not requests:
                continue
                
            # Prepare batch
            states = [req['state'] for req in requests]
            
            # Run inference
            with torch.no_grad():
                batch_tensor = torch.stack([board_to_tensor(s.board) for s in states]).to(self.device)
                policies, values = self.model(batch_tensor)
                
                # Move to CPU/numpy for easier handling
                policies = policies.cpu().numpy()
                values = values.cpu().numpy()
            
            # Distribute results
            for i, req in enumerate(requests):
                req['result'] = (policies[i], values[i])
                req['event'].set()

    def predict(self, state):
        """
        Thread-safe method to request inference for a single state.
        Blocks until the result is available.
        """
        event = threading.Event()
        req = {'state': state, 'event': event, 'result': None}
        self.queue.put(req)
        event.wait()
        return req['result']

    def stop(self):
        self.running = False
        self.thread.join()

class MCTSNode:
    """
    Represents a node in the MCTS tree.
    """
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}  # Map from action to MCTSNode
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior  # P(s, a)
        
    @property
    def value(self):
        """Mean value of the node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

class MCTS:
    """
    Monte Carlo Tree Search implementation using PUCT.
    """
    def __init__(self, inference_service, c_puct=1.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.inference_service = inference_service
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, root, num_simulations=800):
        """
        Performs MCTS simulations starting from the root state.
        """
        # Initialize root node if it's just a state
        if not isinstance(root, MCTSNode):
            root = MCTSNode(root, prior=1.0)
            
        # Add exploration noise to root if it's not expanded yet or we want to refresh it
        # Typically noise is added once at the beginning of the search for the root
        if not root.is_expanded():
             self._expand(root, add_noise=True)

        for _ in range(num_simulations):
            node = root
            
            # 1. Selection
            while node.is_expanded():
                node = self._select_child(node)
            
            # 2. Expansion & 3. Evaluation
            value = self._expand(node)
            
            # 4. Backpropagation
            self._backpropagate(node, value)
            
        return root

    def _select_child(self, node):
        """
        Selects the child with the highest PUCT score.
        """
        best_score = -float('inf')
        best_child = None
        
        # Precompute sqrt(sum(N)) for PUCT formula
        sqrt_total_visits = math.sqrt(node.visit_count)
        
        for action, child in node.children.items():
            q_value = child.value
            u_value = self.c_puct * child.prior * sqrt_total_visits / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def _expand(self, node, add_noise=False):
        """
        Expands the leaf node using the neural network.
        Returns the value of the state (from network or terminal condition).
        """
        state = node.state
        
        # Check if terminal
        if state.is_terminal():
            # get_result returns '1-0', '0-1', '1/2-1/2'
            # We need to convert this to value from the perspective of the current player
            # Note: Model predicts value for the player whose turn it is.
            # If white wins (1), and it's white's turn, value is 1.
            # If white wins (1), and it's black's turn, value is -1.
            
            winner = state.get_winner() # 1 (white), -1 (black), 0 (draw), None (not over)
            if winner is None:
                # Should not happen if is_terminal is true
                return 0 
            
            # Value is from the perspective of the player to move *in the parent state*?
            # Usually standard AlphaZero: Value is for the player to move in current state 's'.
            # If s is terminal, value is simple.
            
            # If it is White's turn, and White wins, value is +1.
            # If it is Black's turn, and Black wins (winner=-1), value is +1 (good for Black).
            
            turn_multiplier = 1 if state.get_turn() == chess.WHITE else -1
            return winner * turn_multiplier

        # Inference
        policy_logits, value = self.inference_service.predict(state)
        value = value[0] # Extract scalar
        
        # Generate legal moves and corresponding probabilities
        legal_moves = state.get_legal_moves()
        policy_map = {}
        
        # Convert logits to probabilities for legal moves only
        # policy_logits shape is (8, 8, 73)
        
        for move in legal_moves:
            from_sq, action_type = move_to_action(move, state.board)
            # from_sq is 0-63, action_type is 0-72
            
            # Map 0-63 from_sq to row/col
            r = from_sq // 8
            c = from_sq % 8
            
            logit = policy_logits[r, c, action_type]
            policy_map[move] = logit
            
        # Softmax over legal moves
        max_logit = max(policy_map.values()) if policy_map else 0
        sum_exp = sum(math.exp(logit - max_logit) for logit in policy_map.values())
        
        # Dirichlet noise parameters
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
        
        for i, move in enumerate(legal_moves):
            prob = math.exp(policy_map[move] - max_logit) / sum_exp
            
            if add_noise:
                prob = (1 - self.dirichlet_epsilon) * prob + self.dirichlet_epsilon * noise[i]
            
            # Create child node
            # We need to create a new state for the child
            next_state = state.copy()
            next_state.make_move(move)
            
            child = MCTSNode(next_state, parent=node, action=move, prior=prob)
            node.children[move] = child
            
        return value

    def _backpropagate(self, node, value):
        """
        Updates the node statistics along the path to the root.
        Value is from the perspective of the player at the leaf node.
        """
        curr = node
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += value
            
            # The value is relative to the player who just moved (or the player to move at this node?)
            # AlphaZero convention:
            # v is the value for the player to move at state s.
            # When moving up the tree, the player to move switches.
            # So we negate the value.
            
            value = -value
            curr = curr.parent

    def get_action_prob(self, root, temperature=1.0):
        """
        Returns the action probabilities from the root node visits.
        """
        visits = [child.visit_count for child in root.children.values()]
        actions = [action for action in root.children.keys()]
        
        if sum(visits) == 0:
             # Should not happen if searched
             return actions, [1/len(actions)] * len(actions)
        
        if temperature == 0:
            # Greedy selection
            max_visit = max(visits)
            best_actions = [a for a, v in zip(actions, visits) if v == max_visit]
            probs = [1.0 / len(best_actions) if a in best_actions else 0.0 for a in actions]
            return actions, probs

        # Apply temperature
        visits = [v ** (1.0 / temperature) for v in visits]
        sum_visits = sum(visits)
        probs = [v / sum_visits for v in visits]
        
        return actions, probs

