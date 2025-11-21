"""
mcts.py

Core implementation of the Monte Carlo Tree Search (MCTS) algorithm for the Chess AI.
This module implements the search strategy described in `vision.md`, combining
classical tree search with neural network guidance (PUCT).

Key Components:
- BatchInference: Orchestrates efficient GPU usage by batching requests from multiple workers.
- MCTSNode: Represents the game state and search statistics.
- MCTS: Implements the selection, expansion, evaluation, and backpropagation phases.

References:
    - Project Vision: "Core Components > mcts.py"
    - AlphaZero Paper: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
"""

import math
import queue
import threading
import numpy as np
import torch
import chess
from typing import List, Tuple, Dict, Optional, Any, Union

from encode import board_to_tensor, move_to_action

# --- Constants ---

# Controls the level of exploration in the PUCT formula.
# Higher values encourage exploring nodes with lower visit counts.
DEFAULT_C_PUCT = 1.0

# Dirichlet noise parameters for root node exploration (Phase 4 / Self-play).
DEFAULT_DIRICHLET_ALPHA = 0.3
DEFAULT_DIRICHLET_EPSILON = 0.25


class BatchInference:
    """
    Manages batched inference for multiple MCTS workers/threads.
    
    As described in `vision.md` ("Scaling and orchestration"), this class allows
    multiple CPU workers to push states to a shared queue. A background thread
    batches these states and queries the GPU-based `model.py` in one go, 
    significantly improving throughput compared to sequential inference.
    """

    def __init__(self, model: torch.nn.Module, batch_size: int = 8, device: str = 'cpu'):
        self.model = model
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.queue: queue.Queue = queue.Queue()
        self.running = True
        
        # Ensure model is in eval mode and on correct device
        self.model.to(self.device)
        self.model.eval()
        
        # Start the background inference thread
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _inference_loop(self):
        """
        Continuous loop that pulls from the queue, batches requests,
        runs the neural network, and distributes results.
        """
        while self.running:
            requests = []
            
            # 1. Blocking get for the first item (waits until at least one request exists)
            try:
                req = self.queue.get(timeout=0.1)
                requests.append(req)
            except queue.Empty:
                continue
            
            # 2. Non-blocking get for subsequent items to fill the batch
            while len(requests) < self.batch_size:
                try:
                    req = self.queue.get_nowait()
                    requests.append(req)
                except queue.Empty:
                    break
            
            if not requests:
                continue
                
            # 3. Prepare batch
            # Assuming req['state'] is a ChessGame object or similar wrapper
            states = [req['state'] for req in requests]
            
            # 4. Run inference (no_grad for efficiency)
            with torch.no_grad():
                # Convert all boards to a single tensor batch
                batch_tensor = torch.stack([board_to_tensor(s.board) for s in states]).to(self.device)
                
                # Model outputs: policy (logits), value (scalar)
                policies, values = self.model(batch_tensor)
                
                # Move to CPU/numpy for easier handling in MCTS
                policies = policies.cpu().numpy()
                values = values.cpu().numpy()
            
            # 5. Distribute results back to waiting threads
            for i, req in enumerate(requests):
                req['result'] = (policies[i], values[i])
                req['event'].set()

    def predict(self, state: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Thread-safe method to request inference for a single state.
        Blocks until the batch containing this request is processed.
        
        Args:
            state: The game state object (must have a .board attribute).
            
        Returns:
            Tuple of (policy_logits, value).
        """
        event = threading.Event()
        req = {'state': state, 'event': event, 'result': None}
        self.queue.put(req)
        event.wait() # Block until result is ready
        return req['result']

    def stop(self):
        """Stops the inference thread."""
        self.running = False
        self.thread.join()


class MCTSNode:
    """
    Represents a node in the Monte Carlo Search Tree.
    
    Attributes:
        state: The game state corresponding to this node.
        parent: The parent MCTSNode (None for root).
        action: The action (move) taken to reach this node from parent.
        children: Dictionary mapping actions to child MCTSNodes.
        visit_count (N): Number of times this node has been visited.
        value_sum (W): Sum of value estimates for this node.
        prior (P): Prior probability of picking this action (from neural net).
    """
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, 
                 action: Optional[chess.Move] = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        
    @property
    def value(self) -> float:
        """Mean action value (Q = W / N)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """Returns True if the node has been expanded (has children)."""
        return len(self.children) > 0
    
    def __repr__(self):
        return f"<MCTSNode N={self.visit_count} Q={self.value:.3f} P={self.prior:.3f}>"


class MCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation using the PUCT algorithm.
    
    As outlined in `vision.md`, this class coordinates the four phases of search:
    1. Selection
    2. Expansion
    3. Evaluation
    4. Backpropagation
    """

    def __init__(self, inference_service: BatchInference, 
                 c_puct: float = DEFAULT_C_PUCT, 
                 dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA, 
                 dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON):
        self.inference_service = inference_service
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, root: Union[MCTSNode, Any], num_simulations: int = 800) -> MCTSNode:
        """
        Performs MCTS simulations starting from the root state.
        
        Args:
            root: The root MCTSNode or a game state object.
            num_simulations: Number of simulations to run (default 800, adjustable per `vision.md` scaling).
            
        Returns:
            The root MCTSNode with updated statistics.
        """
        # Ensure root is a node
        if not isinstance(root, MCTSNode):
            root = MCTSNode(root, prior=1.0)
            
        # Initial expansion with noise if needed (Phase 4: Exploration)
        if not root.is_expanded():
             self._expand(root, add_noise=True)

        for _ in range(num_simulations):
            node = root
            
            # 1. Selection: Traverse tree to a leaf node
            while node.is_expanded():
                child = self._select_child(node)
                if child is None:
                    # Should be rare, but protects against empty children dicts
                    break
                node = child
            
            # 2. Expansion & 3. Evaluation: Expand leaf and get value
            value = self._expand(node)
            
            # 4. Backpropagation: Update stats up the tree
            self._backpropagate(node, value)
            
        return root

    def _select_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Selects the child with the highest PUCT score.
        Formula: U(s,a) = Q(s,a) + C_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        best_score = -float('inf')
        best_child = None
        
        sqrt_total_visits = math.sqrt(node.visit_count)
        
        for action, child in node.children.items():
            q_value = child.value
            # PUCT calculation
            u_value = self.c_puct * child.prior * sqrt_total_visits / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def _expand(self, node: MCTSNode, add_noise: bool = False) -> float:
        """
        Expands a leaf node using the neural network.
        
        Responsibilities:
        - Check for terminal states (game over).
        - Query `inference_service` for Policy (priors) and Value.
        - Mask invalid moves and normalize policy.
        - Create child nodes for all valid moves.
        
        Returns:
            The value estimate (v) of the state.
        """
        state = node.state
        
        # Handle Terminal States
        if state.is_terminal():
            # Get result: 1 (White win), -1 (Black win), 0 (Draw)
            winner = state.get_winner()
            if winner is None:
                return 0.0
            
            # Value is from the perspective of the player who just moved (parent node's actor).
            # If it's White's turn in `state`, then Black just moved.
            # Standard AlphaZero: value is relative to the current player in `state`.
            # If state.turn == WHITE and winner == 1, value = 1.
            # If state.turn == BLACK and winner == 1, value = -1 (White won, bad for Black).
            
            turn_multiplier = 1 if state.get_turn() == chess.WHITE else -1
            return winner * turn_multiplier

        # Inference: Get raw policy logits and value from the network
        policy_logits, value = self.inference_service.predict(state)
        value = value[0] # Unpack scalar from (1,) array
        
        # Legal Move Masking & Policy normalization
        legal_moves = state.get_legal_moves()
        policy_map = {}
        
        for move in legal_moves:
            # Convert move to action index (0-4671)
            action_data = move_to_action(move, state.board)
            
            # Robustness check: move_to_action returns None for unencodable moves
            if action_data is None:
                continue
                
            from_sq, action_type = action_data
            
            # Map linear action index to 8x8x73 tensor coordinates
            r = from_sq // 8
            c = from_sq % 8
            
            logit = policy_logits[r, c, action_type]
            policy_map[move] = logit
            
        # Softmax over ONLY legal moves (prevents exploring illegal actions)
        # Uses log-sum-exp trick for numerical stability
        max_logit = max(policy_map.values()) if policy_map else 0
        sum_exp = sum(math.exp(logit - max_logit) for logit in policy_map.values())
        
        # Add Dirichlet noise to the root (if requested) to encourage exploration
        noise = None
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
        
        # Create children
        for i, move in enumerate(legal_moves):
            if move not in policy_map:
                continue
                
            # Calculate probability
            prob = math.exp(policy_map[move] - max_logit) / sum_exp
            
            # Mix in noise
            if add_noise and noise is not None:
                prob = (1 - self.dirichlet_epsilon) * prob + self.dirichlet_epsilon * noise[i]
            
            # Advance state for child
            next_state = state.copy()
            next_state.make_move(move)
            
            child = MCTSNode(next_state, parent=node, action=move, prior=prob)
            node.children[move] = child
            
        return value

    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagates the value estimate up the tree.
        
        Since Chess is a zero-sum game, the value flips sign at each level:
        If state S is +1 for White, the parent (Black's turn) sees it as -1.
        """
        curr = node
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += value
            
            value = -value
            curr = curr.parent

    def get_action_prob(self, root: MCTSNode, temperature: float = 1.0) -> Tuple[List[chess.Move], List[float]]:
        """
        Returns the action probabilities based on visit counts.
        
        Args:
            root: The root MCTSNode after searching.
            temperature: Controls the shape of the distribution.
                         1.0 = proportional to visits (standard).
                         0.0 = greedy (max visits).
                         >1.0 = flatter distribution (more exploration).
                         
        Returns:
            Tuple of (actions, probabilities).
        """
        actions = list(root.children.keys())
        visits = [child.visit_count for child in root.children.values()]
        
        if not visits:
             return actions, []
        
        if temperature == 0:
            # Greedy: Return 1.0 for the max-visit action(s)
            max_visit = max(visits)
            best_indices = [i for i, v in enumerate(visits) if v == max_visit]
            probs = [0.0] * len(actions)
            for i in best_indices:
                probs[i] = 1.0 / len(best_indices)
            return actions, probs

        # Softmax-like scaling with temperature
        # v^(1/T)
        visits_powered = [v ** (1.0 / temperature) for v in visits]
        sum_visits = sum(visits_powered)
        probs = [v / sum_visits for v in visits_powered]
        
        return actions, probs
