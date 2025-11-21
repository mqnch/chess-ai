alphazero chess project
roadmap:
Phase 1 – Set up environment and basic modules (Week 1)
- [x] Set up development environment: Install Python 3, python‑chess, PyTorch with CUDA support and standard libraries; configure version control.
- [x] Implement basic board management (board.py): wrap the python‑chess Board class to expose game state, legal moves and terminal detection. Design a clean API for MCTS and encoding.
- [x] Design the encoding (encode.py): decide on the number of planes and implement functions to convert a Board into a tensor and back. Implement decoding of the policy output into legal moves.


Phase 2 – Neural network design and baseline model (Week 2)
- [x] Design the CNN architecture (model.py): implement a convolutional block followed by a configurable number of residual blocks. Start with 8–10 blocks and 128–256 channels to fit on a single GPUgithub.com.
- [x] Implement policy and value heads: a small convolution followed by a fully connected layer for the value; another convolution for the policy that maps to 8×8×73 move encodings. Use PyTorch modules, batch normalization and activations (ReLU for hidden layers, log_softmax and tanh for outputs).
- [x] Unit tests: verify that the model forward pass works on synthetic data and returns correct shapes.


Phase 3 – Monte Carlo Tree Search (Weeks 3–4)
- [x] Implement the PUCT algorithm (mcts.py): create node and edge data structures, implement the selection, expansion, evaluation and back‑propagation phases. Use the neural network to obtain the prior and value at expansion.
- [x] Batch inference: design a queue where multiple MCTS workers can push states to a shared GPU inference server; implement asynchronous calls so that CPU workers do not idle while waiting for network evaluations.
- [x] Add Dirichlet noise to the root priors during self‑play to encourage exploration.


Phase 4 – Self‑play and replay buffer (Week 5)
- [ ] Self‑play loop (game.py): using your MCTS implementation, generate complete games. Sample moves from the MCTS visit distribution using a temperature parameter (>1.0 for early game, gradually annealed). Record (state, policy, value) tuples.
- [ ] Replay buffer (replay_buffer.py): implement a fixed‑size dataset that stores these tuples and supports random sampling. Ensure thread‑safety if multiple workers push to the buffer concurrently.


Phase 5 – Training loop (Week 6)
- [ ] Implement the training script (train.py): sample mini‑batches from the replay buffer, compute the loss (policy cross‑entropy plus value mean‑squared error), perform back‑propagation and update the network.
- [ ] Learning rate schedule and optimizer: start with Adam or SGD; experiment with cyclic or step decays.
- [ ] Validation: hold out a small portion of games for validation to monitor over‑fitting.


Phase 6 – Evaluation and gating (Week 7)
- [ ] Evaluator (eval.py): write an arena that pits two networks against each other using MCTS. Set a threshold for replacing the current best model (for example the new network must score >55 % over 20–50 games). Reuse the self‑play MCTS implementation.
- [ ] Metrics: log win rates, draw rates, average game length and evaluation time.


Phase 7 – Parallelization and scaling (Weeks 8–9)
- [ ] Multiprocessing: spawn multiple self‑play workers (for example 4–16) that share the neural network via shared memory; manage GPU inference requests using a central queue.
- [ ] Checkpointing: save the network and replay buffer periodically to allow resuming training and analysing intermediate strength.
- [ ] Hyperparameter tuning: adjust the number of MCTS simulations (for example 200–800), replay buffer size, batch size, learning rate and Dirichlet noise to balance exploration versus exploitation.


Phase 8 – Extended training and strength measurement (Weeks 10–12 and beyond)
- [ ] Long‑term self‑play: run your pipeline continuously for several weeks. On modest hardware this is the bottleneck; generating millions of games can take weeks. For example, a Medium article about a hobby project noted that training for one day on a single computer produced only limited improvementmedium.com. Expect to run the system for several weeks to reach around 1500–2000 Elo, depending on hardware and hyperparameters.
- [ ] Periodic evaluation: track Elo progression by playing matches against Stockfish at low search depth or open‑source engines.
- [ ] Visualization and analysis: implement scripts to visualise policy distributions, opening preferences and typical mistakes.

