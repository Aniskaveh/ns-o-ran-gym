"""Minimal DQN implementation tailored for the HO environment."""

from __future__ import annotations

# deque: Python's double-ended queue, used for efficient circular buffer
from collections import deque
import random
from typing import Deque, Iterable, List, Sequence, Tuple

import numpy as np
# PyTorch: Main library (like TensorFlow)
# nn: Neural network layers (like tf.keras.layers)
# optim: Optimizers (like tf.keras.optimizers)
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore


def _ensure_array(value: np.ndarray | Sequence[float]) -> np.ndarray:
    """
    Helper function to convert input to a flat numpy array.
    In PyTorch, we often work with numpy arrays and convert to tensors when needed.
    """
    # Convert to numpy array with float32 (PyTorch default float type)
    arr = np.asarray(value, dtype=np.float32)
    # Flatten to 1D array: reshape(-1) means "make it 1D, figure out size automatically"
    return arr.reshape(-1)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    Stores (state, action, reward, next_state, done) tuples and samples random batches.
    Similar to TensorFlow's replay buffer, but simpler - just a Python deque.
    """

    def __init__(self, capacity: int, batch_size: int) -> None:
        # Validate inputs
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        if batch_size <= 0:
            raise ValueError("ReplayBuffer batch_size must be positive.")
        # Maximum number of transitions to store
        self.capacity = capacity
        # Number of transitions to sample per training batch
        self.batch_size = batch_size
        # deque with maxlen: automatically removes oldest items when full (circular buffer)
        # Type hint: stores tuples of (state, action, reward, next_state, done)
        self._storage: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition tuple (s, a, r, s', done) in the buffer.
        In DQN, we store experiences and later sample random batches for training.
        """
        # Flatten state to 1D array (neural networks expect flat inputs)
        processed_state = _ensure_array(state)
        # Flatten next_state to 1D array
        processed_next_state = _ensure_array(next_state)
        # Convert action to int64 (PyTorch uses long/int64 for indices)
        # reshape(-1) flattens to 1D
        processed_action = np.asarray(action, dtype=np.int64).reshape(-1)
        # Append to deque (if full, oldest item is automatically removed)
        self._storage.append(
            (processed_state, processed_action, float(reward), processed_next_state, bool(done))
        )

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return len(self._storage)

    def can_sample(self) -> bool:
        """Return True when enough transitions exist for a training step."""
        return len(self._storage) >= self.batch_size

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random minibatch of transitions.
        Returns batched arrays ready for PyTorch tensor conversion.
        """
        # Check if we have enough samples
        if not self.can_sample():
            raise ValueError("Not enough samples to sample a batch.")
        # Randomly sample batch_size transitions (uniform sampling, breaks correlation)
        batch = random.sample(self._storage, self.batch_size)
        # zip(*batch) unpacks: [(s1,a1,r1,s1',d1), (s2,a2,r2,s2',d2), ...]
        #   -> [(s1,s2,...), (a1,a2,...), (r1,r2,...), (s1',s2',...), (d1,d2,...)]
        # map(np.array, ...) converts each group to numpy array
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        # Return with proper dtypes (PyTorch expects specific types)
        return (
            states.astype(np.float32),      # States: float32 (standard for neural nets)
            actions.astype(np.int64),       # Actions: int64 (for indexing)
            rewards.astype(np.float32),     # Rewards: float32
            next_states.astype(np.float32), # Next states: float32
            dones.astype(np.float32),       # Done flags: float32 (for masking)
        )


class _MLP(nn.Module):
    """
    Multi-Layer Perceptron (feed-forward neural network) for Q-function approximation.
    In PyTorch, nn.Module is like tf.keras.Model - it's the base class for all neural networks.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: Iterable[int]):
        # Call parent constructor (required for nn.Module)
        super().__init__()
        # Build layers dynamically
        layers: List[nn.Module] = []
        prev_dim = input_dim  # Start with input dimension
        # Add hidden layers: Linear -> ReLU -> Linear -> ReLU -> ...
        for hidden_dim in hidden_layers:
            # nn.Linear is like tf.keras.layers.Dense: fully connected layer
            # Input: prev_dim, Output: hidden_dim
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # ReLU activation (like tf.keras.activations.relu)
            layers.append(nn.ReLU())
            prev_dim = hidden_dim  # Update for next layer
        # Final output layer (no activation - raw Q-values)
        layers.append(nn.Linear(prev_dim, output_dim))
        # nn.Sequential chains layers together (like tf.keras.Sequential)
        # *layers unpacks the list into individual arguments
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute output given input.
        In PyTorch, you define forward() explicitly (unlike TensorFlow's __call__).
        PyTorch automatically calls backward() for gradients when you call loss.backward().
        """
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Network (DQN) agent for MultiDiscrete action spaces.
    Uses independent Q-heads: one Q-vector per action dimension, then sums them.
    """

    def __init__(self, observation_space, action_space, **hyperparameters) -> None:
        # Store Gymnasium spaces (for validation and sampling)
        self.observation_space = observation_space
        self.action_space = action_space

        # Check that action space is MultiDiscrete (has nvec attribute)
        # MultiDiscrete means: [action1, action2, ...] where each action has its own choices
        if not hasattr(action_space, "nvec"):
            raise ValueError("DQNAgent currently supports gymnasium.spaces.MultiDiscrete only.")

        # PyTorch device: "cpu" or "cuda" (GPU)
        # .to(device) moves tensors/networks to this device
        # In TensorFlow, this is handled by tf.device() context manager
        self.device = torch.device(hyperparameters.get("device", "cpu"))
        # Discount factor for future rewards (standard RL hyperparameter)
        self.gamma = float(hyperparameters.get("gamma", 0.99))
        # Learning rate for optimizer (like TensorFlow's learning_rate in Adam)
        self.learning_rate = float(hyperparameters.get("learning_rate", 1e-3))
        # Hidden layer sizes: [256, 256] means two hidden layers of 256 neurons each
        self.hidden_sizes = hyperparameters.get("hidden_sizes", [256, 256])
        # How often to copy online network to target network (DQN stability trick)
        self.target_update_freq = int(hyperparameters.get("target_update_freq", 250))
        # Gradient clipping: prevents exploding gradients (clip by norm)
        self.max_grad_norm = float(hyperparameters.get("max_grad_norm", 10.0))

        # Epsilon-greedy exploration parameters
        self.epsilon = float(hyperparameters.get("epsilon_start", 1.0))  # Start: 100% random
        self.epsilon_end = float(hyperparameters.get("epsilon_end", 0.05))  # End: 5% random
        self.epsilon_decay = float(hyperparameters.get("epsilon_decay", 0.995))  # Decay per step

        # Compute observation dimension: flatten all dimensions
        # np.prod multiplies all shape dimensions: (10, 8) -> 80
        self.obs_dim = int(np.prod(self.observation_space.shape))
        # Action dimensions: e.g., [7, 7, 7, ...] means each UE has 7 action choices
        self.action_dims = np.asarray(self.action_space.nvec, dtype=np.int64)
        # Total Q-network output size: sum of all action dimensions
        # Example: 3 UEs with 7 actions each -> 7+7+7 = 21 outputs
        self.total_action_dim = int(self.action_dims.sum())

        # Create online network (the one we train)
        # .to(device) moves network to CPU/GPU (like tf.device in TensorFlow)
        self.online_network = _MLP(self.obs_dim, self.total_action_dim, self.hidden_sizes).to(self.device)
        # Create target network (used for stable Q-targets, updated periodically)
        self.target_network = _MLP(self.obs_dim, self.total_action_dim, self.hidden_sizes).to(self.device)
        # Copy weights from online to target (start them identical)
        # load_state_dict is like model.set_weights() in TensorFlow
        self.target_network.load_state_dict(self.online_network.state_dict())
        # Set target network to evaluation mode (disables dropout, batch norm updates)
        # In PyTorch, .eval() vs .train() controls behavior (like training=False in TF)
        self.target_network.eval()

        # Adam optimizer (like tf.keras.optimizers.Adam)
        # .parameters() returns all trainable weights (like model.trainable_variables in TF)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        # Mean Squared Error loss (like tf.keras.losses.MSE)
        self.loss_fn = nn.MSELoss()

        # Track training steps for target network updates
        self._train_step = 0

    def _preprocess_obs(self, observation: np.ndarray) -> np.ndarray:
        """Flatten observation to 1D array."""
        return _ensure_array(observation)

    def _torchify(self, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.
        Similar to tf.convert_to_tensor(), but explicitly sets device and dtype.
        """
        # torch.as_tensor: convert numpy to tensor (shares memory if possible)
        # dtype: tensor data type (float32, int64, etc.)
        # device: where tensor lives (CPU or GPU)
        return torch.as_tensor(array, dtype=dtype, device=self.device)

    def _gather_q(self, q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Extract Q-values for the selected actions in a MultiDiscrete space.
        
        Q-network outputs one Q-vector per action dimension:
        - UE1: [Q(a=0), Q(a=1), ..., Q(a=6)]  (7 values)
        - UE2: [Q(a=0), Q(a=1), ..., Q(a=6)]  (7 values)
        - ...
        
        This function selects the Q-value for each UE's chosen action and sums them.
        Similar to tf.gather() but for multi-dimensional indexing.
        """
        # Initialize output: one Q-value per batch sample
        # q_values.size(0) is batch size (like shape[0] in TensorFlow)
        batch_q = torch.zeros(q_values.size(0), device=self.device)
        offset = 0  # Track position in flattened Q-vector
        # For each action dimension (each UE)
        for dim_idx, dim_size in enumerate(self.action_dims):
            # Extract Q-values for this dimension: [batch_size, dim_size]
            # Example: segment shape is [32, 7] for batch_size=32, 7 actions
            segment = q_values[:, offset : offset + dim_size]
            # Get action index for this dimension: [batch_size, 1]
            # actions[:, dim_idx] gets action for this UE, unsqueeze adds dimension
            idx = actions[:, dim_idx].unsqueeze(1)
            # Gather: select Q-value at the chosen action index
            # .gather(1, idx) is like tf.gather() - selects along dimension 1
            # .squeeze(1) removes the extra dimension
            selected = segment.gather(1, idx).squeeze(1)
            # Add to total (sum Q-values across all UEs)
            batch_q += selected
            # Move to next dimension's Q-values
            offset += dim_size
        return batch_q

    def _max_q(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        Compute max Q-value for each batch sample (for target Q-value calculation).
        For MultiDiscrete: takes max per dimension, then sums.
        Used in: target = reward + gamma * max_next_q
        """
        # Initialize output: one max-Q per batch sample
        batch_q = torch.zeros(q_values.size(0), device=self.device)
        offset = 0
        # For each action dimension
        for dim_size in self.action_dims:
            # Extract Q-values for this dimension
            segment = q_values[:, offset : offset + dim_size]
            # .max(dim=1) finds max along dimension 1 (across actions)
            # .values gets the max values (ignores indices)
            # Result: [batch_size] tensor with max Q-value for each sample
            batch_q += segment.max(dim=1).values
            offset += dim_size
        return batch_q

    def select_action(self, observation: np.ndarray, epsilon: float | None = None) -> np.ndarray:
        """
        Epsilon-greedy action selection.
        With probability epsilon: random action (exploration)
        With probability (1-epsilon): best action according to Q-network (exploitation)
        """
        # Flatten observation to 1D
        obs_vector = self._preprocess_obs(observation)
        # Use provided epsilon or current agent epsilon
        use_epsilon = self.epsilon if epsilon is None else float(epsilon)

        # Epsilon-greedy: random action with probability epsilon
        if np.random.rand() < use_epsilon:
            # Random exploration: sample from action space
            action = self.action_space.sample()
        else:
            # Exploitation: use Q-network to select best action
            # Convert observation to tensor and add batch dimension
            # .unsqueeze(0) adds dimension at position 0: [obs_dim] -> [1, obs_dim]
            # PyTorch networks expect batch dimension (like TensorFlow)
            obs_tensor = self._torchify(obs_vector, torch.float32).unsqueeze(0)
            # torch.no_grad(): disable gradient computation (faster, saves memory)
            # We don't need gradients for inference, only for training
            with torch.no_grad():
                # Forward pass: get Q-values from online network
                # Output shape: [1, total_action_dim] (batch_size=1)
                q_values = self.online_network(obs_tensor).cpu().numpy().squeeze(0)
            # .cpu() moves tensor to CPU (if it was on GPU)
            # .numpy() converts tensor to numpy array
            # .squeeze(0) removes batch dimension: [1, 21] -> [21]
            
            # Extract best action for each dimension (each UE)
            action = []
            offset = 0
            for dim_size in self.action_dims:
                # Get Q-values for this UE's actions
                segment = q_values[offset : offset + dim_size]
                # np.argmax finds index of maximum Q-value (best action)
                action.append(int(np.argmax(segment)))
                offset += dim_size
            # Convert to numpy array
            action = np.array(action, dtype=np.int64)

        # Decay epsilon if using agent's internal epsilon
        if epsilon is None:
            self._decay_epsilon()
        return np.asarray(action, dtype=np.int64)

    def _decay_epsilon(self) -> None:
        """
        Decay epsilon (reduce exploration over time).
        Multiplicative decay: epsilon *= decay_rate, but never below epsilon_end.
        """
        # Multiply by decay rate, but clamp to minimum value
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update(self, replay_buffer: ReplayBuffer) -> float | None:
        """
        Perform one DQN training step.
        
        DQN Algorithm:
        1. Sample batch of transitions from replay buffer
        2. Compute Q(s,a) from online network
        3. Compute target: r + gamma * max_a' Q_target(s',a')
        4. Update online network to minimize (Q(s,a) - target)^2
        5. Periodically copy online network to target network
        
        Returns loss value if training occurred, None otherwise.
        """
        # Check if we have enough samples
        if not replay_buffer.can_sample():
            return None

        # Sample random batch of transitions (breaks correlation)
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        # All arrays are numpy, shape: [batch_size, ...]

        # Convert numpy arrays to PyTorch tensors (move to device)
        # torch.long is int64 (used for indexing)
        states_t = self._torchify(states, torch.float32)          # [batch, obs_dim]
        actions_t = self._torchify(actions, torch.long)           # [batch, num_ues]
        rewards_t = self._torchify(rewards, torch.float32)        # [batch]
        next_states_t = self._torchify(next_states, torch.float32)  # [batch, obs_dim]
        dones_t = self._torchify(dones, torch.float32)           # [batch]

        # ===== FORWARD PASS: Compute current Q-values =====
        # Pass states through online network
        # Output: [batch_size, total_action_dim] - Q-values for all actions
        q_values = self.online_network(states_t)
        # Extract Q-values for the actions we actually took
        # current_q shape: [batch_size] - one Q-value per sample
        current_q = self._gather_q(q_values, actions_t)

        # ===== COMPUTE TARGET Q-VALUES =====
        # Use target network for stable targets (DQN trick)
        # torch.no_grad(): don't compute gradients (target network is not trained)
        with torch.no_grad():
            # Get Q-values from target network for next states
            target_q_values = self.target_network(next_states_t)
            # Find max Q-value for next states (best action in next state)
            # max_next_q shape: [batch_size]
            max_next_q = self._max_q(target_q_values)
            # Bellman equation: Q_target = r + gamma * max Q(s',a') if not done
            # (1.0 - dones_t) masks out future rewards if episode ended
            # If done=1: target = reward (no future value)
            # If done=0: target = reward + gamma * max_next_q
            targets = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        # ===== COMPUTE LOSS =====
        # Mean Squared Error: (Q(s,a) - target)^2
        # PyTorch loss functions return a tensor (unlike TF which can return scalars)
        loss = self.loss_fn(current_q, targets)

        # ===== BACKWARD PASS: Compute gradients and update weights =====
        # Zero gradients from previous step (PyTorch accumulates gradients by default)
        # In TensorFlow, gradients are computed fresh each time
        self.optimizer.zero_grad()
        # Backpropagation: compute gradients of loss w.r.t. network parameters
        # This builds the computation graph backward and computes gradients
        # Similar to tf.GradientTape in TensorFlow, but automatic
        loss.backward()
        # Gradient clipping: prevent exploding gradients (clip by norm)
        # This is a common technique for training stability
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.online_network.parameters(), self.max_grad_norm)
        # Update weights: optimizer applies gradients to parameters
        # This is like optimizer.apply_gradients() in TensorFlow
        self.optimizer.step()

        # ===== TARGET NETWORK UPDATE =====
        # Increment training step counter
        self._train_step += 1
        # Periodically copy online network weights to target network
        # This stabilizes training (target network changes slowly)
        if self._train_step % self.target_update_freq == 0:
            # Copy all weights from online to target
            self.target_network.load_state_dict(self.online_network.state_dict())

        # Return loss as Python float (for logging/monitoring)
        # .item() extracts scalar from single-element tensor
        return float(loss.item())