"""Graph-based RL agent (GAT encoder + DQN head) for handover decisions."""

from __future__ import annotations #tells Python: Do NOT evaluate type hints right now. Just store them as strings. Evaluate them later only when needed.
import random
from collections import deque  #A double-ended queue. Used for efficient circular buffer. q = deque(), q.append(1), q.appendleft(0), q.popleft()
from dataclasses import dataclass #A decorator that automatically generates: __init__, ... For simple classes used for storing data.This eliminates manual repetitive code.
from typing import Deque, Dict, List, Optional, Tuple #These are type hint tools used to describe what kind of data your variables hold
                                                      #optional: Means the value can be: a specific type or None
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn #Neural network layers.
import torch.optim as optim #Optimizers.
# torch_geometric is for GNN, allows representing UE ↔ gNB network as a graph, capturing both node features and edge features
from torch_geometric.data import Data #Data class for storing graph data.
from torch_geometric.nn import GATConv #Graph Attention Network layer.
from torch_geometric.utils import softmax, scatter #softmax: Normalizes attention weights to sum to 1. scatter: Aggregates neighbor features.


@dataclass
class GraphTransition: #needed for replay buffer in DQN training.
    """Container for one graph transition used in replay."""
    #Store one single experience: (graph_state, action_vector, reward, next_graph_state, done)

    state: Dict[str, np.ndarray] #graph_state: Dictionary containing UE and gNB features, edge_index, and edge_attr
    action: np.ndarray #action_vector: Array of action indices for each UE
    reward: float #reward: Scalar reward for the transition
    next_state: Dict[str, np.ndarray] #next_graph_state: Dictionary containing next UE and gNB features, edge_index, and edge_attr
    done: bool #done: Boolean indicating if the episode is done

# ****************************Edge-aware Graph Attention****************************
# Standard GATConv only uses node features. Edge attributes are critical for HO decisions.
# Extends standard GAT by including edge features in the attention mechanism:
# e_ij = LeakyReLU(a^T [W·h_i || W·h_j || U·e_ij])
# This preserves edge-specific information (distance, is_serving_cell, sinr_to_gNB)
# without information loss from aggregation.
# Each edge's attention weight depends on both node features AND edge features.
# Use normalized edge attributes
# h = torch.relu(self.egat1(node_h, edge_index, edge_attr_norm))
# h = torch.relu(self.egat2(h, edge_index, edge_attr_norm))
class EGATConv(nn.Module): #EGAT layer updates each node's features by looking at its neighbors, 
                        #weighting them using attention scores where the attention depends on: node features, target node features, and edge features.
    # This class defines one GAT layer that takes node features, look at neighbors and outputs new node features
    def __init__(self, in_channels: int, out_channels: int, edge_attr_dim: int, heads: int = 1, 
                 concat: bool = False, negative_slope: float = 0.2, dropout: float = 0.0, 
                 bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels #size of input node features vector
        self.out_channels = out_channels #size of output per head
        self.edge_attr_dim = edge_attr_dim #size of edge vector
        self.heads = heads #number of attention heads, each head will learn a different attention pattern.
        self.concat = concat #if True, concatenate heads' outputs instead of averaging.
        self.negative_slope = negative_slope #slope for LeakyReLU activation, default 0.2, negative inputs are multiplied by 0.2 instead of becoming 0
        self.dropout = dropout #dropout rate, default 0.0, no dropout
        
        # --------- Raw features → hidden embeddings ---------
        # Node feature transformation: [in_channels] -> [heads * out_channels]
        # Why multiply by heads? each head needs its own feature space
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        # Edge feature transformation: [edge_attr_dim] -> [heads * out_channels]
        self.edge_lin = nn.Linear(edge_attr_dim, heads * out_channels, bias=False)
        
        # Attention mechanism: computes attention weights
        # why 3* out_channels? cause: Input: [source_node_features || target_node_features || edge_features]
        # Output: attention score for each head
        self.att = nn.Parameter(torch.empty(1, heads, 3 * out_channels)) #This is the attention vector a
        
        if bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.edge_lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with edge-aware attention.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge attributes [E, edge_attr_dim]
        
        Returns:
            Updated node features [N, heads * out_channels] or [N, out_channels]
        """
        # Transform node features: [N, in_channels] -> [N, heads * out_channels]
        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)  # [N, heads, out_channels]
        
        # Transform edge features: [E, edge_attr_dim] -> [E, heads * out_channels]
        edge_attr_emb = self.edge_lin(edge_attr)
        edge_attr_emb = edge_attr_emb.view(-1, self.heads, self.out_channels)  # [E, heads, out_channels]
        
        # Prepare source and target node features for each edge
        row, col = edge_index
        x_i = x[row]  # Source node features [E, heads, out_channels]
        x_j = x[col]  # Target node features [E, heads, out_channels]
        # Attention score computation
        # Concatenate: [source_features || target_features || edge_features]
        # Shape: [E, heads, 3 * out_channels]
        alpha_input = torch.cat([x_i, x_j, edge_attr_emb], dim=-1)
        
        # Compute attention scores: [E, heads]
        alpha = (alpha_input * self.att).sum(dim=-1)  # [E, heads]
        alpha = torch.nn.functional.leaky_relu(alpha, self.negative_slope)
        
        # Attention normalization
        alpha = softmax(alpha, row, num_nodes=x.size(0))  # [E, heads]
        
        # Apply dropout to attention weights
        if self.training and self.dropout > 0:
            alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=True)
        
        # Aggregate neighbor features weighted by attention
        # For each edge, multiply target node features by attention weight
        # Message passing: each neighbor's message is weighted by attention
        out = x_j * alpha.unsqueeze(-1)  # [E, heads, out_channels]
        
        # Aggregate messages: sum over neighbors for each node
        # Reshape for scatter: [E, heads, out_channels] -> [E, heads * out_channels]
        out_reshaped = out.view(-1, self.heads * self.out_channels)  # [E, heads * out_channels]
        # Aggregate neighbors. scatter is a PyTorch function that aggregates messages from neighbors to the target node.
        out = scatter(out_reshaped, row, dim=0, dim_size=x.size(0), reduce='add')  # [N, heads * out_channels]
        out = out.view(-1, self.heads, self.out_channels)  # [N, heads, out_channels]
        
        # Concatenate or average heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)  # [N, out_channels]
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GraphReplayBuffer:
    """Simple replay buffer that keeps graph observations."""

    def __init__(self, capacity: int, batch_size: int) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self.storage: Deque[GraphTransition] = deque(maxlen=capacity)

    def store(self, transition: GraphTransition) -> None:
        self.storage.append(transition)

    def __len__(self) -> int:
        return len(self.storage)

    def can_sample(self) -> bool:
        return len(self.storage) >= self.batch_size

    def sample(self) -> List[GraphTransition]:
        return random.sample(self.storage, self.batch_size)


class FeatureNormalizer(nn.Module):
    """
    Learnable feature normalization layer.
    Normalizes features: (x - shift) / scale
    where shift and scale are learnable parameters.
    
    Can be initialized with statistics for better starting point.
    """
    def __init__(self, feat_dim: int, init_scale: Optional[torch.Tensor] = None, 
                 init_shift: Optional[torch.Tensor] = None, eps: float = 1e-8):
        super().__init__()
        self.feat_dim = feat_dim
        self.eps = eps
        
        # Initialize scale: if provided use it, otherwise use 1.0 (will learn)
        if init_scale is not None:
            if init_scale.numel() == 1:
                init_scale = torch.full((feat_dim,), float(init_scale)) #Expands scalar → vector of length feat_dim
            self.scale = nn.Parameter(init_scale.clone())
        else:
            self.scale = nn.Parameter(torch.ones(feat_dim))
        
        # Initialize shift: if provided use it, otherwise use 0.0 (will learn)
        if init_shift is not None:
            if init_shift.numel() == 1:
                init_shift = torch.full((feat_dim,), float(init_shift))
            self.shift = nn.Parameter(init_shift.clone()) # we declare shift and scale as learnable parameters in nn .
        else:
            self.shift = nn.Parameter(torch.zeros(feat_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) / (self.scale.abs() + self.eps) #why abs? prevents negative scaling


class GraphQNetwork(nn.Module):
    """GAT encoder over a bipartite UE↔gNB graph with a DQN head."""

    def __init__(self, ue_feat_dim: int, gnb_feat_dim: int, hidden_dim: int, num_actions: int, gat_heads: int = 2, edge_attr_dim: int = 3, 
                 ue_init_scale: Optional[torch.Tensor] = None, ue_init_shift: Optional[torch.Tensor] = None,
                 gnb_init_scale: Optional[torch.Tensor] = None, gnb_init_shift: Optional[torch.Tensor] = None,
                 edge_init_scale: Optional[torch.Tensor] = None, edge_init_shift: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        
        # Feature normalization layers to normalize inputs before encoding
        # This ensures all features are on similar scales for better training
        # Can be initialized with statistics for immediate normalization
        self.ue_normalizer = FeatureNormalizer(ue_feat_dim, init_scale=ue_init_scale, init_shift=ue_init_shift)
        self.gnb_normalizer = FeatureNormalizer(gnb_feat_dim, init_scale=gnb_init_scale, init_shift=gnb_init_shift)
        self.edge_normalizer = FeatureNormalizer(edge_attr_dim, init_scale=edge_init_scale, init_shift=edge_init_shift)
        
        # Separate encoders for UE and gNB node features so we can project them
        # into a common hidden space.
        self.ue_encoder = nn.Linear(ue_feat_dim, hidden_dim)
        self.gnb_encoder = nn.Linear(gnb_feat_dim, hidden_dim)

        # Two EGAT layers that incorporate edge attributes directly into attention
        # EGAT preserves edge-specific information (distance, is_serving_cell, sinr_to_gNB)
        # without information loss from aggregation
        self.egat1 = EGATConv(hidden_dim, hidden_dim, edge_attr_dim=edge_attr_dim, 
                               heads=gat_heads, concat=False)
        self.egat2 = EGATConv(hidden_dim, hidden_dim, edge_attr_dim=edge_attr_dim, 
                               heads=gat_heads, concat=False)

        # Deeper Q-head: sequential layers for better action value estimation
        # q_head → MLP maps UE embeddings to Q-values per action.
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

    def normalize_features(self, ue_feats: torch.Tensor, gnb_feats: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize features and return them along with statistics.
        Useful for visualization and debugging.
        """
        ue_feats_norm = self.ue_normalizer(ue_feats)
        gnb_feats_norm = self.gnb_normalizer(gnb_feats)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        elif edge_attr.dim() > 2:
            edge_attr = edge_attr.squeeze()
        edge_attr_norm = self.edge_normalizer(edge_attr)
        return ue_feats_norm, gnb_feats_norm, edge_attr_norm
    
    def get_normalization_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get current normalization parameters (scale and shift) for inspection.
        Returns a dictionary with 'ue', 'gnb', 'edge' keys, each containing 'scale' and 'shift'.
        """
        return {
            'ue': {
                'scale': self.ue_normalizer.scale.data.clone(),
                'shift': self.ue_normalizer.shift.data.clone()
            },
            'gnb': {
                'scale': self.gnb_normalizer.scale.data.clone(),
                'shift': self.gnb_normalizer.shift.data.clone()
            },
            'edge': {
                'scale': self.edge_normalizer.scale.data.clone(),
                'shift': self.edge_normalizer.shift.data.clone()
            }
        }

    def forward(self, ue_feats: torch.Tensor, gnb_feats: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Normalize features before encoding to ensure similar scales
        # This prevents features with large values (e.g., throughput) from dominating
        # features with small values (e.g., SINR, numActiveUes)
        ue_feats_norm = self.ue_normalizer(ue_feats)
        gnb_feats_norm = self.gnb_normalizer(gnb_feats)
        
        # Encode UEs and gNBs separately then stack into a single node matrix.
        #The small "encoders" in the code (Linear + ReLU) just map raw features of each node type (UE/gNB) into a common hidden dimension. 
        # They do not mix information between nodes.
        #encoders = per-node feature projection; 
        ue_h = torch.relu(self.ue_encoder(ue_feats_norm))
        gnb_h = torch.relu(self.gnb_encoder(gnb_feats_norm))
        node_h = torch.cat([ue_h, gnb_h], dim=0)  # [N_total, hidden]

        # Ensure edge_attr has correct shape [E, edge_attr_dim]
        if edge_attr is None:
            raise ValueError("EGAT requires edge_attr to be provided")
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        elif edge_attr.dim() > 2:
            edge_attr = edge_attr.squeeze()
        
        # Normalize edge attributes as well
        edge_attr_norm = self.edge_normalizer(edge_attr)

        # Run EGAT over the combined graph with edge-aware attention.
        # EGAT layers incorporate edge attributes directly into attention computation:
        # e_ij = LeakyReLU(a^T [W·h_i || W·h_j || U·e_ij])
        # This preserves edge-specific information (distance, is_serving_cell, sinr_to_gNB)
        # without information loss from aggregation.
        # Each edge's attention weight depends on both node features AND edge features.
        # Use normalized edge attributes
        h = torch.relu(self.egat1(node_h, edge_index, edge_attr_norm))
        h = torch.relu(self.egat2(h, edge_index, edge_attr_norm))
        #Think of edge_index as a two-row table of integer node IDs that lists all directed edges in the graph. 
        # Each column is one edge: the top row is the source node, the bottom row is the destination node. 
        # That’s all “COO format” means here. example (3 UEs, 2 gNBs, fully connecting each UE to each gNB):
        #We number UE nodes first: UE0→0, UE1→1, UE2→2.
        #Then gNB nodes: gNB0→3, gNB1→4 (offset by n_ue = 3).
        #All directed edges UE→gNB are:
        #(UE0→gNB0), (UE0→gNB1), (UE1→gNB0), (UE1→gNB1), (UE2→gNB0), (UE2→gNB1).
        #Put them into a 2×E array:
        #edge_index =[[0, 0, 1, 1, 2, 2],   # sources [3, 4, 3, 4, 3, 4]]   # destinatio

        # Slice back the UE nodes (first n_ue rows).
        n_ue = ue_feats.size(0)
        ue_h = h[:n_ue]

        # Compute Q-values per UE: shape [n_ue, num_actions].
        #GAT alone produces contextualized node embeddings; it doesn’t, by itself, define a control policy.
        # You still need a head that maps embeddings to action values or action probabilities.
        q_values = self.q_head(ue_h)
        return q_values


class GRLAgent:
    """GAT + DQN-style agent for MultiDiscrete HO actions."""

    def __init__(self, action_space: spaces.MultiDiscrete, ue_feat_dim: int, gnb_feat_dim: int, device: str = "cpu", gamma: float = 0.99, learning_rate: float = 1e-3, hidden_dim: int = 128, target_update_freq: int = 200, epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: float = 0.995, max_grad_norm: float = 10.0, edge_attr_dim: int = 3,
                 ue_init_scale: Optional[torch.Tensor] = None, ue_init_shift: Optional[torch.Tensor] = None,
                 gnb_init_scale: Optional[torch.Tensor] = None, gnb_init_shift: Optional[torch.Tensor] = None,
                 edge_init_scale: Optional[torch.Tensor] = None, edge_init_shift: Optional[torch.Tensor] = None) -> None:
        if not hasattr(action_space, "nvec"):
            raise ValueError("GRLAgent expects a MultiDiscrete action space.")

        self.device = torch.device(device) #is a PyTorch object that tells PyTorch where (on which hardware) tensors and models should live, "cpu" or "cuda" (GPU)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.max_grad_norm = max_grad_norm
        # Gradient clipping: prevents exploding gradients (clip by norm)
        # This is a common technique for training stability
        # Epsilon-greedy parameters.
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Action layout: one action dimension per UE.
        self.action_dims = np.asarray(action_space.nvec, dtype=np.int64)
        self.num_actions = int(self.action_dims[0])  # assume same for all UEs
        self.n_ue = len(self.action_dims)

        # Online / target networks.
        self.online_net = GraphQNetwork(ue_feat_dim=ue_feat_dim,
                                        gnb_feat_dim=gnb_feat_dim,
                                        hidden_dim=hidden_dim,
                                        num_actions=self.num_actions,
                                        edge_attr_dim=edge_attr_dim,
                                        ue_init_scale=ue_init_scale, ue_init_shift=ue_init_shift,
                                        gnb_init_scale=gnb_init_scale, gnb_init_shift=gnb_init_shift,
                                        edge_init_scale=edge_init_scale, edge_init_shift=edge_init_shift).to(self.device)
        self.target_net = GraphQNetwork(ue_feat_dim=ue_feat_dim,
                                        gnb_feat_dim=gnb_feat_dim,
                                        hidden_dim=hidden_dim,
                                        num_actions=self.num_actions,
                                        edge_attr_dim=edge_attr_dim,
                                        ue_init_scale=ue_init_scale, ue_init_shift=ue_init_shift,
                                        gnb_init_scale=gnb_init_scale, gnb_init_shift=gnb_init_shift,
                                        edge_init_scale=edge_init_scale, edge_init_shift=edge_init_shift).to(self.device)
        # Copy weights from online to target (start them identical)
        # load_state_dict is like model.set_weights() in TensorFlow
        self.target_net.load_state_dict(self.online_net.state_dict())
        # Set target network to evaluation mode (disables dropout, batch norm updates)
        # In PyTorch, .eval() vs .train() controls behavior (like training=False in TF)
        self.target_net.eval()
        #Adam optimizer updates:GAT weights, Q-head weights, FeatureNormalizer scale & shift
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.train_step = 0

    # --- Utility: convert numpy dict to torch tensors on device ---
    def _to_tensors(self, graph_obs: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ue = torch.as_tensor(graph_obs["ue_features"], dtype=torch.float32, device=self.device)
        gnb = torch.as_tensor(graph_obs["gnb_features"], dtype=torch.float32, device=self.device)
        edge_index = torch.as_tensor(graph_obs["edge_index"], dtype=torch.long, device=self.device)
        edge_attr = None
        if "edge_attr" in graph_obs and graph_obs["edge_attr"].size > 0:
            edge_attr = torch.as_tensor(graph_obs["edge_attr"], dtype=torch.float32, device=self.device)
            # GATConv expects edge_attr shape [E, edge_attr_dim], squeeze if needed
            if edge_attr.dim() > 2:
                edge_attr = edge_attr.squeeze()
        return ue, gnb, edge_index, edge_attr

    # --- Action selection ---
    def select_action(self, graph_obs: Dict[str, np.ndarray]) -> np.ndarray:
        ue, gnb, edge_index, edge_attr = self._to_tensors(graph_obs)
        
        # Extract action mask: for each UE, which action to mask (-1 = no mask)
        action_mask = graph_obs.get("action_mask", None)
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.long, device=self.device)
        else:
            # Fallback: no masking if mask not provided
            action_mask = torch.full((self.n_ue,), -1, dtype=torch.long, device=self.device)
        if random.random() < self.epsilon:
            # Pure exploration: sample from valid actions (exclude masked action)
            print("Exploration")
            actions = []
            for u_idx in range(self.n_ue):
                masked_action = int(action_mask[u_idx].item()) if action_mask[u_idx] >= 0 else -1
                # Create list of valid actions (exclude masked action)
                valid_actions = [a for a in range(self.num_actions) if a != masked_action]
                if len(valid_actions) > 0:
                    actions.append(random.choice(valid_actions))
                else:
                    # Fallback: if all actions masked (shouldn't happen), use action 0 (no HO)
                    actions.append(0)
            return np.array(actions, dtype=np.int64), np.array(action_mask, dtype=np.int64)

        with torch.no_grad():
            print("Exploitation")
            q = self.online_net(ue, gnb, edge_index, edge_attr=edge_attr)  # [n_ue, num_actions]
            
            # Mask invalid actions (same as serving cell) by setting Q-value to -inf
            for u_idx in range(self.n_ue):
                masked_action = int(action_mask[u_idx].item()) if action_mask[u_idx] >= 0 else -1
                if masked_action >= 0 and masked_action < self.num_actions:
                    q[u_idx, masked_action] = float('-inf')
            
            action = torch.argmax(q, dim=1).cpu().numpy().astype(np.int64)
        return action, np.array(action_mask, dtype=np.int64)

    # --- Training step ---
    def update(self, buffer: GraphReplayBuffer) -> Optional[float]:
        if not buffer.can_sample():
            return None

        batch = buffer.sample()

        losses = []
        for transition in batch:
            ue, gnb, edge_index, edge_attr = self._to_tensors(transition.state)
            next_ue, next_gnb, next_edge_index, next_edge_attr = self._to_tensors(transition.next_state)

            actions = torch.as_tensor(transition.action, dtype=torch.int64, device=self.device)
            reward = torch.tensor(transition.reward, dtype=torch.float32, device=self.device)
            done = torch.tensor(transition.done, dtype=torch.float32, device=self.device)

            # Q(s,a)
            q_values = self.online_net(ue, gnb, edge_index, edge_attr=edge_attr)  # [n_ue, num_actions]
            q_selected = q_values.gather(1, actions.view(-1, 1)).squeeze(1).sum()

            # max_a' Q_target(s', a')
            with torch.no_grad():
                next_q = self.target_net(next_ue, next_gnb, next_edge_index, edge_attr=next_edge_attr)
                next_q_max = next_q.max(dim=1).values.sum()
                target = reward + (1.0 - done) * self.gamma * next_q_max

            loss = self.loss_fn(q_selected, target)
            losses.append(loss)

        # Backprop (average over batch)
        batch_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Update target network periodically.
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Decay epsilon.
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(batch_loss.item())


