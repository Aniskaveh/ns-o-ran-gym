# Handover Environment - State and Action Explanation

## Overview
This document explains how the state/observation vector works in the Handover Environment for centralized DQN-based handover decision making.

## State Vector Structure

### Input: KPMs from Datalake
The `_get_obs()` method reads KPMs (Key Performance Metrics) from the datalake using:
```python
kpms = self.datalake.read_kpms(timestamp, ['nrCellId', 'L3 serving SINR', 'DRB.BufferSize.Qos.UEID'])
```

**Return format from datalake:**
- List of tuples, one per UE
- Each tuple: `(ueImsiComplete, nrCellId, L3_serving_SINR, DRB_BufferSize)`
- Example: `[(1001, 2, 15.5, 1024), (1002, 3, 18.2, 2048), ...]`

### Processing: Flattening
The method extracts features (skipping `ueImsiComplete`) and flattens them:

**For 2 UEs with 3 features each:**
```
Input:  [(1001, 2, 15.5, 1024), (1002, 3, 18.2, 2048)]
        ↓ Extract features (skip ueImsiComplete)
UE1:    [2, 15.5, 1024]
UE2:    [3, 18.2, 2048]
        ↓ Flatten
Output: [2, 15.5, 1024, 3, 18.2, 2048]
Shape:  (6,) = (2 UEs × 3 features)
```

### State Size Calculation
- **Number of UEs**: `num_ues = ues_per_gnb × num_gnbs`
  - From config: `ues = 2`, `num_gnbs = 7` → `num_ues = 14`
- **Features per UE**: `num_features = len(columns_state) = 3`
- **Total state size**: `state_size = num_ues × num_features = 14 × 3 = 42`

## DQN Model Architecture

### Input Layer
```python
Dense(128, input_dim=42)  # Receives flattened state vector of size 42
```

### Output Layer
```python
Dense(14 × 7)  # 14 UEs × 7 actions per UE = 98 Q-values
```

The output is reshaped to `(14, 7)` for easier indexing:
- Each row = one UE
- Each column = one action (0-6)

### Action Selection
```python
q_values = model.predict(state)  # Shape: (98,)
q_values_reshaped = q_values.reshape(14, 7)  # Shape: (14, 7)
action = np.argmax(q_values_reshaped, axis=1)  # Shape: (14,)
# Example: action = [1, 5, 3, 0, 2, 1, 4, 0, 3, 2, 1, 0, 5, 4]
```

## Action Interpretation

### Action Format
- **Shape**: `(num_ues,)` - one integer per UE
- **Values**: Each element ∈ [0, 6]
  - `0`: No handover
  - `1-6`: Handover to gNB (mapped to cellId 2-7 in ns3)

### Action to ns3 Mapping
```python
# In _compute_action():
for ueId, targetCellId in enumerate(action):
    if targetCellId != 0:
        # action value + 2 = actual cellId in ns3
        action_list.append((ueId + 1, targetCellId + 2))
```

**Example:**
```
Action vector: [1, 5, 3, 0, 2, ...]
              ↓
UE0: action=1 → cellId = 1+2 = 3 → handover to gNB 3
UE1: action=5 → cellId = 5+2 = 7 → handover to gNB 7
UE2: action=3 → cellId = 3+2 = 5 → handover to gNB 5
UE3: action=0 → no handover
```

## Complete Example Flow

### Scenario: 14 UEs, 3 features per UE

1. **State Construction:**
   ```
   State size = 14 × 3 = 42
   State vector = [cellId_UE0, SINR_UE0, Buffer_UE0,
                   cellId_UE1, SINR_UE1, Buffer_UE1,
                   ...
                   cellId_UE13, SINR_UE13, Buffer_UE13]
   ```

2. **DQN Forward Pass:**
   ```
   Input:  [2, 15.5, 1024, 3, 18.2, 2048, ..., ...]  # shape: (42,)
   ↓
   Hidden layers (128, 128, 64 neurons)
   ↓
   Output: [Q(UE0,0), Q(UE0,1), ..., Q(UE0,6),
           Q(UE1,0), Q(UE1,1), ..., Q(UE1,6),
           ...
           Q(UE13,0), Q(UE13,1), ..., Q(UE13,6)]  # shape: (98,)
   ```

3. **Action Selection:**
   ```
   Reshape to (14, 7):
   [[Q(UE0,0), Q(UE0,1), ..., Q(UE0,6)],
    [Q(UE1,0), Q(UE1,1), ..., Q(UE1,6)],
    ...
    [Q(UE13,0), Q(UE13,1), ..., Q(UE13,6)]]
   
   argmax for each UE:
   action = [1, 5, 3, 0, 2, 1, 4, 0, 3, 2, 1, 0, 5, 4]
   ```

4. **Action Execution:**
   ```
   Convert to ns3 format:
   ho-actions-for-ns3.csv:
   timestamp,ueId,targetCellId
   1000,1,3      # UE0 (index 0) → ueId=1, action=1 → cellId=3
   1000,2,7      # UE1 (index 1) → ueId=2, action=5 → cellId=7
   1000,3,5      # UE2 (index 2) → ueId=3, action=3 → cellId=5
   # UE3 skipped (action=0, no handover)
   ...
   ```

5. **Reward and Next State:**
   ```
   ns3 executes handovers → collects new KPMs
   → _compute_reward() calculates reward from throughput
   → _get_obs() gets next state
   → Experience stored in replay buffer
   → DQN trains on batch from buffer
   ```

## Key Points

1. **State is flattened**: All UE features concatenated into single 1D vector
2. **Centralized decision**: One model makes decisions for all UEs
3. **Action is vector**: One action per UE, all decided simultaneously
4. **State size**: `num_ues × num_features` (e.g., 14 × 3 = 42)
5. **Action size**: `num_ues` (e.g., 14)
6. **Q-values size**: `num_ues × actions_per_ue` (e.g., 14 × 7 = 98)

## Verification

To verify the state size matches your scenario:
```python
env = HandoverEnv(...)
obs, _ = env.reset()
print(f"State shape: {obs.shape}")
print(f"Expected: ({env.state_size},)")
print(f"UEs: {env.num_ues}, Features: {env.num_features_per_ue}")
```

Expected output for default config (2 UEs per gNB, 7 gNBs, 3 features):
```
State shape: (42,)
Expected: (42,)
UEs: 14, Features: 3
```
"""
        Get observation/state vector containing KPMs for all UEs.
        
        This method reads KPMs from the datalake and constructs a flattened state vector
        where all features of all UEs are concatenated together. This is suitable for
        a centralized DQN model that makes handover decisions for all UEs simultaneously.
        
        Returns:
            np.ndarray: Flattened state vector of shape (num_ues * num_features,)
            
        Example with Concrete Numbers:
            Scenario: 2 UEs, 3 features per UE
            Features: ['nrCellId', 'L3 serving SINR', 'DRB.BufferSize.Qos.UEID']
            
            Step 1: Read KPMs from datalake
                kpms = datalake.read_kpms(timestamp, ['nrCellId', 'L3 serving SINR', 'DRB.BufferSize.Qos.UEID'])
                Returns: [
                    (ueImsi1=1001, nrCellId=2, SINR=15.5, BufferSize=1024),
                    (ueImsi2=1002, nrCellId=3, SINR=18.2, BufferSize=2048)
                ]
            
            Step 2: Extract features (skip ueImsiComplete)
                UE1 features: [2, 15.5, 1024]
                UE2 features: [3, 18.2, 2048]
            
            Step 3: Flatten into single vector
                state_vector = [2, 15.5, 1024, 3, 18.2, 2048]
                Shape: (6,) = (2 UEs * 3 features)
                Type: np.ndarray, dtype=np.float32
            
            Step 4: Feed into DQN model
                model._build_model():
                    - Input layer: Dense(128, input_dim=6)  # 6 = state_size
                    - Hidden layers: Dense(128), Dense(64)
                    - Output layer: Dense(16)  # 16 = 2 UEs * 8 actions
                
                model.predict(state_vector):
                    Input:  [2, 15.5, 1024, 3, 18.2, 2048]  # shape: (6,)
                    Output: [Q(UE1,action0), Q(UE1,action1), ..., Q(UE1,action6),
                            Q(UE2,action0), Q(UE2,action1), ..., Q(UE2,action6)]  # shape: (16,)
                
                Reshape output: (16,) -> (2, 8)
                    [[Q(UE1,0), Q(UE1,1), ..., Q(UE1,7)],
                     [Q(UE2,0), Q(UE2,1), ..., Q(UE2,7)]]
                
                Choose action: argmax for each UE
                    action = [argmax(Q(UE1)), argmax(Q(UE2))]
                    Example: action = [1, 5]  # UE1->gNB1, UE2->gNB5
                
        Note:
            - The state is ordered by UE: [UE1_features, UE2_features, ..., UEN_features]
            - Each UE's features are in the order specified by self.columns_state
            - The returned array is 1D (flattened) to be compatible with standard DQN architectures
        """
