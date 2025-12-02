from nsoran.ns_env import NsOranEnv 
from typing_extensions import override 
import logging
import numpy as np
import pandas as pd
import glob
import csv
import os

#----------------------------------------------------------------------------------------
#                                    Child Class                                        |
#----------------------------------------------------------------------------------------

class HandoverEnv(NsOranEnv):
    """Child class for Handover use case inherited from parent class (NsOranEnv) """
    #----------------------------------------------------------------------------------------
    #                                 Attributes                                            |
    #----------------------------------------------------------------------------------------
    ns3_path:str
    scenario_configuration:dict
    output_folder:str
    optimized:bool
    verbose:bool #Environment specific parameters: verbose (bool): enables logging

    #----------------------------------------------------------------------------------------
    #                                   Methods                                             |
    #----------------------------------------------------------------------------------------
    # ---------------------> Constructor method
    def __init__(self, ns3_path:str=None, scenario_configuration:dict=None, output_folder:str=None, optimized:bool=None, verbose:bool=True):
        super().__init__(ns3_path = ns3_path, scenario = 'scenario-dc', scenario_configuration = scenario_configuration,
                         output_folder = output_folder, optimized = optimized,
                         control_header = ['timestamp','ueId','nrCellId'], log_file='HoActions.txt', control_file='ts_actions_for_ns3.csv')
        
        n_gnbs = 7   
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(filename='./reward_ho.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')
        
        # Initialize handover tracking
        self.handovers_dict = {}  # Track last handover time per UE
        self.previous_kpms = None  # Track previous state for handover detection
        self.num_steps = 0  # Step counter for monitoring
        
        #################### State columns (needed KPMs) ####################
        # SIMPLIFIED VERSION: Only using UE throughput as state
        # Complex version (commented out):
        '''self.columns_state = [
                # === Per-UE Metrics ===
                'DRB.UEThpDl.UEID',                    # UE throughput
                'DRB.BufferSize.Qos.UEID',             # UE buffer size
                'L3 serving SINR',                     # Serving cell SINR
                'L3 serving SINR 3gpp',                # Serving cell SINR (3GPP encoded)
                'nrCellId',                            # Current serving cell ID
                # === All Neighbor Information (will be sorted in _get_obs) ===
                'L3 neigh SINR 1', 'L3 neigh SINR 2', 'L3 neigh SINR 3', 
                'L3 neigh SINR 4', 'L3 neigh SINR 5', 'L3 neigh SINR 6',
                'L3 neigh Id 1 (cellId)', 'L3 neigh Id 2 (cellId)', 'L3 neigh Id 3 (cellId)',
                'L3 neigh Id 4 (cellId)', 'L3 neigh Id 5 (cellId)', 'L3 neigh Id 6 (cellId)',
                # === Per-Cell Load Metrics ===
                'dlPrbUsage', 'dlAvailablePrbs', 'DRB.MeanActiveUeDl', 'RRU.PrbUsedDl',
                'TB.TotNbrDl.1.UEID', 'QosFlow.PdcpPduVolumeDL_Filter.UEID'
        ]'''
        self.columns_state = ['DRB.UEThpDl.UEID']
        #################### Reward columns ####################
        # SIMPLIFIED VERSION: Only using UE throughput for reward calculation
        # Complex version (commented out):
        #self.columns_reward = ['DRB.UEThpDl.UEID', 'nrCellId', 'DRB.BufferSize.Qos.UEID', 'TB.ErrTotalNbrDl.1.UEID']
        self.columns_reward = ['DRB.UEThpDl.UEID']
    # Implement Codes into the defined Abstract methods of parent class (NsOranEnv)  
    @override
    def _compute_action(self, action) -> list[tuple]:
        # converts gym actions into a readable format for ns3
        # action form shall become a list of ueId, targetCell.
        # If a targetCell is 0, it means No Handover, thus we don't send it
        action_list = []
        #suppose action = [1,5,3,0,...] so enumerate(action) gives us: (0,1) (1,50) (2,3) (3,0) ...
        for ueId, targetCellId in enumerate(action):
            if targetCellId != 0: 
                # Once we are in this condition, we need to transform the action from the one of gym to the one of ns-O-RAN
                action_list.append((ueId + 1, targetCellId + 2)) #targetCellId + 2 cause cellID starts from 2-8
        if self.verbose:
            logging.debug(f'Action list {action_list}')
        return action_list
    
    '''@override
    def _get_obs(self) -> list:
        # gets required KPMs from our ns3-based ENV to feed it to agents
        # Get all neighbor SINR data
        neighbor_sinr_columns = ['L3 neigh SINR 1', 'L3 neigh SINR 2', 'L3 neigh SINR 3', 
                                 'L3 neigh SINR 4', 'L3 neigh SINR 5', 'L3 neigh SINR 6']
        neighbor_id_columns = ['L3 neigh Id 1 (cellId)', 'L3 neigh Id 2 (cellId)', 'L3 neigh Id 3 (cellId)',
                               'L3 neigh Id 4 (cellId)', 'L3 neigh Id 5 (cellId)', 'L3 neigh Id 6 (cellId)']
        
        # Get basic state KPMs
        basic_state_columns = ['DRB.UEThpDl.UEID', 'DRB.BufferSize.Qos.UEID', 'L3 serving SINR', 
                                'L3 serving SINR 3gpp', 'nrCellId', 'dlPrbUsage', 'dlAvailablePrbs',
                                'DRB.MeanActiveUeDl', 'RRU.PrbUsedDl', 'TB.TotNbrDl.1.UEID',
                                'QosFlow.PdcpPduVolumeDL_Filter.UEID']
    
        all_columns = basic_state_columns + neighbor_sinr_columns + neighbor_id_columns
        kpms = self.datalake.read_kpms(self.last_timestamp, all_columns)
        
        # Process each UE's data
        state = []
        for ue_data in kpms:
            # Extract basic metrics
            basic_metrics = ue_data[:len(basic_state_columns)]
            
            # Extract neighbor SINRs and IDs
            neighbor_sinrs = ue_data[len(basic_state_columns):len(basic_state_columns)+6]
            neighbor_ids = ue_data[len(basic_state_columns)+6:]
            
            # Create list of (SINR, cellId) pairs, filtering out invalid entries
            valid_neighbors = []
            for i, (sinr, cell_id) in enumerate(zip(neighbor_sinrs, neighbor_ids)):
                if sinr is not None and cell_id is not None and not np.isnan(sinr):
                    valid_neighbors.append((sinr, cell_id))
            
            # Sort by SINR strength (descending - highest SINR first)
            valid_neighbors.sort(key=lambda x: x[0], reverse=True)
            
            # Take top 3 neighbors (or fewer if less than 3 available)
            top_3_neighbors = valid_neighbors[:3]
            
            # Pad with zeros if less than 3 neighbors
            while len(top_3_neighbors) < 3:
                top_3_neighbors.append((0.0, 0))  # (SINR=0, cellId=0 for no neighbor)
            
            # Extract SINRs and cell IDs for top 3
            top_3_sinrs = [neighbor[0] for neighbor in top_3_neighbors]
            top_3_cell_ids = [neighbor[1] for neighbor in top_3_neighbors]
            
            # Combine all metrics
            ue_observation = list(basic_metrics) + top_3_sinrs + top_3_cell_ids
            state.append(ue_observation)
        
        return np.array(state)'''

    @override
    def _get_obs(self) -> list:
        """
        Simplified observation function - returns only UE throughput
        """
        # Get only UE throughput as state
        kpms = self.datalake.read_kpms(self.last_timestamp, self.columns_state)
        
        # Data structure: [(ueImsiComplete, throughput), ...]
        # Extract only throughput values
        throughput_values = []
        for kpm in kpms:
            ueImsi, throughput = kpm
            throughput_values.append(throughput)
        
        return np.array(throughput_values)
    
    '''@override
    def _compute_reward(self):
        """
        Compute reward based on handover optimization objectives:
        1. Maximize overall throughput (sum of all UEs)
        2. Minimize handover cost (penalize frequent handovers)
        3. Minimize packet loss (penalize buffer size and transport block errors)
        """
        # Get current KPMs for reward calculation
        # columns_reward = ['DRB.UEThpDl.UEID', 'nrCellId', 'DRB.BufferSize.Qos.UEID', 'TB.ErrTotalNbrDl.1.UEID']
        # Returns: [(ueImsiComplete, throughput, cellId, bufferSize, errors), ...]
        current_kpms = self.datalake.read_kpms(self.last_timestamp, self.columns_reward)
        
        # If this is the first iteration, initialize previous state
        if self.previous_kpms is None:
            self.previous_kpms = current_kpms
            return 0.0
        
        # Validate data structure
        if current_kpms is None or len(current_kpms) == 0:
            if self.verbose:
                logging.warning("No KPMs data available for reward calculation")
            return 0.0
        
        # Check if data structure is as expected (5 elements per tuple)
        if len(current_kpms[0]) != 5:
            if self.verbose:
                logging.error(f"Unexpected data structure: expected 5 elements, got {len(current_kpms[0])}")
            return 0.0
        
        total_reward = 0.0
        total_throughput = 0.0
        total_handover_cost = 0.0
        total_buffer_size = 0.0
        total_errors = 0.0
        total_packet_loss_penalty = 0.0
        handover_count = 0
        
        # Process each UE's data
        # Data structure: (ueImsiComplete, DRB.UEThpDl.UEID, nrCellId, DRB.BufferSize.Qos.UEID, TB.ErrTotalNbrDl.1.UEID)
        for t_prev, t_curr in zip(self.previous_kpms, current_kpms):
            ueImsi_prev, thp_prev, cell_prev, buffer_prev, errors_prev = t_prev
            ueImsi_curr, thp_curr, cell_curr, buffer_curr, errors_curr = t_curr
            
            if ueImsi_prev == ueImsi_curr:  # Same UE
                # 1. THROUGHPUT REWARD (Maximize)
                # Use logarithmic throughput to prevent dominance by high-throughput UEs
                thp_prev_safe = max(thp_prev, 1e-6)  # Avoid log(0)
                thp_curr_safe = max(thp_curr, 1e-6)
                
                throughput_reward = np.log10(thp_curr_safe) - np.log10(thp_prev_safe)
                total_throughput += thp_curr
                
                # 2. HANDOVER COST (Minimize)
                handover_cost = 0.0
                if cell_curr != cell_prev:  # Handover detected
                    handover_count += 1
                    last_ho_time = self.handovers_dict.get(ueImsi_curr, 0)
                    
                    if last_ho_time != 0:  # Not the first handover
                        # Exponential decay cost: higher cost for frequent handovers
                        time_since_last_ho = (self.last_timestamp - last_ho_time) * 0.001  # Convert ms to seconds
                        handover_cost = 1.0 * ((1 - 0.1) ** time_since_last_ho)  # Decay factor = 0.1
                    
                    # Update last handover time
                    self.handovers_dict[ueImsi_curr] = self.last_timestamp
                
                total_handover_cost += handover_cost
                
                # 3. PACKET LOSS PENALTY (Minimize)
                # Buffer size penalty (higher buffer = more congestion/latency)
                buffer_penalty = 0.01 * (buffer_curr / 1000)  # Normalize buffer size
                
                # Transport block error penalty (direct packet loss indicator)
                error_penalty = 0.1 * errors_curr  # Penalize each error
                
                packet_loss_penalty = buffer_penalty + error_penalty
                
                # Accumulate totals for monitoring
                total_buffer_size += buffer_curr
                total_errors += errors_curr
                total_packet_loss_penalty += packet_loss_penalty
                
                # UE-level reward
                ue_reward = throughput_reward - handover_cost - packet_loss_penalty
                total_reward += ue_reward
                
                if self.verbose:
                    logging.debug(f"UE {ueImsi_curr}: thp_reward={throughput_reward:.4f}, ho_cost={handover_cost:.4f}, "
                                 f"buffer_penalty={buffer_penalty:.4f}, error_penalty={error_penalty:.4f}, ue_reward={ue_reward:.4f}")
        
        # Update previous state and increment step counter
        self.previous_kpms = current_kpms
        self.num_steps += 1
        
        # Logging for monitoring
        if self.verbose:
            logging.debug(f"Total reward: {total_reward:.4f}, Total throughput: {total_throughput:.2f}, "
                         f"Total HO cost: {total_handover_cost:.4f}, HO count: {handover_count}, "
                         f"Total buffer: {total_buffer_size:.2f}, Total errors: {total_errors:.0f}, "
                         f"Packet loss penalty: {total_packet_loss_penalty:.4f}")
        
        # Update Grafana table for monitoring
        db_row = {
            'timestamp': self.last_timestamp,
            'ueImsiComplete': None,
            'time_grafana': self.last_timestamp,
            'step': getattr(self, 'num_steps', 0),
            'total_throughput': total_throughput / 1e6,  # Convert to Mbps
            'handover_cost': total_handover_cost,
            'handover_count': handover_count,
            'total_buffer_size': total_buffer_size,
            'total_errors': total_errors,
            'packet_loss_penalty': total_packet_loss_penalty,
            'reward': total_reward
        }
        
        # Insert into datalake
        self.datalake.insert_data("grafana", db_row)
        
        return total_reward'''

    @override
    def _compute_reward(self):
        """
        Simplified reward function - maximizes overall throughput only
        """
        # Get UE throughput data
        kpms = self.datalake.read_kpms(self.last_timestamp, self.columns_reward)
        
        if kpms is None or len(kpms) == 0:
            if self.verbose:
                logging.warning("No KPMs data available for reward calculation")
            return 0.0
        
        # Calculate total throughput reward
        total_throughput = 0.0
        total_reward = 0.0
        
        for kpm in kpms:
            ueImsi, throughput = kpm
            # Use logarithmic throughput to prevent dominance by high-throughput UEs
            thp_safe = max(throughput, 1e-6)  # Avoid log(0)
            throughput_reward = np.log10(thp_safe)
            total_throughput += throughput
            total_reward += throughput_reward
        
        # Increment step counter
        self.num_steps += 1
        
        # Logging for monitoring
        if self.verbose:
            logging.debug(f"Total reward: {total_reward:.4f}, Total throughput: {total_throughput:.2f}")
        
        # Update Grafana table for monitoring (simplified)
        # System-Wide Data (Aggregated metrics), for this reason we don't need ueImsiComplete
        # otherwise we would need to aggregate the metrics for each UE then: 'ueImsiComplete': ueImsi
        db_row = {
            'timestamp': self.last_timestamp,
            'ueImsiComplete': None,
            'time_grafana': self.last_timestamp,
            'step': self.num_steps,
            'total_throughput': total_throughput / 1e6,  # Convert to Mbps
            'reward': total_reward
        }
        
        # Insert into datalake
        self.datalake.insert_data("grafana", db_row)
        
        return total_reward
    
    @override
    def _init_datalake_usecase(self):
        # initialize additional tables to store use case data
        # reply_buffer table
        buffer_keys = {"timestamp": "INTEGER",
                        "step": "INTEGER",
                        "ueImsiComplete": "INTEGER",
                        "state": "REAL",
                        "action": "REAL",
                        "reward": "REAL",
                        "next_state": "REAL",
                        "terminate": "bool",
                        "truncated": "bool"} 
        # Grafana table (simplified)
        grafana_keys = {"timestamp": "INTEGER",
                        "ueImsiComplete": "INTEGER",
                        "time_grafana": "INTEGER",
                        "step": "INTEGER",
                        "total_throughput": "REAL",
                        "reward": "REAL"} 
        
        self.datalake._create_table("buffer_table",buffer_keys)
        self.datalake._create_table("grafana",grafana_keys) 
        return super()._init_datalake_usecase()
    
    @override
    def _fill_datalake_usecase(self):
        """
        Fill reply buffer with RL experience tuples for DQN training.
        This method is called after each step to store (state, action, reward, next_state, done) tuples.
        """
        # For reply buffer, we don't read from files - we store RL experiences
        # The actual buffer insertion happens in the step() method via insert_experience()
        pass
    
    def insert_experience(self, state, action, reward, next_state, terminated, truncated):
        """
        Insert a single experience tuple into the reply buffer for DQN training.
        
        Args:
            state: Current state observation
            action: Action taken by the agent
            reward: Reward received
            next_state: Next state observation
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
        """
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(state, np.ndarray):
            state = state.tolist()
        if isinstance(next_state, np.ndarray):
            next_state = next_state.tolist()
        if isinstance(action, np.ndarray):
            action = action.tolist()
        
        # Prepare buffer row
        buffer_row = {
            'timestamp': self.last_timestamp,
            'step': self.num_steps,
            'ueImsiComplete': None,  # System-wide experience
            'state': str(state),  # Convert to string for storage
            'action': str(action),
            'reward': float(reward),
            'next_state': str(next_state),
            'terminate': bool(terminated),
            'truncated': bool(truncated)
        }
        
        # Insert into buffer table
        self.datalake.insert_data("buffer_table", buffer_row)
        
        if self.verbose:
            logging.debug(f"Inserted experience: step={self.num_steps}, reward={reward:.4f}")
    
    def sample_buffer(self, batch_size=32):
        """
        Sample a batch of experiences from the reply buffer for DQN training.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of experience tuples [(state, action, reward, next_state, done), ...]
        """
        # Get all experiences from buffer table
        experiences = self.datalake.read_table("buffer_table")
        
        if len(experiences) < batch_size:
            # Not enough experiences yet
            return []
        
        # Sample random batch
        import random
        sampled_experiences = random.sample(experiences, batch_size)
        
        # Convert back to proper format
        batch = []
        for exp in sampled_experiences:
            # exp format: (timestamp, step, ueImsiComplete, state, action, reward, next_state, terminate, truncated)
            state_str = exp[3]  # state as string
            action_str = exp[4]  # action as string
            reward = exp[5]  # reward as float
            next_state_str = exp[6]  # next_state as string
            terminated = exp[7]  # terminate as bool
            truncated = exp[8]  # truncated as bool
            
            # Parse string representations back to arrays
            import ast
            try:
                state = ast.literal_eval(state_str)
                action = ast.literal_eval(action_str)
                next_state = ast.literal_eval(next_state_str)
                
                # Convert to numpy arrays
                state = np.array(state)
                action = np.array(action)
                next_state = np.array(next_state)
                
                # Determine if episode is done
                done = terminated or truncated
                
                batch.append((state, action, reward, next_state, done))
                
            except (ValueError, SyntaxError) as e:
                if self.verbose:
                    logging.warning(f"Error parsing experience: {e}")
                continue
        
        return batch
    
    def buffer_size(self):
        """
        Get the current size of the reply buffer.
        
        Returns:
            Number of experiences in the buffer
        """
        experiences = self.datalake.read_table("buffer_table")
        return len(experiences)