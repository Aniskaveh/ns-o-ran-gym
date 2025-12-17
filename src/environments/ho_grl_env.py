"""Gym-compatible environment wrapping ns-O-RAN for Graph-based RL (GAT encoder + DQN head)."""

from __future__ import annotations
from typing_extensions import override
from typing import Dict, Tuple, List, Any
from nsoran.ns_env import NsOranEnv
from gymnasium import spaces
import logging
import os
import re
import numpy as np
import math


class HandoverGRLEnv(NsOranEnv):
    """Scenario-ten environment with reward/training telemetry hooks."""

    def __init__(self,ns3_path: str,scenario_configuration: dict,output_folder: str,
                optimized: bool,verbose: bool = False,time_factor: float = 0.001,
                Cf: float = 1.0,lambdaf: float = 0.1,):
        # Cf: Cost factor for handovers.
        # lambdaf: Decay factor for handover cost.
        # time_factor: Time factor for the environment.
        # Calls base class initializer and sets up the environment.
        super().__init__(ns3_path=ns3_path, scenario="scenario-ten", 
                        scenario_configuration=scenario_configuration,
                        output_folder=output_folder,optimized=optimized, 
                        control_header=["timestamp", "ueId", "nrCellId"],
                        log_file="HoActions.txt",control_file="ho_actions_for_ns3.csv",)
        # These features can be hardcoded since they are specific for the use case
        # UE features: position (from ue_positions), serving SINR, throughput, buffer size, top 3 neighbor SINRs (from gnb_cu_cp and du)
        # We also need neighbor IDs to map SINRs to specific gNBs for edge attributes
        self.columns_state_ue = ["ue_x", "ue_y", 
                                "L3 serving SINR", "L3 serving Id(m_cellId)",
                                "DRB.UEThpDl.UEID", "DRB.BufferSize.Qos.UEID",
                                "L3 neigh SINR 1", "L3 neigh Id 1 (cellId)",
                                "L3 neigh SINR 2", "L3 neigh Id 2 (cellId)",
                                "L3 neigh SINR 3", "L3 neigh Id 3 (cellId)"]
        # gNB features: position (from gnb_positions), numActiveUes (from gnb_cu_cp, aggregated), PRB metrics (from du, aggregated)
        self.columns_state_gnb = ["gnb_x", "gnb_y",
                                  "numActiveUes", "RRU.PrbUsedDl", "dlAvailablePrbs"]
        # We need the throughput as well as the cell id to determine whether a handover occurred
        self.columns_reward = ["DRB.UEThpDl.UEID", "nrCellId"]

        n_ue = self.scenario_configuration.get("ues", 14)
        n_gnbs = 7
        self.observation_space = spaces.Dict(
            {
                "ue_features": spaces.Box(low=-np.inf, high=np.inf, shape=(n_ue, len(self.columns_state_ue))),
                "gnb_features": spaces.Box(low=-np.inf, high=np.inf, shape=(n_gnbs, len(self.columns_state_gnb))),
                "edge_index": spaces.Box(low=0, high=np.inf, shape=(2, n_ue * n_gnbs)),
                "edge_attr": spaces.Box(low=-np.inf, high=np.inf, shape=(n_ue * n_gnbs, 3)),  # [distance, is_serving_cell, sinr_to_gNB]
                "action_mask": spaces.Box(low=-1, high=n_gnbs, shape=(n_ue,), dtype=np.int32),  # Action indices to mask per UE (-1 = no mask)
                "gnb_ids": spaces.Box(low=2, high=8, shape=(n_gnbs,), dtype=np.int32),  # CellIds corresponding to gNB indices
            })

        # Scenario-ten has seven gNBs. Each UE can select one gNB (offset of two) or stay put (action 0)
        n_gnbs = 7
        n_actions_ue = n_gnbs+1

        # obs_space size: (# ues_per_gnb * # n_gnbs, # observation_columns + timestamp = 1)
        self.action_space = spaces.MultiDiscrete([n_actions_ue] * self.scenario_configuration["ues"] * n_gnbs)
        # Used for tracking previous KPI snapshots for reward calculation.
        self.previous_df = None
        self.previous_kpms = None
        self.handovers_dict = dict() #stores last HO timestamp per UE for cost computation.
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(filename="./reward_ho.log",level=logging.DEBUG,
                                format="%(asctime)s - %(message)s")
        self.time_factor = time_factor
        self.Cf = Cf
        self.lambdaf = lambdaf
        self.reward_step = 0

    def _compute_action(self, action) -> list[tuple]:
        # action from multidiscrete shall become a list of ueId, targetCell.
        # If a targetCell is 0, it means No Handover, thus we don't send it
        # Converts Gym action vector → ns-O-RAN HO commands.
        action_list = []
        for ueId, targetCellId in enumerate(action):
            if targetCellId != 0: # and 
                # Once we are in this condition, we need to transform the action from the one of gym to the one of ns-O-RAN
                action_list.append((ueId + 1, targetCellId + 1))
        if self.verbose:
            logging.debug(f'Action list {action_list}')
        return action_list

    # Parses positions from ns-3 gnuplot files. Returns {id: (x, y)}.
    @staticmethod
    def _parse_positions(file_path: str) -> Dict[int, Tuple[float, float]]:
        """Parse positions from ns-3 gnuplot files."""
        positions: Dict[int, Tuple[float, float]] = {}
        if not os.path.exists(file_path):
            return positions
        pattern = re.compile(r'set label\s+"(?P<id>\d+)"\s+at\s+(?P<x>\d+(?:\.\d+)?),(?P<y>\d+(?:\.\d+)?)')
        with open(file_path, "r") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    idx = int(m.group("id"))
                    positions[idx] = (float(m.group("x")), float(m.group("y")))
        return positions

    def _fill_datalake_usecase(self):
        """Populate position tables (ue_positions, gnb_positions) if files exist."""
        # If base env did not create the tables, just skip
        if not hasattr(self, "datalake") or "ue_positions" not in self.datalake.tables or "gnb_positions" not in self.datalake.tables:
            return super()._fill_datalake_usecase()

        sim_path = getattr(self, "sim_path", None)
        ts = int(getattr(self, "last_timestamp", 0) or 0)
        if sim_path:
            ue_pos = self._parse_positions(os.path.join(sim_path, "ues.txt"))
            gnb_pos = self._parse_positions(os.path.join(sim_path, "enbs.txt"))

            if ue_pos:
                for ue_id, (x, y) in ue_pos.items():
                    row = {
                        "timestamp": ts,
                        "ueImsiComplete": int(ue_id),
                        "ue_x": float(x),
                        "ue_y": float(y),
                    }
                    self.datalake.insert_data("ue_positions", row)

            if gnb_pos:
                for cell_id, (x, y) in gnb_pos.items():
                    # store cell_id also in ueImsiComplete column to satisfy UNIQUE(timestamp, ueImsiComplete)
                    row = {
                        "timestamp": ts,
                        "ueImsiComplete": int(cell_id),
                        "cellId": int(cell_id),
                        "gnb_x": float(x),
                        "gnb_y": float(y),
                    }
                    self.datalake.insert_data("gnb_positions", row)

        return super()._fill_datalake_usecase()

    def log_training_metrics(self, step: int, reward_value: float,loss: float, epsilon: float) -> None:
        if not hasattr(self, "datalake") or "ho_reward_metrics" not in self.datalake.tables:
            return

        timestamp = int(getattr(self, "last_timestamp", 0) or 0)
        db_row = {
            "timestamp": timestamp,
            "ueImsiComplete": int(step),
            "step": int(step),
            "reward": float(reward_value),
            "loss": float(loss),
            "epsilon": float(epsilon),
        }
        self.datalake.insert_data("ho_training_metrics", db_row)

    @override
    def _init_datalake_usecase(self):
        training_metrics_schema = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "step": "INTEGER",
            "reward": "REAL",
            "loss": "REAL",
            "epsilon": "REAL",
        }
        # Position tables for GRL/telemetry
        ue_pos_schema = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "ue_x": "REAL",
            "ue_y": "REAL",
        }
        gnb_pos_schema = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",  # kept for UNIQUE constraint
            "cellId": "INTEGER",
            "gnb_x": "REAL",
            "gnb_y": "REAL",
        }
        self.datalake._create_table("ho_training_metrics", training_metrics_schema)
        self.datalake._create_table("ue_positions", ue_pos_schema)
        self.datalake._create_table("gnb_positions", gnb_pos_schema)
        return super()._init_datalake_usecase()

    def _get_obs(self) -> list:
        n_gnbs = 7
        n_ue = self.scenario_configuration["ues"] * n_gnbs
        ue_rows = self.datalake.read_kpms(self.last_timestamp,self.columns_state_ue) or []
        gnb_rows = self.datalake.read_kpms(self.last_timestamp,self.columns_state_gnb) or []

        # What are ue_dict and ue_feats? 
        # ue_dict is a dictionary that holds all UE raw data, keyed by UE ID like: 
        # {ue_id: {x: float, y: float, ...}} while ue_feats is a list of lists that 
        # holds the UE features for each UE like: [[x, y, sinr, ...]],ready for ML input.
        # Initializes a dict to hold all UE data, keyed by UE ID
        ue_dict: Dict[int, Dict[str, float]] = {}
        for r in ue_rows:
            ue_id = int(r[0])
            # According to the columns_state_ue, Columns [7, 8], [9, 10], [11, 12] correspond to up to 3 
            # neighboring cells and their SINRs. like: [SINR, cellId]
            neigh_sinr_map: Dict[int, float] = {} #Creates a dictionary: {neighbor_cell_id: SINR} mapping of neighbor_cellId -> SINR for this UE's neighbors
            if len(r) > 7 and r[7] is not None and r[8] is not None:
                cell_id = int(r[8])
                sinr_val = float(r[7])
                if cell_id > 0:  # Valid cell ID
                    neigh_sinr_map[cell_id] = sinr_val
            if len(r) > 9 and r[9] is not None and r[10] is not None:
                cell_id = int(r[10])
                sinr_val = float(r[9])
                if cell_id > 0:
                    neigh_sinr_map[cell_id] = sinr_val
            if len(r) > 11 and r[11] is not None and r[12] is not None:
                cell_id = int(r[12])
                sinr_val = float(r[11])
                if cell_id > 0:
                    neigh_sinr_map[cell_id] = sinr_val
            # So neigh_sinr_map can be like: {2: 15.0, 3: 10.0, 4: 8.0}
            
            ue_dict[ue_id] = {
                "x": float(r[1]) if r[1] is not None else 0.0,
                "y": float(r[2]) if r[2] is not None else 0.0,
                "sinr": float(r[3]) if r[3] is not None else 0.0,
                "serv_cell": int(r[4]) if r[4] is not None else -1,
                "throughput": float(r[5]) if r[5] is not None else 0.0,
                "buffer": float(r[6]) if r[6] is not None else 0.0,
                "neigh_sinr1": float(r[7]) if len(r) > 7 and r[7] is not None else 0.0,
                "neigh_sinr2": float(r[9]) if len(r) > 9 and r[9] is not None else 0.0,
                "neigh_sinr3": float(r[11]) if len(r) > 11 and r[11] is not None else 0.0,
                "neigh_sinr_map": neigh_sinr_map,  # Map cellId -> SINR for edge attributes
            }
        # Initializes a dict to hold all gnb data, keyed by gnb ID
        gnb_dict: Dict[int, Dict[str, float]] = {} #Creates a dictionary: {gnb_id: {x: float, y: float, numActiveUes: float, prb_usage: float, avail_prbs: float}}
        for r in gnb_rows:
            cid = int(r[0])
            gnb_dict[cid] = {
                "x": float(r[1]) if r[1] is not None else 0.0,
                "y": float(r[2]) if r[2] is not None else 0.0,
                "numActiveUes": float(r[3]) if r[3] is not None else 0.0,  # Will be overwritten below
                "prb_usage": float(r[4]) if r[4] is not None else 0.0,
                "avail_prbs": float(r[5]) if r[5] is not None else 0.0,
            }

        # Count actual number of UEs connected to each cellId
        # by counting how many UEs have each cellId as their serving cell
        cell_ue_count: Dict[int, int] = {}
        for ue_id, ue_data in ue_dict.items():
            serv_cell = ue_data.get("serv_cell", -1)
            if serv_cell > 0:  # Valid cell ID (exclude -1)
                cell_ue_count[serv_cell] = cell_ue_count.get(serv_cell, 0) + 1
        
        # Update numActiveUes in gnb_dict with actual counts
        for cid in gnb_dict.keys():
            gnb_dict[cid]["numActiveUes"] = float(cell_ue_count.get(cid, 0))

        ue_ids = sorted(ue_dict.keys())[:n_ue] or list(range(n_ue)) #ensures only first n_ue UEs are used
        # Keep only NR gNBs (labels 2–8 in enbs.txt), drop LTE anchor (1). #Only considers gNBs 2-8 for handover decisions
        gnb_ids = [cid for cid in sorted(gnb_dict.keys()) if cid >= 2][:n_gnbs]
        if not gnb_ids:
            gnb_ids = list(range(2, 2 + n_gnbs))  # fallback to 2..8

        # Build UE feature vectors: [x, y, serving_sinr, throughput, buffer, neigh_sinr1, neigh_sinr2, neigh_sinr3]
        ue_feats: List[List[float]] = []
        serving_cells: List[int] = [] #keeps track of the gNB each UE is connected to
        for uid in ue_ids:
            u = ue_dict.get(uid, {})
            ue_feats.append([
                u.get("x", 0.0),
                u.get("y", 0.0),
                u.get("sinr", 0.0),
                u.get("throughput", 0.0),
                u.get("buffer", 0.0),
                u.get("neigh_sinr1", 0.0),
                u.get("neigh_sinr2", 0.0),
                u.get("neigh_sinr3", 0.0),
            ])
            serving_cells.append(int(u.get("serv_cell", -1)))
        # pad/truncate UE features to fixed n_ue. Ensures fixed matrix shape, necessary for ML input.
        while len(ue_feats) < n_ue:
            ue_feats.append([0.0] * len(self.columns_state_ue))  # 8 features per UE
            serving_cells.append(-1)
        ue_feats = ue_feats[:n_ue]
        serving_cells = serving_cells[:n_ue]

        # Build gNB feature vectors: [x, y, numActiveUes, prb_usage, avail_prbs]
        gnb_feats: List[List[float]] = []
        for gid in gnb_ids:
            g = gnb_dict.get(gid, {})
            gnb_feats.append([
                g.get("x", 0.0),
                g.get("y", 0.0),
                g.get("numActiveUes", 0.0),
                g.get("prb_usage", 0.0),
                g.get("avail_prbs", 0.0),
            ])
        # pad/truncate gNB features to fixed n_gnb, Ensures fixed matrix shape, necessary for ML input.
        while len(gnb_feats) < n_gnbs:
            gnb_feats.append([0.0] * len(self.columns_state_gnb))  # 5 features per gNB
        gnb_feats = gnb_feats[:n_gnbs]

        # Goal is to create a graph with UEs as nodes and gNBs as edges.
        # Nodes: UEs (indices 0…n_ue-1), gNBs (indices n_ue…n_ue+n_gnbs-1)
        # Edges: UEs → gNBs (connect each UE to every gNB)
        edges_src: List[int] = []
        edges_dst: List[int] = []
        edge_attr: List[List[float]] = []
        for u_idx, u_feat in enumerate(ue_feats[:n_ue]):
            ue_id = ue_ids[u_idx] if u_idx < len(ue_ids) else None
            ue_data = ue_dict.get(ue_id, {}) if ue_id is not None else {}
            ux, uy = u_feat[0], u_feat[1]
            serv_cell = serving_cells[u_idx] if u_idx < len(serving_cells) else -1
            # g_offset = index of the gNB in the gNB list (0…n_gnbs-1)
            # n_ue + g_offset → shifts gNB indices after all UE nodes
            for g_offset in range(n_gnbs):
                gid = gnb_ids[g_offset] if g_offset < len(gnb_ids) else (2 + g_offset)
                gx, gy = gnb_feats[g_offset][0], gnb_feats[g_offset][1]
                
                # Compute edge attributes
                distance = math.hypot(ux - gx, uy - gy)
                
                # is_serving_cell: 1.0 if this gNB is the serving cell, 0.0 otherwise
                is_serving = 1.0 if serv_cell == gid else 0.0
                
                # sinr_to_gNB: SINR from UE to this specific gNB
                # Check if this gNB is in the neighbor SINR map
                neigh_sinr_map = ue_data.get("neigh_sinr_map", {})
                # Use -100.0 dB as default to indicate "no measurement" (distinct from poor SINR like -5 dB)
                sinr_to_gNB = neigh_sinr_map.get(gid, -100.0)
                
                # If gNB is serving cell, use serving SINR instead
                if is_serving > 0.5:
                    sinr_to_gNB = ue_data.get("sinr", -100.0)
                
                edges_src.append(u_idx)
                edges_dst.append(n_ue + g_offset)
                edge_attr.append([distance, is_serving, sinr_to_gNB])

        edge_index = np.vstack([edges_src, edges_dst]).astype(np.int64)
        # edge_attr now has 3 dimensions: [distance, is_serving_cell, sinr_to_gNB]
        edge_attr_arr = np.asarray(edge_attr, dtype=np.float32) if edge_attr else np.zeros((0, 3), dtype=np.float32)

        # Compute action mask: for each UE, which action (gNB) corresponds to its serving cell?
        # Action mapping: action 0 = no HO, action 1-7 = HO to gNB at gnb_ids[0-6]
        # If serving cellId is in gnb_ids, mask the corresponding action
        action_mask: List[int] = []  # For each UE, the action index to mask (-1 if no mask needed)
        for u_idx in range(n_ue):
            serv_cell = serving_cells[u_idx] if u_idx < len(serving_cells) else -1
            masked_action = -1  # -1 means no mask needed
            if serv_cell > 0 and serv_cell in gnb_ids:
                # Find which gNB index corresponds to this serving cell
                try:
                    g_offset = gnb_ids.index(serv_cell)
                    # Action = g_offset + 1 (since action 0 is no HO)
                    masked_action = g_offset + 1
                except ValueError:
                    # serv_cell not in gnb_ids (e.g., LTE anchor cellId 1), no mask needed
                    masked_action = -1
            action_mask.append(masked_action)

        self.observations = {
            "ue_features": np.asarray(ue_feats, dtype=np.float32),
            "gnb_features": np.asarray(gnb_feats, dtype=np.float32),
            "edge_index": edge_index,
            "edge_attr": edge_attr_arr,
            "action_mask": np.asarray(action_mask, dtype=np.int32),  # Action indices to mask per UE (-1 = no mask)
            "gnb_ids": np.asarray(gnb_ids, dtype=np.int32),  # For reference: which cellIds correspond to gNB indices
        }

        return self.observations
    
    def _compute_reward(self) -> float:
        # Reward is per-UE, summed to get a total reward
        # Uses logarithmic difference in throughput and handover cost
        # Reward = Log(new_throughput) - Log(old_throughput) - HandoverCost
        total_reward = 0.0
        current_kpms = self.datalake.read_kpms(self.last_timestamp, self.columns_reward)

        # If this is the first iteration we do not have the previous kpms
        # So it tries to fetch KPIs from one indication period ago.
        # This is needed because reward is based on change in throughput over time
        if self.previous_kpms is None:
            if self.verbose:
                logging.debug(f'Starting first reward computation at timestamp {self.last_timestamp}')
            self.previous_timestamp = self.last_timestamp - (self.scenario_configuration['indicationPeriodicity'] * 1000)
            self.previous_kpms = self.datalake.read_kpms(self.previous_timestamp, self.columns_reward)

        # Safety check: if we still don't have previous or current kpms, return zero reward
        if self.previous_kpms is None or current_kpms is None:
            if self.verbose:
                logging.warning(f'Missing KPM data at timestamp {self.last_timestamp}, returning zero reward')
            self.previous_kpms = current_kpms if current_kpms is not None else []
            self.previous_timestamp = self.last_timestamp
            self.reward = 0.0
            return self.reward

        # Safety check: if either list is empty, return zero reward
        if len(self.previous_kpms) == 0 or len(current_kpms) == 0:
            if self.verbose:
                logging.warning(f'Empty KPM data at timestamp {self.last_timestamp}, returning zero reward')
            self.previous_kpms = current_kpms if len(current_kpms) > 0 else []
            self.previous_timestamp = self.last_timestamp
            self.reward = 0.0
            return self.reward

        #Assuming both current and previous lists are of the same lenght and same UE order.
        for t_o, t_n in zip(self.previous_kpms, current_kpms):
            ueImsi_o, ueThpDl_o, sourceCell = t_o
            ueImsi_n, ueThpDl_n, currentCell = t_n
            if ueImsi_n == ueImsi_o: #ensures same UE is being compared
                HoCost = 0 #penalizes frequent handovers, Formula: HoCost = Cf * ((1 - lambdaf) ** timeDiff)
                if currentCell != sourceCell:
                    lastHo = self.handovers_dict.get(ueImsi_n, 0)  # Retrieve last handover time or default to 0
                    if lastHo != 0: # If this is the first HO the cost is 0
                        timeDiff = (self.last_timestamp - lastHo) * self.time_factor
                        HoCost = self.Cf * ((1 - self.lambdaf) ** timeDiff)
                    self.handovers_dict[ueImsi_n] = self.last_timestamp  # Update dictionary
 
                LogOld = 0
                LogNew = 0
                if ueThpDl_o != 0:
                    LogOld = np.log10(ueThpDl_o)
                if ueThpDl_n != 0:
                    LogNew = np.log10(ueThpDl_n)

                LogDiff = LogNew - LogOld
                reward_ue = LogDiff - HoCost
                if self.verbose:
                    logging.debug(f"Reward for UE {ueImsi_n}: {reward_ue} (LogDiff: {LogDiff}, HoCost: {HoCost})")
                total_reward += reward_ue
            else:
                if self.verbose:
                    logging.error(f"Unexpected UeImsi mismatch: {ueImsi_o} != {ueImsi_n} (current ts: {self.last_timestamp})")
        if(self.verbose):
            logging.debug(f"Total reward: {total_reward}")
        self.previous_kpms = current_kpms
        self.previous_timestamp = self.last_timestamp
        self.reward = total_reward
        return self.reward
