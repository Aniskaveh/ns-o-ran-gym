from typing_extensions import override
from typing import Dict, Tuple
from nsoran.ns_env import NsOranEnv
from gymnasium import spaces
import logging
import os
import re

import numpy as np


class HandoverEnv(NsOranEnv):
    """Scenario-ten environment with reward/training telemetry hooks."""

    REWARD_METRICS_TABLE = "ho_reward_metrics"
    TRAINING_METRICS_TABLE = "ho_training_metrics"

    def __init__(self,
        ns3_path: str,
        scenario_configuration: dict,
        output_folder: str,
        optimized: bool,
        verbose: bool = False,
        time_factor: float = 0.001,
        Cf: float = 1.0,
        lambdaf: float = 0.1,
    ):
        """Environment mirroring TS behaviour but reserved for DQN-based HO."""

        super().__init__(
            ns3_path=ns3_path,
            scenario="scenario-ten",
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized,
            control_header=["timestamp", "ueId", "nrCellId"],
            log_file="HoActions.txt",
            control_file="ho_actions_for_ns3.csv",
        )
        # These features can be hardcoded since they are specific for the use case
        self.columns_state = [
            "RRU.PrbUsedDl",
            "L3 serving SINR",
            "DRB.MeanActiveUeDl",
            "TB.TotNbrDlInitial.Qpsk",
            "TB.TotNbrDlInitial.16Qam",
            "TB.TotNbrDlInitial.64Qam",
            "TB.TotNbrDlInitial",
        ]

        # We need the throughput as well as the cell id to determine whether a handover occurred
        self.columns_reward = ["DRB.UEThpDl.UEID", "nrCellId"]

        # Scenario-one has seven gNBs. Each UE can select one gNB (offset of two) or stay put
        n_gnbs = 7
        n_actions_ue = 7

        # obs_space size: (# ues_per_gnb * # n_gnbs, # observation_columns + timestamp = 1)
        self.observation_space = spaces.Box(
            shape=(self.scenario_configuration["ues"] * n_gnbs, len(self.columns_state) + 1),
            low=-np.inf,
            high=np.inf,
            dtype=np.float64,
        )
        self.action_space = spaces.MultiDiscrete(
            [n_actions_ue] * self.scenario_configuration["ues"] * n_gnbs
        )

        self.previous_df = None
        self.previous_kpms = None
        self.handovers_dict = dict()
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(
                filename="./reward_ho.log",
                level=logging.DEBUG,
                format="%(asctime)s - %(message)s",
            )
        self.time_factor = time_factor
        self.Cf = Cf
        self.lambdaf = lambdaf
        self.reward_step = 0

    def _compute_action(self, action) -> list[tuple]:
        # action from multidiscrete shall become a list of ueId, targetCell.
        # If a targetCell is 0, it means No Handover, thus we don't send it
        action_list = []
        for ueId, targetCellId in enumerate(action):
            if targetCellId != 0: # and 
                # Once we are in this condition, we need to transform the action from the one of gym to the one of ns-O-RAN
                action_list.append((ueId + 1, targetCellId + 2))
        if self.verbose:
            logging.debug(f'Action list {action_list}')
        return action_list

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

    def _get_obs(self) -> list:
        ue_kpms = self.datalake.read_kpms(self.last_timestamp, self.columns_state)                          
        # 'TB.TOTNBRDLINITIAL.QPSK_RATIO', 'TB.TOTNBRDLINITIAL.16QAM_RATIO', 'TB.TOTNBRDLINITIAL.64QAM_RATIO'
        # From per-UE values we need to extract per-Cell Values
        # obs_kpms = []
        # for ue_kpm in ue_kpms:
        #     imsi, kpms = ue_kpm
        #     obs_kpms.append(kpms)

        # _RATIO values are the per Cell value / Tot nbr dl initial

        self.observations = np.array(ue_kpms)
        return self.observations

    def _log_reward_metrics(self, reward_value: float) -> None:
        """Persist per-step reward so Grafana can read it from the datalake."""
        if not hasattr(self, "datalake") or self.REWARD_METRICS_TABLE not in self.datalake.tables:
            return

        timestamp = int(getattr(self, "last_timestamp", 0) or 0)
        self.reward_step += 1
        db_row = {
            "timestamp": timestamp,
            "ueImsiComplete": self.reward_step,
            "step": self.reward_step,
            "reward": float(reward_value),
        }
        self.datalake.insert_data(self.REWARD_METRICS_TABLE, db_row)

    def log_training_metrics(self, step: int, loss: float, epsilon: float) -> None:
        """Record DQN training telemetry (loss + epsilon) for Grafana dashboards."""
        if not hasattr(self, "datalake") or self.TRAINING_METRICS_TABLE not in self.datalake.tables:
            return

        timestamp = int(getattr(self, "last_timestamp", 0) or 0)
        db_row = {
            "timestamp": timestamp,
            "ueImsiComplete": int(step),
            "step": int(step),
            "loss": float(loss),
            "epsilon": float(epsilon),
        }
        self.datalake.insert_data(self.TRAINING_METRICS_TABLE, db_row)

    @override
    def _init_datalake_usecase(self):
        reward_metrics_schema = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "step": "INTEGER",
            "reward": "REAL",
        }
        training_metrics_schema = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "step": "INTEGER",
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
        self.datalake._create_table(self.REWARD_METRICS_TABLE, reward_metrics_schema)
        self.datalake._create_table(self.TRAINING_METRICS_TABLE, training_metrics_schema)
        self.datalake._create_table("ue_positions", ue_pos_schema)
        self.datalake._create_table("gnb_positions", gnb_pos_schema)
        return super()._init_datalake_usecase()
    
    def _compute_reward(self) -> float:
        # Computes the reward for the traffic steering environment. Based off journal on TS
        # The total reward is the sum of per ue rewards, calculated as the difference in the
        # logarithmic throughput between indication periodicities. If an UE experienced an HO,
        # its reward takes into account a cost function related to said handover. The cost
        # function punishes frequent handovers.
        # See the docs for more info.

        total_reward = 0.0
        current_kpms = self.datalake.read_kpms(self.last_timestamp, self.columns_reward)

        # If this is the first iteration we do not have the previous kpms
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
            self._log_reward_metrics(self.reward)
            return self.reward

        # Safety check: if either list is empty, return zero reward
        if len(self.previous_kpms) == 0 or len(current_kpms) == 0:
            if self.verbose:
                logging.warning(f'Empty KPM data at timestamp {self.last_timestamp}, returning zero reward')
            self.previous_kpms = current_kpms if len(current_kpms) > 0 else []
            self.previous_timestamp = self.last_timestamp
            self.reward = 0.0
            self._log_reward_metrics(self.reward)
            return self.reward

        #Assuming they are of the same lenght
        for t_o, t_n in zip(self.previous_kpms, current_kpms):
            ueImsi_o, ueThpDl_o, sourceCell = t_o
            ueImsi_n, ueThpDl_n, currentCell = t_n
            if ueImsi_n == ueImsi_o:
                HoCost = 0
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
        self._log_reward_metrics(self.reward)
        return self.reward
