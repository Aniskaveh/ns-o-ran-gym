
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import argparse
import json
from environments.ho_env import HandoverEnv
from gymnasium.utils.env_checker import check_env
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

################################################################  Main body ################################################################

if __name__ == '__main__':

    #################### Parse arguments ####################
    # This allows us to get the arguments from cmd o.w. it considers default values
    parser = argparse.ArgumentParser(description="Run the handover environment")
    parser.add_argument("--config", type = str, default = "src/environments/scenario_configurations/ho_use_case.json", help = "Path to the configuration file")
    parser.add_argument("--output_folder", type = str, default = "output", help = "Path to the output folder")
    parser.add_argument("--ns3_path", type = str, default = "/home/ubadmin/ns3-mmwave-oran", help = "Path to the ns-3 mmWave O-RAN environment")
    parser.add_argument("--num_steps", type = int, default = 1000, help = "Number of steps to run in the environment")
    parser.add_argument("--optimized", action = "store_true", help = "Enable optimization mode")
    parser.add_argument("--verbose", action = "store_true", help = "Enable verbose logging")
    args = parser.parse_args()

    configuration_path = args.config
    output_folder = args.output_folder
    ns3_path = args.ns3_path
    num_steps = args.num_steps # training steps
    optimized = args.optimized
    verbose = args.verbose

    try:
        with open(configuration_path) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{configuration_path}' file, exiting")
        exit(-1)

    scenario_configuration = json.loads(params)

    ################## Define DRL Agent ################

    class DQNAgent:
        def __init__(self, state_size, action_size, lr= 0.001, gamma=0.95, epsilon=0.2, batch_size=100):
            self.state_size = state_size
            self.action_size = action_size
            self.lr = lr
            self.gamma = gamma
            self.epsilon = epsilon
            self.batch_size = batch_size

            self.model = self._build_model()

        def _build_model(self):
            model = Sequential()
            model.add(Dense(64, input_dim = self.state_size, activation = "relu"))
            model.add(Dense(64, activation = 'relu'))
            model.add(Dense(self.action_size, activation = "linear"))
            model.compile(loss = 'mse', optimizer = Adam(learning_rate = self.lr))
            model.summary()
            return model
        
        def choose_action(self, state):
            if np.random.rand() < self.epsilon:
                action = random.randrange(self.action_size) #explore
                return action 
            else: 
                q_values = self.model.predict(np.array([state]), verbose = 0) #exploit
                action = np.argmax(q_values[0])
            return action 
        
        def train(self, batch):
            """
            Train DQN on a batch of experiences from the reply buffer.
            
            Args:
                batch: List of (state, action, reward, next_state, done) tuples
            """
            if len(batch) == 0:
                return
            
            # Extract components from batch
            states = np.array([exp[0] for exp in batch])
            actions = np.array([exp[1] for exp in batch])
            rewards = np.array([exp[2] for exp in batch])
            next_states = np.array([exp[3] for exp in batch])
            dones = np.array([exp[4] for exp in batch])

            # Get current Q-values
            q_values = self.model.predict(states, verbose=0)
            q_next = self.model.predict(next_states, verbose=0)

            # Update Q-values with Bellman equation
            for i in range(len(batch)):
                target = rewards[i]
                if not dones[i]:
                    target += self.gamma * np.max(q_next[i])
                q_values[i][actions[i]] = target

            # Train model on the batch
            self.model.fit(states, q_values, epochs=1, verbose=0)


    #################### Create ENV ####################

    print('Creating HO Environment')
    env = HandoverEnv(ns3_path = ns3_path, scenario_configuration = scenario_configuration,
                      output_folder = output_folder, optimized = optimized, verbose = verbose)
    print('Environment Created!')

    #################### Reset ENV ####################
    print('Creating Agent')
    agent = DQNAgent(state_size= None, action_size= None)
    print('Agent Created!')

    #################### Train ENV ####################
    min_experience = 100

    for step in range(num_steps):

        print(f'Step t = {step} ', end='', flush=True)
        obs, info = env.reset()

        total_reward = 0

        while not terminated and truncated:

            action = agent.choose_action([obs])
            obs_next, reward, terminated, truncated, info = env.step(action)
            
            # Store experience in reply buffer for DQN training
            env.insert_experience(obs, action, reward, obs_next, terminated, truncated)
            
            # Train DQN when buffer has enough experiences
            if env.buffer_size() >= min_experience:
                batch = env.sample_buffer(batch_size=32)
                if len(batch) > 0:
                    agent.train(batch)
            
            obs = obs_next
            total_reward += reward

        ########### Visualization ###########

        print(f'Total Reward = {total_reward}')
        print(f'Info {info}')