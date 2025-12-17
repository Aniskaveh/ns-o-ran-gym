import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import argparse
import json
from agents import DQNAgent, ReplayBuffer
from environments.ho_env import HandoverEnv

if __name__ == '__main__':
    #######################
    # Parse arguments #
    #######################
    parser = argparse.ArgumentParser(description="Run the handover environment with a placeholder DQN agent")
    parser.add_argument("--config", type=str, default="src/environments/scenario_configurations/ts_use_case.json",
                        help="Path to the configuration file")
    parser.add_argument("--output_folder", type=str, default="output",
                        help="Path to the output folder")
    parser.add_argument("--ns3_path", type=str, default="/home/ubadmin/ns3-mmwave-oran",
                        help="Path to the ns-3 mmWave O-RAN environment")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of steps to run in the environment")
    parser.add_argument("--optimized", action="store_true",
                        help="Enable optimization mode")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--buffer_size", type=int, default=20_000,
                        help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for each DQN update")
    parser.add_argument("--train_start", type=int, default=5,
                        help="Number of transitions to collect before training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Hidden layer width for the Q-network")
    parser.add_argument("--target_update_freq", type=int, default=10,
                        help="How many updates between target-network syncs")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Initial epsilon value")
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                        help="Minimum epsilon value")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                        help="Multiplicative epsilon decay applied per step")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run the DQN on (cpu or cuda)")

    args = parser.parse_args()

    configuration_path = args.config
    output_folder = args.output_folder
    ns3_path = args.ns3_path
    num_steps = args.num_steps # maximum number of steps for each environment is calculated using the indication periodicity, we refer here to training steps
    optimized = args.optimized
    verbose = args.verbose

    try:
        with open(configuration_path) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{configuration_path}' file, exiting")
        exit(-1)

    scenario_configuration = json.loads(params)

    print('Creating HO Environment')
    env = HandoverEnv(ns3_path=ns3_path, scenario_configuration=scenario_configuration,
                      output_folder=output_folder, optimized=optimized, verbose=verbose)

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        hidden_sizes=[args.hidden_size, args.hidden_size],
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        device=args.device,
    )
    replay_buffer = ReplayBuffer(capacity=args.buffer_size, batch_size=args.batch_size)

    print('Environment Created!')

    print('Launch reset ', end='', flush=True)
    obs, info = env.reset()
    print('done')

    print(f'First set of observations {obs}')
    print(f'Info {info}')

    for step in range(1, num_steps + 1):
        action = agent.select_action(obs)
        print(f'Step {step} ', end='', flush=True)
        print(f'Action {action}')
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.store(obs, action, reward, next_obs, done)
        print(f'Replay buffer length {len(replay_buffer)}')
        loss = None
        if len(replay_buffer) >= args.train_start:
            loss = agent.update(replay_buffer)
            print(f'Loss {loss}')
        print('done', flush=True)

        print(f'Status t = {step}')
        print(f'Actions {env._compute_action(action)}')  # used here only for visualization purposes
        print(f'Observations {next_obs}')
        print(f'Reward {reward}')
        print(f'Terminated {terminated}')
        print(f'Truncated {truncated}')
        print(f'Info {info}')
        print(f'Epsilon {agent.epsilon:.4f}')
        if loss is not None:
            env.log_training_metrics(step=step, loss=loss, epsilon=agent.epsilon)
            print(f'Last training loss {loss:.6f}')

        if done:
            print('Episode finished, resetting environment.')
            obs, info = env.reset()
        else:
            obs = next_obs

    # check_env(env)
