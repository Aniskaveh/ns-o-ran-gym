import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import argparse
import json
import torch
import numpy as np
from agents.grl_agent import GRLAgent, GraphReplayBuffer, GraphTransition
from environments.ho_grl_env import HandoverGRLEnv


def print_normalization_comparison(ue_raw, gnb_raw, edge_raw, ue_norm, gnb_norm, edge_norm):
    """Print a formatted comparison table of raw vs normalized features."""
    print("\n  📊 Normalization Statistics:")
    print("  " + "="*70)
    print(f"  {'Feature Type':<15} {'Raw Range':<25} {'Normalized Range':<25}")
    print("  " + "-"*70)
    print(f"  {'UE Features':<15} [{ue_raw.min():>8.2f}, {ue_raw.max():>8.2f}]  [{ue_norm.min():>8.4f}, {ue_norm.max():>8.4f}]")
    print(f"  {'  (mean/std)':<15} {ue_raw.mean():>8.2f} / {ue_raw.std():>8.2f}    {ue_norm.mean():>8.4f} / {ue_norm.std():>8.4f}")
    print(f"  {'gNB Features':<15} [{gnb_raw.min():>8.2f}, {gnb_raw.max():>8.2f}]  [{gnb_norm.min():>8.4f}, {gnb_norm.max():>8.4f}]")
    print(f"  {'  (mean/std)':<15} {gnb_raw.mean():>8.2f} / {gnb_raw.std():>8.2f}    {gnb_norm.mean():>8.4f} / {gnb_norm.std():>8.4f}")
    print(f"  {'Edge Features':<15} [{edge_raw.min():>8.2f}, {edge_raw.max():>8.2f}]  [{edge_norm.min():>8.4f}, {edge_norm.max():>8.4f}]")
    print(f"  {'  (mean/std)':<15} {edge_raw.mean():>8.2f} / {edge_raw.std():>8.2f}    {edge_norm.mean():>8.4f} / {edge_norm.std():>8.4f}")
    print("  " + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Run HO environment with GRL (GAT + DQN) agent")
    parser.add_argument("--config", type=str, default="src/environments/scenario_configurations/ts_use_case.json", help="Path to the configuration file")
    parser.add_argument("--output_folder", type=str, default="output", help="Path to the output folder")
    parser.add_argument("--ns3_path", type=str, default="/home/ubadmin/ns3-mmwave-oran", help="Path to ns-3")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to run")
    parser.add_argument("--optimized", action="store_true", help="Enable ns-3 optimized build")
    # --optimized likely controls: ns-3 build type, Debug vs optimized binaries, Logging verbosity, Runtime performance
    # This is standard practice for simulation-heavy experiments and reproducibility.
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    # GRL hyperparameters
    parser.add_argument("--buffer_size", type=int, default=10_000, help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for updates") #64
    parser.add_argument("--train_start", type=int, default=2, help="Steps before training starts, enforces a pure exploration phase.") #200
    # Why --train_start? The agent does not update weights until: Some interaction history exists, Initial graph states have been observed, 
    # Random actions have populated the buffer, train_start is a stability guard while buffer.can_sample() is a technical guard
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dim for GAT/heads")
    parser.add_argument("--target_update_freq", type=int, default=10, help="Target sync frequency") #200
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Min epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay per step")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    args = parser.parse_args()

    with open(args.config) as f:
        scenario_configuration = json.load(f)

    print("Creating HO GRL Environment")
    env = HandoverGRLEnv(ns3_path=args.ns3_path,scenario_configuration=scenario_configuration,output_folder=args.output_folder,optimized=args.optimized,verbose=args.verbose)

    # First reset to get feature dimensions.
    obs, info = env.reset()
    ue_feat_dim = obs["ue_features"].shape[1]
    gnb_feat_dim = obs["gnb_features"].shape[1]
    n_ue = obs["ue_features"].shape[0]
    n_gnb = obs["gnb_features"].shape[0]

    print(f"\n=== GRL Agent Configuration ===")
    print(f"UE features: {ue_feat_dim}D (shape: {obs['ue_features'].shape})")
    print(f"gNB features: {gnb_feat_dim}D (shape: {obs['gnb_features'].shape})")
    print(f"Number of UEs: {n_ue}, Number of gNBs: {n_gnb}")
    print(f"Edges: {obs['edge_index'].shape[1]}")
    edge_attr_dim = obs["edge_attr"].shape[1] if obs["edge_attr"].size > 0 else 0
    print(f"Edge attributes: {edge_attr_dim}D (shape: {obs['edge_attr'].shape})")
    print(f"Action space: {env.action_space}")
    print(f"Buffer size: {args.buffer_size}, Batch size: {args.batch_size}")
    print(f"Training starts at step: {args.train_start}")
    print(f"Target network update frequency: {args.target_update_freq}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Sample UE features (first UE): {obs['ue_features'][0]}")
    print(f"Sample gNB features (first gNB): {obs['gnb_features'][0]}")
    print("=" * 40 + "\n")

    # Initialize adaptive normalization (not static) with sample statistics for immediate normalization
    # Collect a few samples to estimate mean and std
    # Normalization parameters are: Learnable and Initialized using sample statistics
    # This is more robust than fixed normalization.
    # Why do we collect samples before training? Estimate reasonable initial statistics for normalization to avoid exploding gradients.
    # Without this sampling, normalization layers would start with: mean = 0, std = 1, but your real data might look like: UE RSRP mean ≈ -90,
    #  gNB load mean ≈ 0.6, Edge distance mean ≈ 120. So the first forward pass would produce very large activations, especially inside attention layers.
    sample_obs = [obs]
    for _ in range(min(5, args.train_start)):
        action = env.action_space.sample()
        next_obs, _, _, _, _ = env.step(action)
        sample_obs.append(next_obs)
        obs = next_obs
    
    # Compute statistics from samples
    ue_samples = np.concatenate([o['ue_features'] for o in sample_obs], axis=0)
    gnb_samples = np.concatenate([o['gnb_features'] for o in sample_obs], axis=0)
    edge_samples = np.concatenate([o['edge_attr'] for o in sample_obs], axis=0)
    
    # Initialize with mean and std (standard normalization: (x - mean) / std)
    # This initializes learnable normalization layers inside the network. Think of it as: Start normalization 
    # close to reality instead of guessing.
    ue_mean = torch.tensor(ue_samples.mean(axis=0), dtype=torch.float32)
    ue_std = torch.tensor(ue_samples.std(axis=0), dtype=torch.float32) + 1e-8  # Add small epsilon
    gnb_mean = torch.tensor(gnb_samples.mean(axis=0), dtype=torch.float32)
    gnb_std = torch.tensor(gnb_samples.std(axis=0), dtype=torch.float32) + 1e-8
    edge_mean = torch.tensor(edge_samples.mean(axis=0), dtype=torch.float32)
    edge_std = torch.tensor(edge_samples.std(axis=0), dtype=torch.float32) + 1e-8
    
    # Reset environment after sampling
    obs, info = env.reset()
    
    agent = GRLAgent(action_space=env.action_space,ue_feat_dim=ue_feat_dim,
                    gnb_feat_dim=gnb_feat_dim,device=args.device,
                    gamma=args.gamma,learning_rate=args.learning_rate,
                    hidden_dim=args.hidden_dim,target_update_freq=args.target_update_freq,
                    epsilon_start=args.epsilon_start,epsilon_end=args.epsilon_end,
                    epsilon_decay=args.epsilon_decay,edge_attr_dim=edge_attr_dim,
                    ue_init_scale=ue_std, ue_init_shift=ue_mean,
                    gnb_init_scale=gnb_std, gnb_init_shift=gnb_mean,
                    edge_init_scale=edge_std, edge_init_shift=edge_mean,
                )
    buffer = GraphReplayBuffer(capacity=args.buffer_size, batch_size=args.batch_size)

    # Initial normalization check
    print("\n=== Initial Normalization Check ===")
    ue_tensor = torch.as_tensor(obs['ue_features'], dtype=torch.float32)
    gnb_tensor = torch.as_tensor(obs['gnb_features'], dtype=torch.float32)
    edge_tensor = torch.as_tensor(obs['edge_attr'], dtype=torch.float32)
    # Applies normalization layers inside the network.
    ue_norm, gnb_norm, edge_norm = agent.online_net.normalize_features(ue_tensor, gnb_tensor, edge_tensor)
    print_normalization_comparison(obs['ue_features'], obs['gnb_features'], obs['edge_attr'],
                                  ue_norm.detach().cpu().numpy(),
                                  gnb_norm.detach().cpu().numpy(),
                                  edge_norm.detach().cpu().numpy())
    print("\n✅ Normalization layers initialized (parameters will adapt during training)")
    print("=" * 40 + "\n")

    print("Environment ready, starting training loop\n")

    for step in range(1, args.num_steps + 1):
        #Mask indicates valid gNBs per UE
        action, action_mask = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.store(GraphTransition(obs, action, reward, next_obs, done))

        loss = None
        if step >= args.train_start and buffer.can_sample():
            loss = agent.update(buffer)
            if loss is not None:
                env.log_training_metrics(step=step, reward_value=reward, loss=loss, epsilon=agent.epsilon)

        # Logging for visibility
        buffer_size = len(buffer)
        loss_str = f"{loss:.6f}" if loss is not None else "N/A"
        print(f"Step {step:4d} | action mask: {action_mask} | action: {action} | Reward: {reward:8.4f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Loss: {loss_str:>10} | "
              f"Buffer: {buffer_size:4d}/{args.buffer_size}")
        
        # Print detailed info every 10 steps or when training starts
        if step % 10 == 0 or (loss is not None and step == args.train_start):
            print(f"  └─ Action sample (first 5 UEs): {action[:5]}")
            
            # Raw feature statistics
            ue_raw = obs['ue_features']
            gnb_raw = obs['gnb_features']
            edge_raw = obs['edge_attr']
            
            print(f"  └─ Raw UE feat range: [{ue_raw.min():.2f}, {ue_raw.max():.2f}], mean: {ue_raw.mean():.2f}, std: {ue_raw.std():.2f}")
            print(f"  └─ Raw gNB feat range: [{gnb_raw.min():.2f}, {gnb_raw.max():.2f}], mean: {gnb_raw.mean():.2f}, std: {gnb_raw.std():.2f}")
            print(f"  └─ Raw edge feat range: [{edge_raw.min():.2f}, {edge_raw.max():.2f}], mean: {edge_raw.mean():.2f}, std: {edge_raw.std():.2f}")
            
            # Normalized feature statistics
            ue_tensor = torch.as_tensor(ue_raw, dtype=torch.float32)
            gnb_tensor = torch.as_tensor(gnb_raw, dtype=torch.float32)
            edge_tensor = torch.as_tensor(edge_raw, dtype=torch.float32)
            
            ue_norm, gnb_norm, edge_norm = agent.online_net.normalize_features(ue_tensor, gnb_tensor, edge_tensor)
            
            # Print formatted comparison table
            print_normalization_comparison(ue_raw, gnb_raw, edge_raw, 
                                          ue_norm.detach().cpu().numpy(), 
                                          gnb_norm.detach().cpu().numpy(), 
                                          edge_norm.detach().cpu().numpy())
            
            # Show normalization parameters (first few dimensions)
            if step % 50 == 0:  # Less frequent to avoid clutter
                norm_stats = agent.online_net.get_normalization_stats()
                print("\n  🔧 Normalization Parameters (learnable):")
                print(f"  └─ UE norm params (first 3 dims):")
                print(f"      scale: {norm_stats['ue']['scale'][:3].tolist()}")
                print(f"      shift: {norm_stats['ue']['shift'][:3].tolist()}")
                print(f"  └─ gNB norm params (first 3 dims):")
                print(f"      scale: {norm_stats['gnb']['scale'][:3].tolist()}")
                print(f"      shift: {norm_stats['gnb']['shift'][:3].tolist()}")
                print(f"  └─ Edge norm params:")
                print(f"      scale: {norm_stats['edge']['scale'].tolist()}")
                print(f"      shift: {norm_stats['edge']['shift'].tolist()}")
            
            if loss is not None:
                steps_until_target_update = args.target_update_freq - (agent.train_step % args.target_update_freq)
                print(f"  └─ Training: ✓ (step {agent.train_step}, target update in {steps_until_target_update} steps)")
            else:
                needed = max(0, args.batch_size - buffer_size)
                print(f"  └─ Training: ✗ (need {needed} more samples)")

        if done:
            print("Episode finished, resetting environment.")
            obs, info = env.reset()
        else:
            obs = next_obs


if __name__ == "__main__":
    main()


