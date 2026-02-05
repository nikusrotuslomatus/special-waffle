import autoroot
import autorootcwd
import fire
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn

from pathlib import Path
from mppi_plus_proto.dynamics_models.models import UnicycleSteeringModel
from mppi_plus_proto.trainers.mutual_inf import MIJointTrainer, RolloutCollector
from mppi_plus_proto.nn_models.generative import generic_gaussian, generic_realnvp
from mppi_plus_proto.trainers.data_samplers import sample_uniform


DT = 0.2
HORIZON = 16
SPEED = 1.
ANGULAR_AMPL = float(np.deg2rad(45.))


def plot_trajectories_unicycle(trajectories: np.ndarray,
                               ax: plt.Axes = None,
                               figsize: tuple = (6, 6),
                               title: str = None) -> None:
    """
    Plot multiple unicycle trajectories on a 2D plane.
    
    Args:
        trajectories: Array of shape (n_trajectories, horizon, 3) containing 
                     state trajectories where each state is (x, y, theta)
        ax: Matplotlib axes to plot on. If None, creates new figure
        figsize: Figure size for the plot (only used if ax is None)
        title: Title for the plot
    """
    # Create figure if no axes provided
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    # Plot each trajectory
    for i in range(trajectories.shape[0]):
        traj = trajectories[i]
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.3)
    ax.plot(trajectories[-2, :, 0], trajectories[-2, :, 1], alpha=1., color="red", linestyle="--")
    ax.plot(trajectories[-1, :, 0], trajectories[-1, :, 1], alpha=1., color="blue", linestyle="--")
    
    # Set labels and grid
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Set consistent axis limits based on data
    margin = 0.5
    x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
    y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    if title is not None:
        ax.set_title(title)

    # plt.savefig(output_file, dpi=150, bbox_inches='tight')
    # plt.close()

    # Only show if we created the figure
    # if ax == plt.gca():
    #     plt.show()


def plot_model(ax, 
               model, 
               n_samples: int,
               title: str,
               rollout_collector: RolloutCollector,
               reference_trajectories: np.ndarray):
    with torch.no_grad():
        u_seq, _ = model.sample(n_samples)
        u_seq = u_seq.reshape(n_samples, rollout_collector.horizon, -1)
        x_seq = rollout_collector.rollout(u_seq, clip=True)
        x_seq = x_seq.clone().detach().cpu().numpy()
    
    plot_trajectories_unicycle(np.concat([x_seq, reference_trajectories], axis=0),
                               title=title,
                               ax=ax)


def plot_uniform(ax, 
                 n_samples: int,
                 title: str,
                 rollout_collector: RolloutCollector,
                 reference_trajectories: np.ndarray):
    with torch.no_grad():
        u_seq = sample_uniform(n_samples * rollout_collector.horizon, 
                               lb=rollout_collector.action_lb,
                               ub=rollout_collector.action_ub,
                               device=rollout_collector.action_lb.device)
        u_seq = u_seq.reshape(n_samples, rollout_collector.horizon, -1)
        x_seq = rollout_collector.rollout(u_seq)
        x_seq = x_seq.clone().detach().cpu().numpy()
    
    plot_trajectories_unicycle(np.concat([x_seq, reference_trajectories], axis=0),
                               title=title,
                               ax=ax)


def make_plots(rollout_collector: RolloutCollector,
               experiment_dir: Path,
               model: nn.Module,
               n_samples: int):
    with torch.no_grad():
        max_action = rollout_collector.action_ub
        mid_action = torch.zeros_like(max_action)

        u_seq_max = torch.tile(max_action, (1, rollout_collector.horizon, 1))
        x_seq_max = rollout_collector.rollout(u_seq_max)
        x_seq_max = x_seq_max.clone().detach().cpu().numpy()[0]

        u_seq_mid = torch.tile(mid_action, (1, rollout_collector.horizon, 1))
        x_seq_mid = rollout_collector.rollout(u_seq_mid)
        x_seq_mid = x_seq_mid.clone().detach().cpu().numpy()[0]

        reference = np.stack([x_seq_max, x_seq_mid], axis=0)

    checkpoints = sorted(list(experiment_dir.glob("generator*")))
    n_checkpoints = len(checkpoints)
    
    if n_checkpoints == 0:
        print("No checkpoints found.")
        return
    
    n_cols = 2
    n_rows = (n_checkpoints + 1 + n_cols - 1) // n_cols  # ceiling division
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 10 * n_rows))
    
    # Handle case when there's only 1 row or 1 checkpoint
    if n_checkpoints == 0:
        axs = np.array([[axs, None]])
    elif n_rows == 1:
        axs = axs.reshape(1, -1)
    
    for i, checkpoint in enumerate(checkpoints):
        row = i // n_cols
        col = i % n_cols
        model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
        model.eval()
        name = checkpoint.name.split(".")[0]
        plot_model(axs[row, col], model, n_samples, name, rollout_collector, reference)

    row = i // n_cols
    col = n_checkpoints % n_cols
    plot_uniform(axs[row, col], n_samples, "Action uniform", rollout_collector, reference)
    
    # Hide any unused subplots
    for i in range(n_checkpoints + 1, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(experiment_dir / "samples.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    dynamics = UnicycleSteeringModel(dt=DT,
                                     speed=SPEED,
                                     backend="torch",
                                     device="cuda")
    action_lb = (-ANGULAR_AMPL,)
    action_ub = (ANGULAR_AMPL,)
    rollout_collector = RolloutCollector(dynamics=dynamics,
                                         horizon=HORIZON,
                                         action_lb=action_lb,
                                         action_ub=action_ub,
                                         state_dim_cut=2)

    generator = generic_realnvp(action_dim=rollout_collector.action_dim,
                                horizon=HORIZON,
                                low=action_lb,
                                high=action_ub)
    discriminator = generic_gaussian(state_dim=rollout_collector.state_dim,
                                     action_dim=rollout_collector.action_dim,
                                     horizon=HORIZON)
    
    trainer = MIJointTrainer(rollout_collector=rollout_collector,
                             generator=generator,
                             discriminator=discriminator,
                             batch_size=128,
                             lr=3e-4,
                             wd=0.,
                             scheduler=False,
                             alpha=1.,
                             max_grad_norm=None,
                             device="cuda")
    
    trainer.train(checkpoint_root="checkpoints",
                  experiment_name=f"mi_joint_1d_{HORIZON}",
                  n_epochs=1000,
                  logging_freq=10)
                  
    print("Making plots...")
    make_plots(rollout_collector,
               Path(f"checkpoints/mi_joint_1d_{HORIZON}/"),
               generator,
               10000)


if __name__ == "__main__":
    fire.Fire(main)
