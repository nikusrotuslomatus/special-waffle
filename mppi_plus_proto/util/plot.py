import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(trajectories: np.ndarray, 
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
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.3)
    
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

    # Only show if we created the figure
    if ax == plt.gca():
        plt.show()
