"""
Visualization script for MD simulation trajectories.

Loads saved trajectories and creates scatter plots to visualize
particle motion over time.

Author: COP 2026 Project 1
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_trajectories_2d(trajectories: np.ndarray, show_last_n: int = 1000):
    """
    Plot 2D trajectories showing particle paths.
    
    Args:
        trajectories: Array of shape (n_steps, n_particles, 2)
        show_last_n: Number of recent positions to show for each particle
    """
    n_steps, n_particles, _ = trajectories.shape
    
    plt.figure(figsize=(10, 10))
    
    # Use different colors for each particle
    colors = plt.cm.rainbow(np.linspace(0, 1, n_particles))
    
    for i in range(n_particles):
        # Show last N positions for each particle
        start_idx = max(0, n_steps - show_last_n)
        x = trajectories[start_idx:, i, 0]
        y = trajectories[start_idx:, i, 1]
        
        # Plot trajectory
        plt.plot(x, y, '-', color=colors[i], alpha=0.3, linewidth=1)
        
        # Plot current position
        plt.scatter(x[-1], y[-1], color=colors[i], s=100, 
                   edgecolors='black', linewidth=2, zorder=5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'MD Simulation - Last {show_last_n} time steps')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('trajectories_2d.png', dpi=150)
    print("Saved plot to 'trajectories_2d.png'")
    plt.show()


def plot_trajectories_3d(trajectories: np.ndarray, show_last_n: int = 1000):
    """
    Plot 3D trajectories showing particle paths.
    
    Args:
        trajectories: Array of shape (n_steps, n_particles, 3)
        show_last_n: Number of recent positions to show for each particle
    """
    n_steps, n_particles, _ = trajectories.shape
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use different colors for each particle
    colors = plt.cm.rainbow(np.linspace(0, 1, n_particles))
    
    for i in range(n_particles):
        # Show last N positions for each particle
        start_idx = max(0, n_steps - show_last_n)
        x = trajectories[start_idx:, i, 0]
        y = trajectories[start_idx:, i, 1]
        z = trajectories[start_idx:, i, 2]
        
        # Plot trajectory
        ax.plot(x, y, z, '-', color=colors[i], alpha=0.3, linewidth=1)
        
        # Plot current position
        ax.scatter(x[-1], y[-1], z[-1], color=colors[i], s=100, 
                  edgecolors='black', linewidth=2, zorder=5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'MD Simulation - Last {show_last_n} time steps')
    plt.tight_layout()
    plt.savefig('trajectories_3d.png', dpi=150)
    print("Saved plot to 'trajectories_3d.png'")
    plt.show()


def create_animation_2d(trajectories: np.ndarray, box_size: float = 10.0, 
                       interval: int = 20, skip_frames: int = 10):
    """
    Create an animation of the 2D simulation.
    
    Args:
        trajectories: Array of shape (n_steps, n_particles, 2)
        box_size: Size of the simulation box for setting axis limits
        interval: Delay between frames in milliseconds
        skip_frames: Only show every Nth frame to speed up animation (default 10 for dt=0.001)
    """
    n_steps, n_particles, _ = trajectories.shape
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Initialize scatter plot
    colors = plt.cm.rainbow(np.linspace(0, 1, n_particles))
    scat = ax.scatter(trajectories[0, :, 0], trajectories[0, :, 1], s=200, c=colors, edgecolors='black', linewidth=2)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        scat.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scat, time_text
    
    def update(frame):
        frame_idx = frame * skip_frames
        if frame_idx >= n_steps:
            frame_idx = n_steps - 1
        
        positions = trajectories[frame_idx]
        scat.set_offsets(positions)
        time_text.set_text(f'Step: {frame_idx}/{n_steps}')
        return scat, time_text
    
    n_frames = n_steps // skip_frames
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                        interval=interval, blit=True, repeat=True)
    
    plt.title('MD Simulation Animation')
    plt.tight_layout()
    plt.show()
    
    return anim


def main():
    """
    Main function to load and visualize trajectories.
    """
    # Load trajectories
    try:
        trajectories = np.load('trajectories.npy')
    except FileNotFoundError:
        print("Error: 'trajectories.npy' not found.")
        print("Please run 'python md_simulation.py' first.")
        return
    
    n_steps, n_particles, dimensions = trajectories.shape
    
    print(f"Loaded trajectories: {n_steps} steps, {n_particles} particles, {dimensions}D")
    print()
    
    # Visualize based on dimensions
    if dimensions == 2:
        print("Creating 2D trajectory plot...")
        # With dt=0.001, show more steps to cover same physical time
        plot_trajectories_2d(trajectories, show_last_n=1000)
        
        # Optionally create animation (commented out by default as it can be slow)
        print("Creating animation...")
        create_animation_2d(trajectories, box_size=10.0, interval=20, skip_frames=10)
        
    elif dimensions == 3:
        print("Creating 3D trajectory plot...")
        plot_trajectories_3d(trajectories, show_last_n=5000)
    else:
        print(f"Error: Cannot visualize {dimensions}D trajectories")


if __name__ == "__main__":
    main()
