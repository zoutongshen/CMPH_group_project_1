"""
Energy plotting script for MD simulation.

Three modes:
  Default:       Load energies.npy and plot KE, PE, total energy vs time.
  --compare-dt:  Run identical initial conditions with several dt values and
                 compare energy conservation.
  --collision:   Run a collision course and plot KE/PE exchange.

Author: COP 2026 Project 1
Date: February 2026
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from md_simulation import (run_simulation, initialize_particles,
                           initialize_collision_course)


def plot_energies_from_file(energies_file: str = "energies.npy", dt: float = 0.001):
    """
    Load saved energies and plot KE, PE, and total energy vs time.

    Top panel: all three energies. Bottom panel: total energy alone to
    highlight conservation (or lack thereof with Euler integration).
    """
    energies = np.load(energies_file)
    n_steps = energies.shape[0]
    time = np.arange(n_steps) * dt

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_top.plot(time, energies[:, 0], label="Kinetic Energy")
    ax_top.plot(time, energies[:, 1], label="Potential Energy")
    ax_top.plot(time, energies[:, 2], label="Total Energy", linewidth=2)
    ax_top.set_ylabel("Energy (reduced units)")
    ax_top.legend()
    ax_top.set_title("Energy Evolution")
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(time, energies[:, 2], color="C2", linewidth=2)
    ax_bot.set_xlabel("Time (reduced units)")
    ax_bot.set_ylabel("Total Energy (reduced units)")
    ax_bot.set_title("Total Energy Conservation Check")
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("energy_evolution.png", dpi=150)
    print("Saved plot to 'energy_evolution.png'")
    plt.show()


def compare_timesteps():
    """
    Run simulations with the same initial conditions but different dt values.

    Demonstrates how Euler integration energy drift worsens with larger dt.
    """
    dt_values = [0.001, 0.004, 0.01, 0.02, 0.05]
    n_particles = 4
    box_size = 6
    n_steps = 5000
    temperature = 1.0
    dimensions = 3

    # Generate fixed initial conditions
    np.random.seed(42)
    positions, velocities = initialize_particles(n_particles, box_size, temperature,
                                                 dimensions, min_distance=1.5)

    fig, axes = plt.subplots(len(dt_values), 1, figsize=(10, 3 * len(dt_values)),
                             sharex=False)

    for idx, dt in enumerate(dt_values):
        print(f"Running dt = {dt} ...")
        _, energies = run_simulation(
            n_particles, box_size, n_steps, dt,
            dimensions=dimensions, track_energy=True,
            initial_positions=positions, initial_velocities=velocities)

        time = np.arange(n_steps) * dt
        ax = axes[idx]
        ax.plot(time, energies[:, 0], label="KE", alpha=0.8)
        ax.plot(time, energies[:, 1], label="PE", alpha=0.8)
        ax.plot(time, energies[:, 2], label="Total", linewidth=2)
        ax.set_ylabel("Energy")
        ax.set_title(f"dt = {dt}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (reduced units)")
    fig.suptitle("Time-Step Comparison (Euler Integration)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("energy_dt_comparison.png", dpi=150)
    print("Saved plot to 'energy_dt_comparison.png'")
    plt.show()


def plot_collision():
    """
    Run a collision course scenario and plot the KE/PE energy exchange.
    """
    box_size = 10
    n_steps = 10000
    dt = 0.001
    dimensions = 3

    positions, velocities = initialize_collision_course(box_size, dimensions)

    print("Running collision course simulation ...")
    _, energies = run_simulation(
        2, box_size, n_steps, dt,
        dimensions=dimensions, track_energy=True,
        initial_positions=positions, initial_velocities=velocities)

    time = np.arange(n_steps) * dt

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_top.plot(time, energies[:, 0], label="Kinetic Energy")
    ax_top.plot(time, energies[:, 1], label="Potential Energy")
    ax_top.plot(time, energies[:, 2], label="Total Energy", linewidth=2)
    ax_top.set_ylabel("Energy (reduced units)")
    ax_top.legend()
    ax_top.set_title("Collision Course: Energy Exchange")
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(time, energies[:, 2], color="C2", linewidth=2)
    ax_bot.set_xlabel("Time (reduced units)")
    ax_bot.set_ylabel("Total Energy (reduced units)")
    ax_bot.set_title("Total Energy During Collision")
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("energy_collision.png", dpi=150)
    print("Saved plot to 'energy_collision.png'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot energy evolution from MD simulation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--compare-dt", action="store_true",
                       help="Run same initial conditions with several dt values and compare")
    group.add_argument("--collision", action="store_true",
                       help="Run collision course scenario and plot energy exchange")
    args = parser.parse_args()

    if args.compare_dt:
        compare_timesteps()
    elif args.collision:
        plot_collision()
    else:
        plot_energies_from_file()


if __name__ == "__main__":
    main()
