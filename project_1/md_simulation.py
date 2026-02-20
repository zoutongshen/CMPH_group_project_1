"""
Molecular Dynamics Simulation of Argon Atoms

This module implements an MD simulation using the Lennard-Jones potential
with periodic boundary conditions, the minimum image convention, and the
Velocity-Verlet integration algorithm for energy conservation.

Author: CMPH 2026 Project 1
Date: February 2026
"""

import argparse
import numpy as np
from typing import Tuple


def lennard_jones_potential(distance: float, epsilon: float = 1.0, sigma: float = 1.0) -> float:
    """
    Calculate Lennard-Jones potential energy for a given distance.

    U(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]

    Args:
        distance: Distance between two particles
        epsilon: LJ energy parameter (default: 1.0 in reduced units)
        sigma: LJ length parameter (default: 1.0 in reduced units)

    Returns:
        Potential energy at the given distance
    """
    sigma_over_r = sigma / distance
    return 4 * epsilon * (sigma_over_r**12 - sigma_over_r**6)


def lennard_jones_force_magnitude(distance: float, epsilon: float = 1.0, sigma: float = 1.0) -> float:
    """
    Calculate magnitude of Lennard-Jones force from the potential.

    Force is F = -dU/dr, where U(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
    This gives: F = 24*epsilon/r * [2*(sigma/r)^12 - (sigma/r)^6]

    Args:
        distance: Distance between two particles
        epsilon: LJ energy parameter (default: 1.0 in reduced units)
        sigma: LJ length parameter (default: 1.0 in reduced units)

    Returns:
        Magnitude of force at the given distance
    """
    sigma_over_r = sigma / distance
    return 24 * epsilon / distance * (2 * sigma_over_r**12 - sigma_over_r**6)


def forces_and_potential(positions: np.ndarray, box_size: float,
                         epsilon: float = 1.0, sigma: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Compute forces on all particles and total potential energy from Lennard-Jones interactions.

    Both quantities are computed in a single pair-loop over particles, which is
    standard MD practice since force and potential share the same distance calculation.

    Uses the minimum image convention for periodic boundary conditions.
    Forces are derived from the Lennard-Jones potential: F = -dU/dr
    
    Args:
        positions: Array of shape (n_particles, dimensions) with particle positions
        box_size: Size of the periodic simulation box
        epsilon: LJ energy parameter (default: 1.0 in reduced units)
        sigma: LJ length parameter (default: 1.0 in reduced units)
    
    Returns:
        Tuple of (forces, total_potential_energy)
        - forces: Array of shape (n_particles, dimensions)
        - total_potential_energy: Float, sum of all pairwise potentials
    """
    n_particles = positions.shape[0]
    forces = np.zeros_like(positions)
    total_potential = 0.0
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # Calculate distance vector with minimum image convention
            delta = positions[i] - positions[j]
            
            # Apply minimum image convention using modulo trick
            delta = (delta + box_size / 2) % box_size - box_size / 2
            
            # Calculate distance
            distance = np.linalg.norm(delta)

            # Skip if particles are too close (avoid division by zero)
            if distance < 0.01 * sigma:
                continue

            # Calculate potential energy for this pair
            potential = lennard_jones_potential(distance, epsilon, sigma)
            total_potential += potential

            # Calculate force magnitude from the potential derivative
            force_mag = lennard_jones_force_magnitude(distance, epsilon, sigma)

            # Force vector (pointing from j to i)
            force_vector = force_mag * delta / distance
            
            # Newton's third law: equal and opposite forces
            forces[i] += force_vector
            forces[j] -= force_vector
    
    return forces, total_potential


def apply_periodic_boundaries(positions: np.ndarray, box_size: float) -> np.ndarray:
    """
    Apply periodic boundary conditions to particle positions.
    
    If a particle leaves the box, it re-enters from the opposite side.
    
    Args:
        positions: Array of shape (n_particles, dimensions) with particle positions
        box_size: Size of the periodic simulation box
    
    Returns:
        Wrapped positions within [0, box_size)
    """
    return positions % box_size


def velocity_verlet_positions(positions: np.ndarray, velocities: np.ndarray,
                              forces: np.ndarray, dt: float,
                              mass: float = 1.0) -> np.ndarray:
    """
    Velocity Verlet position update (first half of the algorithm).

    x(t+dt) = x(t) + v(t)*dt + dt^2/(2m) * F(t)

    Args:
        positions: Current positions
        velocities: Current velocities
        forces: Current forces on all particles
        dt: Time step
        mass: Particle mass (default: 1.0 in reduced units)

    Returns:
        Updated positions at time t+dt
    """
    return positions + velocities * dt + 0.5 * forces / mass * dt**2


def velocity_verlet_velocities(velocities: np.ndarray, forces_old: np.ndarray,
                               forces_new: np.ndarray, dt: float,
                               mass: float = 1.0) -> np.ndarray:
    """
    Velocity Verlet velocity update (second half of the algorithm).

    v(t+dt) = v(t) + dt/(2m) * [F(t) + F(t+dt)]

    Args:
        velocities: Current velocities
        forces_old: Forces at time t
        forces_new: Forces at time t+dt (computed from updated positions)
        dt: Time step
        mass: Particle mass (default: 1.0 in reduced units)

    Returns:
        Updated velocities at time t+dt
    """
    return velocities + 0.5 * dt / mass * (forces_old + forces_new)


def initialize_particles(n_particles: int, box_size: float, 
                        temperature: float, dimensions: int = 2,
                        min_distance: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize particle positions and velocities.
    
    Positions are randomly distributed in the box with a minimum separation
    to avoid particles starting too close (which causes huge forces).
    Velocities are drawn from Maxwell-Boltzmann distribution and
    shifted to have zero total momentum.
    
    Args:
        n_particles: Number of particles
        box_size: Size of simulation box
        temperature: Initial temperature (in reduced units)
        dimensions: Number of spatial dimensions (2 or 3)
        min_distance: Minimum allowed distance between particles (default: 1.0 = sigma)
    
    Returns:
        Tuple of (positions, velocities)
    """
    # Initialize positions with minimum distance constraint
    positions = np.zeros((n_particles, dimensions))
    
    for i in range(n_particles):
        max_attempts = 1000
        for attempt in range(max_attempts):
            # Try a random position
            new_pos = np.random.uniform(0, box_size, size=dimensions)
            
            # Check distance to all existing particles
            too_close = False
            for j in range(i):
                delta = new_pos - positions[j]
                # Apply minimum image convention using modulo trick
                delta = (delta + box_size / 2) % box_size - box_size / 2
                distance = np.linalg.norm(delta)

                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                positions[i] = new_pos
                break
        else:
            # If we couldn't find a position after max_attempts, relax constraint
            print(f"Warning: Could not place particle {i} with min_distance={min_distance}")
            print(f"         Using last attempted position. Consider larger box or fewer particles.")
            positions[i] = new_pos
    
    # Velocities from Maxwell-Boltzmann distribution
    # In reduced units, k_B = 1, m = 1, so v_rms = sqrt(T)
    velocities = np.random.normal(0, np.sqrt(temperature), size=(n_particles, dimensions))
    
    # Remove center of mass motion (zero total momentum)
    velocities -= np.mean(velocities, axis=0)
    
    return positions, velocities


def initialize_collision_course(box_size: float, dimensions: int = 3,
                                separation: float = 3.0, speed: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize 2 particles on a head-on collision course along the x-axis.

    Particles are placed symmetrically about the box center, moving toward each
    other with equal and opposite velocities (zero total momentum by construction).

    Args:
        box_size: Size of the simulation box
        dimensions: Number of spatial dimensions (2 or 3)
        separation: Initial distance between particles (default: 3.0 sigma)
        speed: Speed of each particle (default: 1.0)

    Returns:
        Tuple of (positions, velocities), each of shape (2, dimensions)
    """
    center = box_size / 2.0
    positions = np.zeros((2, dimensions))
    velocities = np.zeros((2, dimensions))

    # Place particles along x-axis, centered in box
    positions[0, 0] = center - separation / 2.0
    positions[1, 0] = center + separation / 2.0

    # Center other coordinates in box
    for dim in range(1, dimensions):
        positions[0, dim] = center
        positions[1, dim] = center

    # Move toward each other along x-axis
    velocities[0, 0] = +speed
    velocities[1, 0] = -speed

    return positions, velocities


def kinetic_energy(velocities: np.ndarray, mass: float = 1.0) -> float:
    """
    Total kinetic energy of the system.
    
    KE = (1/2) * m * sum(v^2)
    
    Args:
        velocities: Array of shape (n_particles, dimensions)
        mass: Particle mass (default: 1.0 in reduced units)
    
    Returns:
        Total kinetic energy
    """
    return 0.5 * mass * np.sum(velocities**2)


def run_simulation(n_particles: int, box_size: float, n_steps: int,
                   dt: float, temperature: float = 1.0,
                   dimensions: int = 3, track_energy: bool = True,
                   initial_positions: np.ndarray = None,
                   initial_velocities: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run molecular dynamics simulation.

    Args:
        n_particles: Number of particles to simulate
        box_size: Size of the cubic/square simulation box
        n_steps: Number of time steps to simulate
        dt: Time step size
        temperature: Initial temperature (used only when generating random initial conditions)
        dimensions: Number of spatial dimensions (2 or 3)
        track_energy: Whether to track and return energy values
        initial_positions: Optional pre-set positions of shape (n_particles, dimensions).
                          If None, positions are randomly initialized.
        initial_velocities: Optional pre-set velocities of shape (n_particles, dimensions).
                           If None, velocities are drawn from Maxwell-Boltzmann distribution.

    Returns:
        Tuple of (trajectories, energies)
        - trajectories: Array of shape (n_steps, n_particles, dimensions)
        - energies: Array of shape (n_steps, 3) with [KE, PE, Total] for each step,
                    or None if track_energy is False
    """
    if initial_positions is not None and initial_velocities is not None:
        positions = initial_positions.copy()
        velocities = initial_velocities.copy()
    else:
        # Initialize with minimum distance of 1.5*sigma to avoid strong initial forces
        positions, velocities = initialize_particles(n_particles, box_size, temperature,
                                                    dimensions, min_distance=1.5)
    
    # Storage for trajectories and energies
    trajectories = np.zeros((n_steps, n_particles, dimensions))
    if track_energy:
        energies = np.zeros((n_steps, 3))  # [kinetic, potential, total]
    
    # Compute initial forces for the Velocity-Verlet algorithm
    forces, potential_energy = forces_and_potential(positions, box_size)

    # Main simulation loop (Velocity-Verlet integration)
    for step in range(n_steps):
        # Store current positions
        trajectories[step] = positions

        # Track energies if requested
        if track_energy:
            ke = kinetic_energy(velocities)
            energies[step] = [ke, potential_energy, ke + potential_energy]

        # Step 1: Update positions
        positions = velocity_verlet_positions(positions, velocities, forces, dt)

        # Apply periodic boundary conditions
        positions = apply_periodic_boundaries(positions, box_size)

        # Step 2: Compute new forces at the updated positions
        forces_new, potential_energy = forces_and_potential(positions, box_size)

        # Step 3: Update velocities using old and new forces
        velocities = velocity_verlet_velocities(velocities, forces, forces_new, dt)

        # Store new forces for next step
        forces = forces_new

        # Print progress every 100 steps
        if (step + 1) % 100 == 0:
            if track_energy:
                print(f"Step {step + 1}/{n_steps}: KE={energies[step, 0]:.3f}, PE={energies[step, 1]:.3f}, Total={energies[step, 2]:.3f}")
            else:
                print(f"Step {step + 1}/{n_steps} completed")
    
    if track_energy:
        return trajectories, energies
    return trajectories, None


def main():
    """
    Main function to run the simulation.

    Use --collision flag for a 2-particle collision course scenario.
    Otherwise runs a default 3D simulation with 4 particles.
    """
    parser = argparse.ArgumentParser(description="Molecular Dynamics Simulation")
    parser.add_argument("--collision", action="store_true",
                        help="Run 2-particle collision course scenario")
    args = parser.parse_args()

    if args.collision:
        # Collision course scenario
        n_particles = 2
        box_size = 10
        n_steps = 10000
        dt = 0.001
        dimensions = 3

        positions, velocities = initialize_collision_course(box_size, dimensions)

        print("Starting MD simulation (collision course)...")
        print(f"Parameters: {n_particles} particles, box size {box_size}")
        print(f"Time steps: {n_steps}, dt = {dt}")
        print(f"Dimensions: {dimensions}D")
        print(f"Initial separation: 3.0 sigma, speed: 1.0")
        print()

        trajectories, energies = run_simulation(
            n_particles, box_size, n_steps, dt,
            dimensions=dimensions, track_energy=True,
            initial_positions=positions, initial_velocities=velocities)
    else:
        # Default simulation parameters
        n_particles = 15
        box_size = 6
        n_steps = 5000
        dt = 0.001
        temperature = 1.0
        dimensions = 3

        print("Starting MD simulation...")
        print(f"Parameters: {n_particles} particles, box size {box_size}")
        print(f"Time steps: {n_steps}, dt = {dt}")
        print(f"Initial temperature: {temperature}")
        print(f"Dimensions: {dimensions}D")
        print()

        trajectories, energies = run_simulation(
            n_particles, box_size, n_steps, dt,
            temperature, dimensions, track_energy=True)

    # Save trajectories and energies
    np.save('trajectories.npy', trajectories)
    if energies is not None:
        np.save('energies.npy', energies)

    print()
    print(f"Simulation complete!")
    print(f"Trajectories saved to 'trajectories.npy' with shape: {trajectories.shape}")
    if energies is not None:
        print(f"Energies saved to 'energies.npy' with shape: {energies.shape}")
        print(f"Final energy - KE: {energies[-1, 0]:.3f}, PE: {energies[-1, 1]:.3f}, Total: {energies[-1, 2]:.3f}")
    print("Run 'python visualize.py' to visualize the results.")
    print("Run 'python plot_energies.py' to plot energy evolution.")


if __name__ == "__main__":
    main()
