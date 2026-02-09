"""
Molecular Dynamics Simulation of Argon Atoms

This module implements a basic MD simulation using the Lennard-Jones potential
with periodic boundary conditions and the minimum image convention.

Author: COP 2026 Project 1
Date: February 2026
"""

import numpy as np
from typing import Tuple


def lennard_jones_potential(r: float, epsilon: float = 1.0, sigma: float = 1.0) -> float:
    """
    Calculate Lennard-Jones potential energy for a given distance.
    
    U(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
    
    Args:
        r: Distance between two particles
        epsilon: LJ energy parameter (default: 1.0 in reduced units)
        sigma: LJ length parameter (default: 1.0 in reduced units)
    
    Returns:
        Potential energy at distance r
    """
    sigma_over_r = sigma / r
    return 4 * epsilon * (sigma_over_r**12 - sigma_over_r**6)


def lennard_jones_force_magnitude(r: float, epsilon: float = 1.0, sigma: float = 1.0) -> float:
    """
    Calculate magnitude of Lennard-Jones force from the potential.
    
    Force is F = -dU/dr, where U(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
    This gives: F = 24*epsilon/r * [2*(sigma/r)^12 - (sigma/r)^6]
    
    Args:
        r: Distance between two particles
        epsilon: LJ energy parameter (default: 1.0 in reduced units)
        sigma: LJ length parameter (default: 1.0 in reduced units)
    
    Returns:
        Magnitude of force at distance r
    """
    sigma_over_r = sigma / r
    return 24 * epsilon / r * (2 * sigma_over_r**12 - sigma_over_r**6)


def forces_and_potential(positions: np.ndarray, box_size: float,
                         epsilon: float = 1.0, sigma: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Compute forces on all particles and total potential energy from Lennard-Jones interactions.
    
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
            
            # Apply minimum image convention: wrap to nearest image
            delta = delta - box_size * np.round(delta / box_size)
            
            # Calculate distance
            r = np.linalg.norm(delta)
            
            # Skip if particles are too close (avoid division by zero)
            if r < 0.01 * sigma:
                continue
            
            # Calculate potential energy for this pair
            potential = lennard_jones_potential(r, epsilon, sigma)
            total_potential += potential
            
            # Calculate force magnitude from the potential derivative
            force_mag = lennard_jones_force_magnitude(r, epsilon, sigma)
            
            # Force vector (pointing from j to i)
            force_vector = force_mag * delta / r
            
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


def euler_step(positions: np.ndarray, velocities: np.ndarray, 
               forces: np.ndarray, dt: float, mass: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one Euler integration step.
    
    Updates positions and velocities using the Euler method:
    x(t+dt) = x(t) + v(t)*dt
    v(t+dt) = v(t) + F(t)/m*dt
    
    Args:
        positions: Current positions
        velocities: Current velocities
        forces: Current forces
        dt: Time step
        mass: Particle mass (default: 1.0 in reduced units)
    
    Returns:
        Tuple of (new_positions, new_velocities)
    """
    new_positions = positions + velocities * dt
    new_velocities = velocities + forces / mass * dt
    
    return new_positions, new_velocities


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
                # Apply minimum image convention
                delta = delta - box_size * np.round(delta / box_size)
                r = np.linalg.norm(delta)
                
                if r < min_distance:
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
                   dimensions: int = 2, track_energy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run molecular dynamics simulation.
    
    Args:
        n_particles: Number of particles to simulate
        box_size: Size of the cubic/square simulation box
        n_steps: Number of time steps to simulate
        dt: Time step size
        temperature: Initial temperature
        dimensions: Number of spatial dimensions (2 or 3)
        track_energy: Whether to track and return energy values
    
    Returns:
        If track_energy is True:
            Tuple of (trajectories, energies)
            - trajectories: Array of shape (n_steps, n_particles, dimensions)
            - energies: Array of shape (n_steps, 3) with [KE, PE, Total] for each step
        If track_energy is False:
            Just trajectories array
    """
    # Initialize with minimum distance of 1.5*sigma to avoid strong initial forces
    positions, velocities = initialize_particles(n_particles, box_size, temperature, 
                                                dimensions, min_distance=1.5)
    
    # Storage for trajectories and energies
    trajectories = np.zeros((n_steps, n_particles, dimensions))
    if track_energy:
        energies = np.zeros((n_steps, 3))  # [kinetic, potential, total]
    
    # Main simulation loop
    for step in range(n_steps):
        # Store current positions
        trajectories[step] = positions
        
        # Get forces and potential energy
        forces, potential_energy = forces_and_potential(positions, box_size)
        
        # Track energies if requested
        if track_energy:
            ke = kinetic_energy(velocities)
            energies[step] = [ke, potential_energy, ke + potential_energy]
        
        # Integrate equations of motion (Euler method)
        positions, velocities = euler_step(positions, velocities, forces, dt)
        
        # Apply periodic boundary conditions
        positions = apply_periodic_boundaries(positions, box_size)
        
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
    
    Modify parameters here to explore different conditions.
    """
    # Simulation parameters
    n_particles = 2
    box_size = 4
    n_steps = 1000  # Need ~100-1000 steps to see clear trajectories
    dt = 0.001  # Small timestep needed for Euler stability (Euler is not energy-conserving!)
    temperature = 1.0
    dimensions = 2  # Start with 2D for visualization
    
    print("Starting MD simulation...")
    print(f"Parameters: {n_particles} particles, box size {box_size}")
    print(f"Time steps: {n_steps}, dt = {dt}")
    print(f"Initial temperature: {temperature}")
    print(f"Dimensions: {dimensions}D")
    print()
    
    # Run simulation with energy tracking
    trajectories, energies = run_simulation(n_particles, box_size, n_steps, dt, 
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


if __name__ == "__main__":
    main()
