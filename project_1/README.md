# Project 1: Molecular Dynamics Simulation of Argon

## Overview

This project implements a Molecular Dynamics (MD) simulation of Argon atoms using the Lennard-Jones potential. The simulation can explore different phases of matter (solid, liquid, gas) by varying temperature and density.

## Files

- `md_simulation.py`: Main simulation code with MD integration
- `plot_energies.py`: Script to plot energy evolution and compare time-step effects
- `visualize.py`: Script to visualize saved trajectories
- `README.md`: This file

## How to Run

### Run a simulation (default: 4 particles, 3D):

```bash
python md_simulation.py
```

### Run a collision course (2 particles, head-on):

```bash
python md_simulation.py --collision
```

### Plot energy evolution:

```bash
python plot_energies.py
```

This loads `energies.npy` from the most recent simulation and plots kinetic, potential, and total energy vs time.

### Compare time-step effects:

```bash
python plot_energies.py --compare-dt
```

Runs simulations with dt = 0.001, 0.004, 0.01, 0.02, 0.05 using the same initial conditions. Shows how Euler integration energy drift increases with larger time steps.

### Plot collision energy exchange:

```bash
python plot_energies.py --collision
```

Self-contained: runs a collision course and plots the KE/PE energy exchange during the collision event.

### Visualize trajectories:

```bash
python visualize.py
```

Creates a scatter plot (2D) or 3D plot of the particle trajectories.

## Default Parameters

| Parameter    | Default | Collision mode |
|-------------|---------|----------------|
| n_particles | 4       | 2              |
| box_size    | 6       | 10             |
| n_steps     | 5000    | 10000          |
| dt          | 0.001   | 0.001          |
| temperature | 1.0     | N/A            |
| dimensions  | 3       | 3              |

## Physics Parameters

**Lennard-Jones potential for Argon:**
- epsilon/k_B = 119.8 K
- sigma = 3.405 Angstrom

The simulation uses dimensionless (reduced) units where epsilon, sigma, and particle mass are all set to 1.

## Implementation Notes

- Uses Euler integration (simple but doesn't conserve energy - will be improved in later lectures)
- Periodic boundary conditions with minimum image convention
- Saves trajectories to `trajectories.npy` and energies to `energies.npy`
- Variables follow the >1 character naming guideline (e.g. `distance` instead of `r`)

## Next Steps

1. Implement better integration scheme (e.g., Velocity Verlet)
2. Add energy conservation checks
3. Explore larger systems and phase behavior
