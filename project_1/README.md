# Project 1: Molecular Dynamics Simulation of Argon

## Overview

This project implements a Molecular Dynamics (MD) simulation of Argon atoms using the Lennard-Jones potential. The simulation can explore different phases of matter (solid, liquid, gas) by varying temperature and density.

## Files

- `md_simulation.py`: Main simulation code with MD integration
- `visualize.py`: Script to visualize saved trajectories
- `README.md`: This file

## How to Run

### Run a simulation:

```bash
python md_simulation.py
```

### Parameters to modify (in `md_simulation.py`):

In the `main()` function, you can adjust:
- `n_particles`: Number of particles (default: 5)
- `box_size`: Size of the simulation box (default: 10.0)
- `n_steps`: Number of time steps (default: 1000)
- `dt`: Time step size (default: 0.01)
- `temperature`: Initial temperature (default: 1.0)
- `dimensions`: 2D or 3D simulation (default: 2)

### Visualize results:

```bash
python visualize.py
```

This will create a scatter plot animation of the particle trajectories.

## Physics Parameters

**Lennard-Jones potential for Argon:**
- ε/k_B = 119.8 K
- σ = 3.405 Å

The simulation uses dimensionless units where ε and σ are set to 1.

## Implementation Notes

- Uses Euler integration (simple but doesn't conserve energy - will be improved in later lectures)
- Periodic boundary conditions with minimum image convention
- Stores trajectories in `trajectories.npy` file
- Start with 2D for easier visualization, can switch to 3D by changing `dimensions` parameter

## Next Steps

1. Test with 2 particles on collision course
2. Gradually increase number of particles
3. Verify periodic boundaries work correctly
4. Switch to 3D
5. Implement better integration scheme (e.g., Velocity Verlet)
6. Add energy conservation checks
