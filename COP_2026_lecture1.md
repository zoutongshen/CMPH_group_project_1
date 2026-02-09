# Computational Physics 2026 - Lecture 1

**Matthieu Schaller**

## Course Overview

This course can be taken either as a **3 EC (A)** or **6 EC (A+B)** course.

### Course Structure

**First 6 weeks (Project 1):**
- Code a Molecular Dynamics simulation of Argon atoms
- Study physical properties at different temperatures and densities

**Second half (3 EC additional, Projects 2 & 3):**

**Weeks 7-9 (Project 2):**
- Monte-Carlo simulation of the XY model
- Study magnetization and vortices
- Focus on error analysis

**Weeks 10-12 (Project 3):**
- Choose your own project from a pool of themes
- Can discuss custom projects based on your interests

### Assessment

Course assessment is based on:
- Reports for each project
- Simulation programs you wrote
- Discussion and interpretation of results

---

## Project 1: Molecular Dynamics Simulation of Argon

### Introduction

In our daily lives we encounter different phases of matter. For instance, water can be ice covering lakes, liquid from the tap, or vapor when boiled. In this assignment you write a simulation code to explore different phases of matter quantitatively for a simpler system: **Argon atoms**.

> **Why Argon?** Water is surprisingly complicated and still an ongoing field of research. We need something simpler, and noble gases are ideal since we don't have to worry about molecule formation.

> **Question for you:** Why not Helium?

![MD simulation showing Argon atoms in 3D box with periodic boundary conditions]

**Figure 1:** MD simulation of a system of Argon atoms in a three-dimensional box with periodic boundary conditions. The density ρ and temperature T correspond to the liquid state (specifically ρ = 0.8 and T = 1.0 in dimensionless units). Shown are the last 100 time steps in an equilibrated sample.

---

## Molecular Dynamics (MD) Simulations

### What is Molecular Dynamics?

Molecular dynamics simulations solve the **equation of motion** of a system of particles numerically. The method is indispensable as most systems of interacting particles cannot be solved analytically.

**Historical Context:**
- Even 2 bodies (celestial) can be solved exactly
- 3 bodies → no general solution (three body problem)
- Before computers: Halley's comet trajectory solved by hand
- MD approximates continuous trajectory by discrete time steps

**Example:** Figure 1 shows an MD simulation of 108 Argon atoms in a 3D box. For each atom, the last 100 positions are shown. With small time steps, trajectories look almost continuous.

### Governing Equations

The motion of each particle is governed by **Newton's second law:**

$$m \frac{d^2\mathbf{x}}{dt^2} = \mathbf{F}(\mathbf{x}) = -\nabla U(\mathbf{x})$$

For a potential that only depends on distance $r = \sqrt{x^2 + y^2 + z^2}$:

$$\nabla U(\mathbf{x}) = \frac{dU}{dr} \frac{\mathbf{x}}{r}$$

For pair interactions, the force on atom $i$ is:

$$\mathbf{F}_i = \mathbf{F}(\mathbf{x}_i) = -\sum_j \nabla U(\mathbf{x}_i - \mathbf{x}_j)$$

where the sum runs over all other atoms $j$ in the system.

---

## Numerical Integration

### The Challenge

These equations usually cannot be solved analytically or even with perturbation theory. In MD simulations they are solved **approximately** by replacing continuous time evolution with finite time steps $h$.

### Euler Method (Simple but Flawed)

If $\mathbf{x}_n$ are the positions and $\mathbf{v}_n$ the velocities at time $t_n$, the positions and velocities at the next time point $t_{n+1} = t_n + h$ are:

$$\mathbf{x}_{n+1} = \mathbf{x}_n + \mathbf{v}_n h$$

$$\mathbf{v}_{n+1} = \mathbf{v}_n + \frac{\mathbf{F}(\mathbf{x}_n)}{m} h$$

**Major Downside:** Euler methods **do not conserve energy**. 

Since our goal is to simulate a system with conserved energy (microcanonical ensemble from statistical mechanics), we need a better algorithm. However, for now, we'll use this simple approach. We'll revisit this in the third lecture.

---

## The Argon System

### Why Different Phases?

Depending on externally imposed conditions (temperature and density), the system can be in:
- Gaseous state
- Liquid state  
- Solid state

### Interactions Between Argon Atoms

Unlike an ideal gas, atoms interact. Argon atoms are neutral, so no Coulomb interaction. Instead:

1. **Attractive force (van der Waals):** Small dipole moments from electron cloud displacement → interaction scales as $1/r^6$

2. **Repulsive force (Pauli repulsion):** Quantum mechanical origin at short distances

### Lennard-Jones Potential

Both aspects are captured in the **Lennard-Jones potential:**

$$U(r) = 4\epsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right]$$

- The repulsive $(σ/r)^{12}$ term has historic reasons (convenient as it's the square of the attractive term)
- For Argon: $\epsilon/k_B = 119.8$ K and $\sigma = 3.405$ Å

---

## System Boundaries

### The Problem

We want to simulate an **infinite system** of Argon atoms (not interactions with walls), but a computer cannot simulate infinite space.

### Solution: Periodic Boundary Conditions

Use a finite box of length $L$ with **periodic boundary conditions** to mimic an infinite system.

#### Two Important Consequences:

1. **If a particle leaves the box, it reenters at the opposite side**
   - Question for you: What happens to the velocities?

2. **Interaction evaluation is influenced by periodic boundaries**
   - Introduces an infinite number of copies of particles around the simulation box

---

## Minimum Image Convention

### The Challenge with Infinite Copies

Periodic boundaries create infinite copies of particles. However, the Lennard-Jones potential decays fast ($r^{-6}$), so:

- Only the **closest copy** of each particle contributes significantly
- All other copies can be safely neglected
- This is called the **minimum image convention**

### Example (Figure 2)

![Minimum image convention diagram]

**Figure 2:** The minimum image convention in 2D. 

- Central box: 5 particles (the simulation)
- Surrounding boxes: periodic images (not part of simulation, shown for explanation)
- Orange particle interacts with 4 purple particles
- In principle: infinite copies exist along each dimension
- In practice: Only interact with the **closest "version"** of each particle
- The four interactions to consider are shown as dashed lines

Since $U(r)$ decays as $r^{-6}$, copies contribute negligibly and can be safely neglected.

---

## Implementation Notes

### Time Evolution

Particle positions and velocities evolve according to Eqs. 3 and 4 (Euler method).

### Data Structure

**Recommended:** Store particle positions and velocities in **numpy arrays**

**Benefits:**
- Compact code representation
- Efficient operations
- Easy to represent the whole system of equations

---

## First Milestone

### Task: Code Up and Explore the System

**Start simple and build up:**

1. **Begin with 2D** (easier to visualize), plan to switch to 3D later

2. **Start small:** 
   - Two particles (e.g., on collision course)
   - Then three particles
   - Gradually increase

3. **Add complexity:**
   - Random initial positions and velocities
   - Forces from Lennard-Jones potential
   - Minimum image convention for periodic boundaries

4. **Store and visualize:**
   - Save particle trajectories to file
   - Display using scatter plots (e.g., matplotlib)

### Key Steps

- Simulate time evolution in a periodic box
- Apply Lennard-Jones forces with minimum image convention
- Store trajectories
- Visualize results

---

## Summary

This project will guide you through:
- Implementing a molecular dynamics simulation
- Understanding phase transitions in Argon
- Working with periodic boundary conditions
- Numerical integration of equations of motion
- Visualization of particle systems

Start simple, test thoroughly, and gradually build complexity!
