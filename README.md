# High-Performance Rigid Origami Solver

**Computational Geometry | Python, Numba, Scipy**

A high-performance, rigid-body solver for complex origami patterns. The program simulates the kinematics of rigid faces connected by rotational hinges, solving for valid 3D configurations by enforcing geometric closure constraints. 

It leverages **Just-In-Time (JIT) compilation**, **analytical derivatives**, and **sparse linear algebra** to simulate folding mechanics in real-time, scaling near-linearly to thousands of degrees of freedom.

<img width="3472" height="1971" alt="Screenshot 2025-12-05 at 5 27 09 PM" src="https://github.com/user-attachments/assets/f56c3850-7277-4050-a1a1-c7b54e062529" />

*Simulated Huffman Rectangular Weave pattern solving for geometric closure.*

## Mathematical Formulation

The simulation treats origami as a kinematic network rather than a mass-spring system. The system state is defined by the dihedral angles at every fold, governed by geometric closure constraints around each internal vertex.

https://github.com/user-attachments/assets/801652c6-fb08-45b8-bc42-99f42a1753af

*Visualization of the loop closure constraint on a single Waterbomb unit cell.*

### Governing Equations
For a vertex $v$ with incident edges, the product of rotation matrices must equal the identity matrix $I_3$. This forms the residual function $f(x, p, u)$:

$$R_{z}(\alpha_1)R_{x}(\rho_1) \dots R_{z}(\alpha_k)R_{x}(\rho_k) - I_3 = 0$$

Where:
* $x$: State vector of **free fold angles** (unknowns).
* $u$: Input vector of **driven fold angles** (user-controlled actuators).
* $p$: Static parameters (sector angles $\alpha$, connectivity).

To distinguish between state variables and inputs, the rotation term is defined piecewise, mapping the problem into a robust root-finding operation.

## Numerical Optimization

### Damped Newton-Raphson Solver
Standard Newton iteration is unstable for rigid origami because the "flat sheet" configuration represents a singular point where the Jacobian becomes ill-conditioned. This leads to **bifurcation**, where a solver might jump between "mountain" and "valley" assignments.

To resolve this, the solver implements a **step-limited damping scheme (Trust Region approach)**. This constrains iterative updates to the local basin of attraction, preserving frame-to-frame coherence and preventing non-physical mesh inversions.

![Convergence Graph - Residual vs Iterations](https://placeholder-image-url/graph1.png)
*Figure 1: Convergence comparison. Explicit methods (Forward Euler) fail to converge on non-linear constraints. Undamped Newton methods achieve fast algebraic convergence but suffer from bifurcation. The Damped Newton method (Blue) provides the necessary stability for interactive simulation.*

### LSMR & Regularization
The linear step direction $J \Delta x = -F(x)$ is solved using `scipy.sparse.linalg.lsmr` (Least Squares Minimum Residual). LSMR was selected over LSQR for its superior convergence monotonicity, which is critical when handling rank-deficient matrices inherent in bifurcation points.

## System Architecture & Performance

### Sparse Jacobian Assembly ($O(N)$)
A naive dense Jacobian implementation requires $O(N^2)$ memory and time, which becomes prohibitive for grids larger than $12 \times 12$. 

This solver utilizes a **Sparse Coordinate (COO)** assembly routine. Since geometric constraints are topologically local (a fold only affects its immediate vertex ring), the Jacobian is over **98% sparse** for large grids.
* **Analytical Derivatives**: Gradients are computed via the chain rule on $SO(3)$ rather than finite differences, eliminating numerical noise.
* **Parallelization**: Assembly is parallelized across CPU cores using **Numba `prange`**, bypassing the Python GIL.

### Scalability Benchmarks
The system exhibits a performance crossover point between 158 and 242 Degrees of Freedom (DOFs). Beyond this threshold, the dense solver hits a cubic bottleneck ($O(N^3)$), while the sparse solver scales according to an empirical power law of approximately **$O(N^{1.2})$**.

![Scalability Graph - Time vs DOFs](https://placeholder-image-url/graph2.png)
*Figure 2: Wall-clock execution time vs. Degrees of Freedom. The sparse solver (Blue) enables sub-second convergence for grids >1000 DOFs, maintaining interactive framerates for complex tessellations like the 70x70 Miura-ori.*

### Memory Layout
* **Zero-Copy Operations**: The solver pre-allocates flat arrays for COO formats, allowing worker threads to write derivatives directly into contiguous memory blocks.
* **Cache Coherence**: Custom memory mapping strategies flatten graph-based constraints to optimize CPU cache usage during the linear solve.

## Visualization & Reconstruction

To visualize the 3D structure from 1D fold angles, the system implements a linear-time **Breadth-First Search (BFS) Tree**.

1.  **Pre-computation**: A folding tree is rooted at the center of the mesh.
2.  **Propagation**: Transforms are propagated using optimized **Rodrigues' rotation formulas**.
3.  **Efficiency**: Surface reconstruction is strictly $O(N)$, decoupled from the iterative solver loop.

## Gallery

<img width="3472" height="1971" alt="Screenshot 2025-12-05 at 5 22 51 PM" src="https://github.com/user-attachments/assets/f14c28f8-3763-4bc7-b655-e2ca8cf74423" />

*Complex structures generated via inverse kinematics.*

https://github.com/user-attachments/assets/1d970d4f-d281-4b69-8703-14f0687f53da

*Interactive folding demonstration.*

## Limitations & Robustness
The solver includes failure analysis detection. For non-rigid models (e.g., the "Flapping Bird" which requires panel deformation), the solver correctly stagnates at a non-zero residual, identifying that no geometric state exists where all rigid constraints are satisfied.

---
*This project demonstrates advanced computational geometry techniques, bridging the gap between theoretical kinematics and high-performance software engineering.*