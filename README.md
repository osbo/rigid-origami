# High-Performance Rigid Origami Solver

**Computational Geometry | Python, Numba, Scipy**

A high-performance, rigid-body solver for complex origami patterns. The program solves for valid 3D configurations of crease patterns by enforcing geometric closure constraints around internal vertices. It leverages JIT compilation, analytical derivatives, and sparse linear algebra to simulate folding mechanics in real-time.

![Huffman Rectangular Weave Simulation](assets/hero_weave.jpg)

*Simulated Huffman Rectangular Weave pattern solving for geometric closure.*

## Mathematical Formulation

The simulation treats origami not as a mass-spring system, but as a system of rigid faces connected by rotational hinges. The core problem is formulated as finding the set of fold angles $\rho$ that satisfy the **Loop Closure Constraint** for every internal vertex in the mesh.

<div align="center">

  <img src="assets/demo_unit.gif" width="600" alt="Unit Cell Loop Closure">

  <br>

  <em>Visualization of the loop closure constraint on a single unit cell.</em>

</div>

For a vertex $v$ with incident edges $e_1, \dots, e_k$, the cumulative rotation of the sector angles $\alpha$ and fold angles $\rho$ must equal the identity matrix:

$$R_{z}(\alpha_1)R_{x}(\rho_1) \dots R_{z}(\alpha_k)R_{x}(\rho_k) = I$$

The solver minimizes the residual error of these rotation chains, ensuring the paper does not tear or stretch. By mapping the rotational error into skew-symmetric space, we reduce the non-linear constraints into a robust Newton-Raphson optimization problem.

## Solver Architecture

### Analytical Jacobian & Derivatives

Unlike standard solvers that rely on slow and approximation-prone finite differences, the program calculates the **Analytical Jacobian** of the constraint system. 

Using the chain rule on the rotation group $SO(3)$, the solver computes the exact derivative of the rotation product with respect to every edge angle. This allows for:

1.  **High Precision**: Eliminates numerical noise associated with finite differences.

2.  **Stability**: Provides accurate gradients even near singular configurations (e.g., flat states).

3.  **Speed**: Derivatives are computed in a single pass alongside residuals.

### Damped Newton-Raphson Optimization

The core loop utilizes a Damped Newton-Raphson method to traverse the solution manifold:

1.  **Jacobian Assembly**: Parallel computation of the sparse Jacobian matrix $J$.

2.  **Linear Solve**: Solves the step direction $J \Delta x = -F(x)$ using `scipy.sparse.linalg.lsmr` (Least Squares Minimum Residual). This handles overdetermined systems and rank-deficient matrices inherent in bifurcation points.

3.  **Damping**: Applies adaptive step scaling to prevent face-intersection and numerical flipping.

## Computational Performance

### Real-Time Solving on Large Grids

The system is capable of solving dense grids in real-time. Below is a stress test on a **70x70 Miura-ori grid (approx. 5,000 faces)**, maintaining interactive framerates during manipulation.

![70x70 Miura Grid Performance Test](assets/demo_miura_70x70.gif)

### JIT Compilation & Parallelization

The heavy lifting is offloaded to LLVM via **Numba**. Critical geometric kernels are Just-In-Time compiled to optimized machine code, bypassing the Python Global Interpreter Lock (GIL).

- **Parallel Jacobian Construction**: The calculation of the sparse Jacobian is parallelized across CPU cores using `prange`.

- **Zero-Copy Memory Layout**: The solver pre-allocates flat arrays for sparse Coordinate (COO) formats, allowing worker threads to write derivatives directly into memory without locking or race conditions.

### Sparse Matrix Operations

The constraint matrix for origami is highly sparse (each constraint only involves a small ring of local edges). The system utilizes:

- **COO/CSR Formats**: Efficient storage and slicing of the Jacobian.

- **Optimized Memory**: Custom memory mapping strategies to flatten graph-based constraints into contiguous memory blocks for cache coherence.

## Mesh Reconstruction & Visualization

### The Folding Tree

To visualize the 3D structure from 1D fold angles, the system implements a fast **Breadth-First Search (BFS) Tree**. 

1.  **Pre-computation**: A folding tree is built once during initialization, establishing a parent-child relationship for all faces rooted at the center of the mesh.

2.  **Fast Propagation**: During the render loop, transforms are propagated down the tree using optimized Rodrigues' rotation formulas.

3.  **Linear Complexity**: Surface reconstruction is strictly $O(N)$, enabling high-framerate visualization even for meshes with thousands of faces.

### Gallery

| Cylindrical Structure | Medium Scale Simulation |
| :---: | :---: |
| <img src="assets/tube_structure.jpg" width="100%"> | <img src="assets/demo_medium.gif" width="100%"> |
| *Complex curvature handling* | *Interactive folding stability* |

## Technical Highlights

- **Rodrigues' Rotation Formula**: Optimized inline implementation to avoid heavy library calls for rotation matrices.

- **Bifurcation Handling**: Robustness against singular values via LSMR regularization.

- **Interactive Driving**: Supports "Driven Edges" where specific folds are enforced, allowing the solver to find the resultant configuration of the remaining passive folds.

---

*This project demonstrates advanced computational geometry techniques, bridging the gap between theoretical kinematics and high-performance software engineering.*
