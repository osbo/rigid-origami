import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsmr
from scipy.sparse import coo_matrix
import matplotlib

from rigid_origami import (
    generate_miura_ori_grid,
    generate_waterbomb_grid,
    compute_residual,
    compute_jacobian_analytical,
    NewtonSolver
)

# ===================================================================
# 1. Alternative Solvers (Undamped, Dense, and Forward Euler)
# ===================================================================

def NewtonSolverUndamped(eval_f, eval_Jf, x_start, p, errf=1e-10, errDeltax=1e-8, MaxIter=100, held_edges=None, verbose=False):
    """
    Undamped Newton-Raphson solver (no step scaling).
    Pure Newton-Raphson without damping - can oscillate/diverge for highly non-linear constraints.
    """
    k = 0
    N = len(x_start)
    x_curr = x_start.copy()
    
    if held_edges is None:
        held_edges = np.zeros(N, dtype=bool)
    
    # Initial error
    f = eval_f(x_curr, p)
    
    # Handle case where there are no constraints (empty residual)
    if len(f) == 0:
        return x_curr
    
    errf_k = np.linalg.norm(f, np.inf)
    
    while k < MaxIter and errf_k > errf:
        # 1. Jacobian
        Jf = eval_Jf(x_curr, p)
        
        # 2. Linear Solve (sparse)
        free_indices = np.where(~held_edges)[0]
        J_free = Jf[:, free_indices]
        f_vec = -f
        
        result = lsmr(J_free, f_vec, atol=1e-8, btol=1e-8)
        dx_free = result[0]

        # Reconstruct full dx
        dx = np.zeros(N)
        dx[free_indices] = dx_free
        
        # 3. Update (NO DAMPING - pure Newton step)
        x_curr += dx
        
        # 4. Check errors
        f = eval_f(x_curr, p)
        
        if len(f) == 0:
            break
        
        errf_k = np.linalg.norm(f, np.inf)
        errDx = np.linalg.norm(dx, np.inf)
        
        k += 1
        
        if errDx < errDeltax:
            break

    return x_curr

def NewtonSolverDense(eval_f, eval_Jf, x_start, p, errf=1e-10, errDeltax=1e-8, MaxIter=100, held_edges=None, verbose=False):
    """
    Dense Newton-Raphson solver (converts sparse Jacobian to dense and uses dense solve).
    This is the "naive" approach that doesn't exploit sparsity.
    """
    k = 0
    N = len(x_start)
    x_curr = x_start.copy()
    
    if held_edges is None:
        held_edges = np.zeros(N, dtype=bool)
    
    # Initial error
    f = eval_f(x_curr, p)
    
    # Handle case where there are no constraints (empty residual)
    if len(f) == 0:
        return x_curr
    
    errf_k = np.linalg.norm(f, np.inf)
    
    while k < MaxIter and errf_k > errf:
        # 1. Jacobian (sparse)
        Jf_sparse = eval_Jf(x_curr, p)
        
        # 2. Convert to dense (THE BOTTLENECK)
        Jf = Jf_sparse.toarray()
        
        # 3. Linear Solve (dense)
        free_indices = np.where(~held_edges)[0]
        J_free = Jf[:, free_indices]
        f_vec = -f
        
        # Use dense least squares
        dx_free = np.linalg.lstsq(J_free, f_vec, rcond=None)[0]

        # Reconstruct full dx
        dx = np.zeros(N)
        dx[free_indices] = dx_free
        
        # 4. Update (with damping like the original)
        step_scale = 1.0
        max_step = 0.5  # radians
        if np.max(np.abs(dx)) > max_step:
            step_scale = max_step / np.max(np.abs(dx))
            
        x_curr += dx * step_scale
        
        # 5. Check errors
        f = eval_f(x_curr, p)
        
        if len(f) == 0:
            break
        
        errf_k = np.linalg.norm(f, np.inf)
        errDx = np.linalg.norm(dx * step_scale, np.inf)
        
        k += 1
        
        if errDx < errDeltax:
            break

    return x_curr

def ForwardEulerSolver(eval_f, eval_Jf, x_start, p, errf=1e-10, errDeltax=1e-8, MaxIter=100, held_edges=None, dt=0.01, verbose=False):
    """
    Forward Euler solver (time-stepping method).
    Uses simple time integration: x_new = x_old + dt * dx
    where dx is computed from the residual using the Jacobian.
    This is a first-order explicit method, typically slower to converge than Newton-Raphson.
    """
    k = 0
    N = len(x_start)
    x_curr = x_start.copy()
    
    if held_edges is None:
        held_edges = np.zeros(N, dtype=bool)
    
    # Initial error
    f = eval_f(x_curr, p)
    
    # Handle case where there are no constraints (empty residual)
    if len(f) == 0:
        return x_curr
    
    errf_k = np.linalg.norm(f, np.inf)
    
    while k < MaxIter and errf_k > errf:
        # 1. Compute residual
        f = eval_f(x_curr, p)
        
        if len(f) == 0:
            break
        
        # 2. Compute Jacobian
        Jf = eval_Jf(x_curr, p)
        
        # 3. Forward Euler step: solve J * dx = -f for direction
        # Use least squares to get update direction
        free_indices = np.where(~held_edges)[0]
        J_free = Jf[:, free_indices]
        f_vec = -f
        
        # Solve for direction (but don't take full step - use time step dt)
        result = lsmr(J_free, f_vec, atol=1e-8, btol=1e-8)
        dx_free = result[0]
        
        # Reconstruct full dx
        dx = np.zeros(N)
        dx[free_indices] = dx_free
        
        # 4. Forward Euler update: x_new = x_old + dt * dx
        # Apply time step scaling
        x_curr += dt * dx
        
        # 5. Check errors
        f = eval_f(x_curr, p)
        
        if len(f) == 0:
            break
        
        errf_k = np.linalg.norm(f, np.inf)
        errDx = np.linalg.norm(dt * dx, np.inf)
        
        k += 1
        
        if errDx < errDeltax:
            break
    
    return x_curr

def NewtonSolverHomotopy(eval_f, eval_Jf, x_start, p, errf=1e-10, errDeltax=1e-8, MaxIter=100, held_edges=None, num_homotopy_steps=10, verbose=False):
    """
    Newton's method with homotopy continuation.
    
    Uses homotopy continuation to gradually transition from an easy problem to the full problem.
    The homotopy parameter λ goes from 0 (easy/linearized problem) to 1 (full problem).
    
    At each homotopy step:
    1. Solve the intermediate problem: H(x, λ) = (1-λ)*f_linear(x) + λ*f_full(x) = 0
    2. Use Newton's method to solve H(x, λ) = 0
    3. Use solution as initial guess for next λ
    
    Args:
        num_homotopy_steps: Number of homotopy continuation steps
    """
    N = len(x_start)
    x_curr = x_start.copy()
    
    if held_edges is None:
        held_edges = np.zeros(N, dtype=bool)
    
    # Homotopy parameter: λ ∈ [0, 1]
    # λ = 0: easy problem (linearized/initial guess)
    # λ = 1: full problem (actual constraints)
    lambda_values = np.linspace(0.0, 1.0, num_homotopy_steps + 1)
    
    # For homotopy, we use: H(x, λ) = λ * f(x) + (1-λ) * f_linear(x)
    # For simplicity, we can use: H(x, λ) = λ * f(x) where f_linear(x) = 0 at x_start
    # This means we gradually scale up the constraint violations
    
    for lambda_idx, lam in enumerate(lambda_values[1:], 1):  # Skip λ=0, start from first non-zero
        # Solve H(x, λ) = λ * f(x) = 0 using Newton's method
        # This is equivalent to solving f(x) = 0 but with scaled residuals
        
        k = 0
        x_lambda = x_curr.copy()  # Start from solution of previous λ
        
        # Initial error for this homotopy step
        f = eval_f(x_lambda, p)
        if len(f) == 0:
            continue
        
        # Homotopy residual: H = λ * f
        H = lam * f
        errH_k = np.linalg.norm(H, np.inf)
        
        # Newton iterations for this λ
        while k < MaxIter // num_homotopy_steps and errH_k > errf:
            # Jacobian of homotopy: J_H = λ * J_f
            Jf = eval_Jf(x_lambda, p)
            
            free_indices = np.where(~held_edges)[0]
            J_free = Jf[:, free_indices]
            H_vec = -H  # Solve J_H * dx = -H
            
            # Scale Jacobian by homotopy parameter
            J_H_free = lam * J_free
            
            # Solve for update
            result = lsmr(J_H_free, H_vec, atol=1e-8, btol=1e-8)
            dx_free = result[0]
            
            dx = np.zeros(N)
            dx[free_indices] = dx_free
            
            # Update with damping
            step_scale = 1.0
            max_step = 0.5
            if np.max(np.abs(dx)) > max_step:
                step_scale = max_step / np.max(np.abs(dx))
            
            x_lambda += dx * step_scale
            
            # Check convergence
            f = eval_f(x_lambda, p)
            if len(f) == 0:
                break
            
            H = lam * f
            errH_k = np.linalg.norm(H, np.inf)
            errDx = np.linalg.norm(dx * step_scale, np.inf)
            
            k += 1
            
            if errDx < errDeltax:
                break
        
        # Update current solution for next homotopy step
        x_curr = x_lambda.copy()
    
    return x_curr

# ===================================================================
# 2. Graph A: Convergence Stability Comparison
# ===================================================================

def plot_convergence_comparison(grid_size=5, max_iter=200):
    """
    Graph A: Stability Plot (Convergence)
    X-Axis: Iterations
    Y-Axis: ||f(x)||_inf (Residual) in Log Scale
    
    Generates two plots:
    1. Short-term view (up to 30 iterations)
    2. Long-term view (up to 200 iterations)
    
    Compares:
    - Damped Newton (from rigid_origami.py - main solver)
    - Undamped Newton (pure Newton-Raphson for comparison)
    - Forward Euler (time-stepping method)
    - Newton with Homotopy (continuation method)
    """
    print("Generating Graph A: Convergence Stability Comparison...")
    
    # Setup a specific case - use waterbomb grid
    data = generate_waterbomb_grid(grid_size, grid_size)
    proc = data['proc']
    proc.warmup_numba_jit()
    
    # Setup initial guess with constraint edges applied to create challenging problem
    # Apply a moderate initial angle to constraint edges
    initial_angle_deg = 30.0
    x0 = data['x_start_base'].copy()
    held_edges = data['held_mask_base'].copy()
    
    # Apply constraint edges with initial angle
    for edge_idx, scale_factor in data['constraint_edges']:
        angle = np.deg2rad(scale_factor * initial_angle_deg)
        x0[edge_idx] = angle
        held_edges[edge_idx] = True
    
    # Add perturbation to non-held edges to make it more challenging
    np.random.seed(42)  # For reproducibility
    free_mask = ~held_edges
    x0[free_mask] += np.random.normal(0, 0.2, np.sum(free_mask))
    
    residuals_damped = []
    residuals_undamped = []
    residuals_forward_euler = []
    residuals_homotopy = []
    
    # Custom stepper to capture history for damped version (main solver)
    def run_solve_damped():
        history = []
        x = x0.copy()
        held = held_edges.copy()
        
        for k in range(max_iter):
            f = compute_residual(x, proc)
            if len(f) == 0:
                break
            history.append(np.linalg.norm(f, np.inf))
            # if history[-1] < 1e-6:
            #     break
            
            J = compute_jacobian_analytical(x, proc)
            
            # Solve with held edges
            free_indices = np.where(~held)[0]
            J_free = J[:, free_indices]
            f_vec = -f
            
            result = lsmr(J_free, f_vec, atol=1e-8, btol=1e-8)
            dx_free = result[0]
            
            dx = np.zeros(len(x))
            dx[free_indices] = dx_free
            
            # Damping logic (matching original NewtonSolver)
            step_scale = 1.0
            max_step = 0.5
            if np.max(np.abs(dx)) > max_step:
                step_scale = max_step / np.max(np.abs(dx))
            
            x += dx * step_scale
        
        return history
    
    # Custom stepper to capture history for undamped version (for comparison)
    def run_solve_undamped():
        history = []
        x = x0.copy()
        held = held_edges.copy()
        
        for k in range(max_iter):
            f = compute_residual(x, proc)
            if len(f) == 0:
                break
            residual_norm = np.linalg.norm(f, np.inf)
            history.append(residual_norm)
            # if residual_norm < 1e-6:
            #     break
            
            J = compute_jacobian_analytical(x, proc)
            
            # Solve with held edges
            free_indices = np.where(~held)[0]
            J_free = J[:, free_indices]
            f_vec = -f
            
            result = lsmr(J_free, f_vec, atol=1e-8, btol=1e-8)
            dx_free = result[0]
            
            dx = np.zeros(len(x))
            dx[free_indices] = dx_free
            
            # NO DAMPING - pure Newton step (for comparison)
            x += dx
            
            # Check for NaN/Inf in solution
            if not np.all(np.isfinite(x)):
                history.append(1e10)  # Mark as diverged
                break
        
        return history
    
    # Custom stepper to capture history for forward Euler version
    def run_solve_forward_euler():
        history = []
        x = x0.copy()
        held = held_edges.copy()
        dt = 0.01  # Time step for forward Euler
        
        for k in range(max_iter):
            f = compute_residual(x, proc)
            if len(f) == 0:
                break
            residual_norm = np.linalg.norm(f, np.inf)
            history.append(residual_norm)
            if residual_norm < 1e-6:
                break
            
            J = compute_jacobian_analytical(x, proc)
            
            # Solve with held edges
            free_indices = np.where(~held)[0]
            J_free = J[:, free_indices]
            f_vec = -f
            
            result = lsmr(J_free, f_vec, atol=1e-8, btol=1e-8)
            dx_free = result[0]
            
            dx = np.zeros(len(x))
            dx[free_indices] = dx_free
            
            # Forward Euler step: x_new = x_old + dt * dx
            x += dt * dx
            
            # Check for NaN/Inf in solution
            if not np.all(np.isfinite(x)):
                history.append(1e10)  # Mark as diverged
                break
        
        return history
    
    # Custom stepper to capture history for homotopy version
    def run_solve_homotopy():
        history = []
        x = x0.copy()
        held = held_edges.copy()
        num_homotopy_steps = 10
        
        # Homotopy parameter: λ ∈ [0, 1]
        lambda_values = np.linspace(0.0, 1.0, num_homotopy_steps + 1)
        
        for lambda_idx, lam in enumerate(lambda_values[1:], 1):  # Skip λ=0
            # Newton iterations for this homotopy step
            iter_per_step = max_iter // num_homotopy_steps
            
            for k in range(iter_per_step):
                f = compute_residual(x, proc)
                if len(f) == 0:
                    break
                
                # Homotopy residual: H = λ * f
                H = lam * f
                residual_norm = np.linalg.norm(H, np.inf)
                history.append(residual_norm)
                
                if residual_norm < 1e-6:
                    break
                
                J = compute_jacobian_analytical(x, proc)
                
                # Solve with held edges
                free_indices = np.where(~held)[0]
                J_free = J[:, free_indices]
                H_vec = -H
                
                # Scale Jacobian by homotopy parameter
                J_H_free = lam * J_free
                
                result = lsmr(J_H_free, H_vec, atol=1e-8, btol=1e-8)
                dx_free = result[0]
                
                dx = np.zeros(len(x))
                dx[free_indices] = dx_free
                
                # Update with damping
                step_scale = 1.0
                max_step = 0.5
                if np.max(np.abs(dx)) > max_step:
                    step_scale = max_step / np.max(np.abs(dx))
                
                x += dx * step_scale
                
                # Check for NaN/Inf in solution
                if not np.all(np.isfinite(x)):
                    history.append(1e10)  # Mark as diverged
                    break
                
                if residual_norm < 1e-6:
                    break
        
        return history
    
    residuals_damped = run_solve_damped()
    residuals_undamped = run_solve_undamped()
    residuals_forward_euler = run_solve_forward_euler()
    residuals_homotopy = run_solve_homotopy()
    
    # Count driven edges for title
    num_driven_edges = len(data['constraint_edges'])
    
    # Prepare data for plotting (first 20 iterations only)
    data_short = {
        'damped': residuals_damped[:20] if len(residuals_damped) > 20 else residuals_damped,
        'undamped': residuals_undamped[:20] if len(residuals_undamped) > 20 else residuals_undamped,
        'forward_euler': residuals_forward_euler[:20] if len(residuals_forward_euler) > 20 else residuals_forward_euler,
        'homotopy': residuals_homotopy[:20] if len(residuals_homotopy) > 20 else residuals_homotopy
    }
    
    # Plot convergence (first 20 iterations)
    plt.figure(figsize=(10, 6))
    plt.semilogy(data_short['damped'], 'b-o', label='Damped Newton', linewidth=2, markersize=6)
    plt.semilogy(data_short['undamped'], 'r--x', label='Undamped Newton', linewidth=2, markersize=6)
    plt.semilogy(data_short['forward_euler'], 'g-.s', label='Forward Euler', linewidth=2, markersize=5)
    plt.semilogy(data_short['homotopy'], 'm:^', label='Newton with Homotopy', linewidth=2, markersize=5)
    
    # Add convergence threshold line
    convergence_threshold = 1e-8
    plt.axhline(y=convergence_threshold, color='gray', linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'Convergence Threshold ({convergence_threshold:.0e})')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Residual, Log Scale', fontsize=12)
    title_line1 = f'Residual Convergence Comparison: Damped vs Undamped Newton-Raphson vs Forward Euler vs Homotopy'
    title_line2 = f'Waterbomb Grid {grid_size}×{grid_size}: Diagonal Edges Driven at {initial_angle_deg}°'
    plt.title(f'{title_line1}\n{title_line2}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 19)
    plt.tight_layout()
    plt.show()
    
    print(f"  Damped: {len(residuals_damped)} iterations, final residual: {residuals_damped[-1]:.2e}")
    print(f"  Undamped: {len(residuals_undamped)} iterations, final residual: {residuals_undamped[-1]:.2e}")
    print(f"  Forward Euler: {len(residuals_forward_euler)} iterations, final residual: {residuals_forward_euler[-1]:.2e}")
    print(f"  Homotopy: {len(residuals_homotopy)} iterations, final residual: {residuals_homotopy[-1]:.2e}")
    
    return residuals_damped, residuals_undamped, residuals_forward_euler, residuals_homotopy

# ===================================================================
# 3. Graph B: HPC Scalability Comparison
# ===================================================================

def test_solver_scalability(max_grid_size=20):
    """
    Graph B: HPC Plot (Scalability)
    X-Axis: System Size (Number of Edges/DOFs)
    Y-Axis: Time to Solution (seconds)
    
    Compares:
    - Sparse Solver (scipy.sparse.linalg.lsmr + coo_matrix)
    - Dense Solver (Convert to dense and use np.linalg.lstsq)
    """
    print("Generating Graph B: Solver Scalability Comparison...")
    
    sizes = []
    # Generate sizes: doubled number of points, evenly spaced
    # Original pattern was: 2, 4, 8, 12, 16, 20
    # Now: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    s = 2
    while s <= max_grid_size:
        sizes.append(s)
        s += 1  # Step by 1 for evenly spaced points (doubled density)
    
    times_sparse = []
    times_dense = []
    dofs = []
    
    print(f"\n{'Grid':<10} | {'DOFs':<10} | {'Sparse (s)':<12} | {'Dense (s)':<12}")
    print("-" * 60)
    
    for n in sizes:
        try:
            # 1. Setup Data - use waterbomb grid
            data = generate_waterbomb_grid(n, n)
            proc = data['proc']
            proc.warmup_numba_jit()
            
            # Setup initial guess (slightly perturbed)
            x0 = data['x_start_base'] + np.random.normal(0, 0.01, len(data['x_start_base']))
            held_edges = data['held_mask_base'].copy()
            
            # 2. Time Sparse (Your current method)
            t0 = time.time()
            x_solution = NewtonSolver(
                compute_residual,
                compute_jacobian_analytical,
                x0.copy(),
                proc,
                held_edges=held_edges,
                errf=1e-6,
                errDeltax=1e-6,
                MaxIter=100,  # 2x iterations to allow longer runs
                verbose=False
            )
            t_sparse = time.time() - t0
            
            # 3. Time Dense (Naive comparison)
            t_dense = None
            if n <= 24:  # Dense is too slow for big grids (2x the previous limit)
                try:
                    t0 = time.time()
                    x_solution_dense = NewtonSolverDense(
                        compute_residual,
                        compute_jacobian_analytical,
                        x0.copy(),
                        proc,
                        held_edges=held_edges,
                        errf=1e-6,
                        errDeltax=1e-6,
                        MaxIter=50,
                        verbose=False
                    )
                    t_dense = time.time() - t0
                except Exception as e:
                    print(f"    Dense solver failed for {n}x{n}: {e}")
                    t_dense = None
            else:
                t_dense = None
            
            times_sparse.append(t_sparse)
            times_dense.append(t_dense)
            dofs.append(len(x0))
            
            dense_str = f"{t_dense:.4f}" if t_dense is not None else "N/A (too slow)"
            print(f"{n}x{n:<8} | {len(x0):<10} | {t_sparse:<12.4f} | {dense_str:<12}")
            
        except Exception as e:
            print(f"Error at grid size {n}x{n}: {e}")
            break
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Filter out None values for dense plot
    dofs_plot = dofs
    sparse_plot = times_sparse
    dense_plot = [t for t in times_dense if t is not None]
    dofs_dense_plot = [dofs[i] for i, t in enumerate(times_dense) if t is not None]
    
    plt.plot(dofs_plot, sparse_plot, 'b-o', label='Sparse Solver', linewidth=2, markersize=8)
    if len(dense_plot) > 0:
        plt.plot(dofs_dense_plot, dense_plot, 'r--x', label='Dense Solver', linewidth=2, markersize=8)
    
    plt.xlabel('Degrees of Freedom (Number of Edges)', fontsize=12)
    plt.ylabel('Time to Convergence (seconds)', fontsize=12)
    plt.ylim(0, 3)
    title_line1 = 'Solver Scalability: Sparse vs Dense'
    title_line2 = 'n×n Waterbomb Grid: Diagonal Edges Driven at 30°'
    plt.title(f'{title_line1}\n{title_line2}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return dofs, times_sparse, times_dense

# ===================================================================
# 4. Graph C: Sparsity Pattern Visualization
# ===================================================================

def plot_sparsity_pattern(grid_size=5):
    """
    Graph C: Sparsity Pattern (Visual)
    A "Spy Plot" of the Jacobian Matrix (J).
    Shows the diagonal banding representing localized constraints.
    """
    print("Generating Graph C: Sparsity Pattern Visualization...")
    
    # Setup a specific case - use waterbomb grid
    data = generate_waterbomb_grid(grid_size, grid_size)
    proc = data['proc']
    proc.warmup_numba_jit()
    
    # Setup initial guess
    x0 = data['x_start_base'] + np.random.normal(0, 0.01, len(data['x_start_base']))
    
    # Compute Jacobian
    J = compute_jacobian_analytical(x0, proc)
    
    # Plot sparsity pattern
    plt.figure(figsize=(10, 10))
    plt.spy(J, markersize=0.5, precision=1e-10)
    plt.xlabel('Column Index (Edge/DOF)', fontsize=12)
    plt.ylabel('Row Index (Constraint)', fontsize=12)
    plt.title(f'Jacobian Sparsity Pattern: {grid_size}x{grid_size} Waterbomb Grid', 
              fontsize=14, fontweight='bold')
    
    # Add text annotation with statistics
    m, n = J.shape
    nnz = J.nnz
    sparsity = (1 - nnz / (m * n)) * 100
    stats_text = f'Size: {m}×{n}\nNon-zeros: {nnz}\nSparsity: {sparsity:.2f}%'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print(f"  Jacobian size: {J.shape}")
    print(f"  Non-zero elements: {J.nnz}")
    print(f"  Sparsity: {(1 - J.nnz / (J.shape[0] * J.shape[1])) * 100:.2f}%")
    
    return J

# ===================================================================
# 4. Convergence Time Comparison Test
# ===================================================================

def test_convergence_times():
    """
    Test convergence times for various origami systems.
    All systems start flat and fold to 30 degrees.
    Prints a comparison table.
    """
    print("=" * 70)
    print("Convergence Time Comparison Test")
    print("=" * 70)
    print("Testing systems: all start flat, fold to 30°")
    print()
    
    from rigid_origami import input_fold, generate_miura_ori_grid, generate_waterbomb_grid
    
    # Define test cases
    test_cases = [
        ("1×1 Miura-Ori", lambda: generate_miura_ori_grid(1, 1, 70)),
        ("6×6 Miura-Ori", lambda: generate_miura_ori_grid(6, 6, 70)),
        ("6×6 Waterbomb", lambda: generate_waterbomb_grid(6, 6)),
        ("Flapping Bird", lambda: input_fold("flappingBird")),
    ]
    
    results = []
    
    for name, generator in test_cases:
        print(f"Testing {name}...", end=" ", flush=True)
        
        try:
            # Generate the system
            data = generator()
            proc = data['proc']
            proc.warmup_numba_jit()
            
            # Setup: start flat, fold to 30 degrees
            x0 = data['x_start_base'].copy()
            held_edges = data['held_mask_base'].copy()
            
            # Apply 30 degree constraint to driven edges
            target_angle_deg = 30.0
            for edge_idx, scale_factor in data['constraint_edges']:
                angle = np.deg2rad(scale_factor * target_angle_deg)
                x0[edge_idx] = angle
                held_edges[edge_idx] = True
            
            # Time the solver and track iterations
            max_iter = 100
            t0 = time.time()
            
            # Wrapper to track iterations
            k = 0
            N = len(x0)
            x_curr = x0.copy()
            f = compute_residual(x_curr, proc)
            
            if len(f) == 0:
                num_iterations = 0
                x_solution = x_curr
            else:
                errf_k = np.linalg.norm(f, np.inf)
                free_indices = np.where(~held_edges)[0]
                
                while k < max_iter and errf_k > 1e-6:
                    Jf = compute_jacobian_analytical(x_curr, proc)
                    J_free = Jf[:, free_indices]
                    f_vec = -f
                    
                    result = lsmr(J_free, f_vec, atol=1e-8, btol=1e-8)
                    dx_free = result[0]
                    
                    dx = np.zeros(N)
                    dx[free_indices] = dx_free
                    
                    step_scale = 1.0
                    max_step = 0.5
                    if np.max(np.abs(dx)) > max_step:
                        step_scale = max_step / np.max(np.abs(dx))
                    
                    x_curr += dx * step_scale
                    f = compute_residual(x_curr, proc)
                    
                    if len(f) == 0:
                        break
                    
                    errf_k = np.linalg.norm(f, np.inf)
                    errDx = np.linalg.norm(dx * step_scale, np.inf)
                    
                    k += 1
                    
                    if errDx < 1e-6:
                        break
                
                num_iterations = k
                x_solution = x_curr
            
            t_elapsed = time.time() - t0
            
            # Check final residual
            f_final = compute_residual(x_solution, proc)
            if len(f_final) > 0:
                final_residual = np.linalg.norm(f_final, np.inf)
            else:
                final_residual = 0.0
            
            # Count DOFs
            num_dofs = len(x0)
            
            results.append({
                'name': name,
                'time': t_elapsed,
                'dofs': num_dofs,
                'residual': final_residual,
                'iterations': num_iterations,
                'max_iter': max_iter,
                'status': 'Success' if final_residual < 1e-6 else 'Failed',
                'error': None
            })
            
            status_str = "✓" if final_residual < 1e-6 else "✗"
            iter_ratio = num_iterations / max_iter if max_iter > 0 else 0.0
            print(f"{status_str} {t_elapsed:.4f}s (DOFs: {num_dofs}, iterations: {num_iterations}/{max_iter} ({iter_ratio:.3f}), residual: {final_residual:.2e})")
            
        except Exception as e:
            results.append({
                'name': name,
                'time': None,
                'dofs': None,
                'residual': None,
                'iterations': None,
                'max_iter': max_iter,
                'status': 'Failed',
                'error': str(e)
            })
            print(f"✗ Failed: {e}")
    
    # Print comparison table
    print("\n" + "=" * 110)
    print("CONVERGENCE TIME COMPARISON TABLE")
    print("=" * 110)
    print(f"{'System':<20} | {'DOFs':<10} | {'Time (s)':<12} | {'Iterations/MaxIter':<20} | {'Final Residual':<15} | {'Status':<10}")
    print("-" * 110)
    
    for r in results:
        name = r['name']
        dofs_str = f"{r['dofs']}" if r['dofs'] is not None else "N/A"
        time_str = f"{r['time']:.4f}" if r['time'] is not None else "N/A"
        if r['iterations'] is not None and r['max_iter'] is not None:
            iter_ratio = r['iterations'] / r['max_iter']
            iter_str = f"{r['iterations']}/{r['max_iter']} ({iter_ratio:.3f})"
        else:
            iter_str = "N/A"
        residual_str = f"{r['residual']:.2e}" if r['residual'] is not None else "N/A"
        status = r['status']
        
        print(f"{name:<20} | {dofs_str:<10} | {time_str:<12} | {iter_str:<20} | {residual_str:<15} | {status:<10}")
    
    print("=" * 110)
    print()
    
    return results

# ===================================================================
# 5. Main Execution
# ===================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Rigid Origami: Performance Analysis Graphs")
    print("=" * 70)
    print()
    
    # # Graph A: Convergence Stability
    # print("\n" + "=" * 70)
    # print("GRAPH A: Convergence Stability Comparison")
    # print("=" * 70)
    # plot_convergence_comparison(grid_size=6, max_iter=200)
    
    # # Graph B: Scalability
    # print("\n" + "=" * 70)
    # print("GRAPH B: Solver Scalability Comparison")
    # print("=" * 70)
    # test_solver_scalability(max_grid_size=20)
    
    # # Graph C: Sparsity Pattern
    # print("\n" + "=" * 70)
    # print("GRAPH C: Jacobian Sparsity Pattern")
    # print("=" * 70)
    # plot_sparsity_pattern(grid_size=6)
    
    # Convergence Time Comparison Test
    print("\n" + "=" * 70)
    print("CONVERGENCE TIME COMPARISON TEST")
    print("=" * 70)
    test_convergence_times()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
