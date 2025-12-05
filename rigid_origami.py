import numpy as np
from scipy.sparse import lil_matrix, coo_matrix
from scipy.sparse.linalg import lsmr
import pyvista as pv
from collections import deque
from numba import njit, prange
import json

# ===================================================================
# 1. Geometry & Kinematics Engine
# ===================================================================

def rot_x(angle):
    """Fast rotation matrix about X-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c]
    ])

def rot_z(angle):
    """Fast rotation matrix about Z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0]
    ])

def rotation_matrix(axis, angle):
    """
    Rotation matrix about arbitrary axis using Rodrigues' rotation formula.
    Optimized version without scipy dependency.
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3)
    axis = axis / norm
    
    # Rodrigues' rotation formula: R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
    # where [k]× is the skew-symmetric matrix of the axis
    c, s = np.cos(angle), np.sin(angle)
    kx, ky, kz = axis[0], axis[1], axis[2]
    
    # Skew-symmetric matrix [k]×
    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])
    
    # Rodrigues' formula
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R

def get_transforms_for_vertex(x, edge_indices, sector_angles):
    """
    Computes the cumulative transform chain for a vertex loop.
    """
    num_creases = len(edge_indices)
    transforms = []
    for j in range(num_creases):
        edge_idx = edge_indices[j]
        fold_angle = x[edge_idx]
        sector_alpha = sector_angles[j]
        
        R_fold = rot_x(fold_angle)
        R_sector = rot_z(sector_alpha)
        
        transforms.append(R_sector @ R_fold) 
        
    return transforms

class MeshProcessor:
    def __init__(self, nodes, faces):
        self.initial_nodes = nodes
        self.faces = faces
        self.num_faces = len(faces)
        self.edge_map = {}
        self.edges = []
        self.face_adjacency = {i: [] for i in range(self.num_faces)}
        face_to_edges = {i: [] for i in range(self.num_faces)}

        edge_counter = 0
        for f_idx, face in enumerate(faces):
            n = len(face)
            for i in range(n):
                u, v = face[i], face[(i+1)%n]
                key = tuple(sorted((u, v)))
                if key not in self.edge_map:
                    self.edge_map[key] = edge_counter
                    self.edges.append(key)
                    edge_counter += 1
                face_to_edges[f_idx].append(self.edge_map[key])

        self.num_edges = len(self.edges)
        self.edge_to_faces = {i: [] for i in range(self.num_edges)}
        for f_idx, e_list in face_to_edges.items():
            for e_idx in e_list:
                self.edge_to_faces[e_idx].append(f_idx)
        
        # Build adjacency for surface reconstruction
        for e_idx, f_list in self.edge_to_faces.items():
            if len(f_list) == 2:
                self.face_adjacency[f_list[0]].append((f_list[1], e_idx))
                self.face_adjacency[f_list[1]].append((f_list[0], e_idx))

        # A vertex is on the boundary if it belongs to a boundary edge (edge with < 2 faces)
        boundary_vertices = set()
        for e_idx, f_list in self.edge_to_faces.items():
            if len(f_list) < 2:
                u, v = self.edges[e_idx]
                boundary_vertices.add(u)
                boundary_vertices.add(v)

        self.constraints = []
        v_to_neighbors = {}
        for e_idx, (u, v) in enumerate(self.edges):
            v_to_neighbors.setdefault(u, []).append((v, e_idx))
            v_to_neighbors.setdefault(v, []).append((u, e_idx))

        for v_idx in range(len(nodes)):
            # SKIP BOUNDARY VERTICES
            if v_idx in boundary_vertices:
                continue

            neighbors = v_to_neighbors.get(v_idx, [])
            # Only internal vertices (degree >= 3) impose closure constraints
            if len(neighbors) < 3: continue 

            center = nodes[v_idx]
            sorted_n = []
            for n_idx, e_idx in neighbors:
                vec = nodes[n_idx] - center
                ang = np.arctan2(vec[1], vec[0])
                sorted_n.append((ang, n_idx, e_idx))
            sorted_n.sort()

            sector_angles = []
            edge_chain = []
            for i in range(len(sorted_n)):
                _, idx_curr, e_curr = sorted_n[i]
                _, idx_next, _ = sorted_n[(i+1)%len(sorted_n)]
                v1 = (nodes[idx_curr] - center); v1 /= np.linalg.norm(v1)
                v2 = (nodes[idx_next] - center); v2 /= np.linalg.norm(v2)
                dp = np.clip(np.dot(v1, v2), -1, 1)
                ang = np.arccos(dp)
                sector_angles.append(ang)
                edge_chain.append(e_curr)

            # Add closure constraint for vertices with 4+ edges
            if len(edge_chain) >= 4:
                self.constraints.append({
                    'vertex_idx': v_idx,
                    'edge_indices': edge_chain,
                    'sector_angles': np.array(sector_angles)
                })

    def reconstruct_surface(self, x):
        """
        Surface reconstruction. Uses fast version if folding tree is precomputed.
        """
        if hasattr(self, 'folding_tree') and len(self.folding_tree) > 0:
            return self.reconstruct_surface_fast(x)
        
        # Fallback to original BFS-based method if tree not precomputed
        face_transforms = {0: (np.eye(3), np.zeros(3))}
        queue = deque([0])
        visited = {0}
        while queue:
            parent = queue.popleft()
            R_p, T_p = face_transforms[parent]
            for child, e_idx in self.face_adjacency[parent]:
                if child in visited: continue
                u, v = self.edges[e_idx]
                p_u = self.initial_nodes[u]
                pf_verts = self.faces[parent]
                try:
                    idx_u = pf_verts.index(u)
                    # Determine sign based on vertex winding order
                    sgn = 1.0 if pf_verts[(idx_u + 1) % len(pf_verts)] == v else -1.0
                except ValueError: sgn = 1.0

                edge_vec = self.initial_nodes[v] - p_u
                edge_dir = edge_vec / np.linalg.norm(edge_vec)
                R_hinge = rotation_matrix(edge_dir, sgn * x[e_idx])
                R_c = R_p @ R_hinge
                T_c = T_p + (R_p @ p_u) - (R_c @ p_u)
                face_transforms[child] = (R_c, T_c)
                visited.add(child)
                queue.append(child)

        new_nodes = np.zeros_like(self.initial_nodes)
        counts = np.zeros(len(self.initial_nodes))
        for f, (Rf, Tf) in face_transforms.items():
            for v in self.faces[f]:
                new_nodes[v] += Rf @ self.initial_nodes[v] + Tf
                counts[v] += 1
        
        # Avoid division by zero for isolated vertices (though unlikely in valid mesh)
        counts[counts == 0] = 1
        return new_nodes / counts[:, None]

    def prepare_constraint_arrays(self):
        """
        Flatten constraints into arrays for Numba JIT compilation.
        Returns:
            constraint_edge_indices: flat array of all edge indices
            constraint_sector_angles: flat array of all sector angles
            constraint_offsets: array where constraint_offsets[i] is start index for constraint i
            constraint_lengths: array where constraint_lengths[i] is num edges in constraint i
            write_offsets: array for sparse COO format - write_offsets[i] is where constraint i starts writing
        """
        if len(self.constraints) == 0:
            return (np.array([], dtype=np.int32), 
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    np.array([0], dtype=np.int32))
        
        # Flatten all constraints into contiguous arrays
        all_edge_indices = []
        all_sector_angles = []
        offsets = []
        lengths = []
        
        offset = 0
        for c in self.constraints:
            edge_indices = c['edge_indices']
            sector_angles = c['sector_angles']
            num_edges = len(edge_indices)
            
            all_edge_indices.extend(edge_indices)
            all_sector_angles.extend(sector_angles)
            offsets.append(offset)
            lengths.append(num_edges)
            
            offset += num_edges
        
        constraint_edge_indices = np.array(all_edge_indices, dtype=np.int32)
        constraint_sector_angles = np.array(all_sector_angles, dtype=np.float64)
        constraint_offsets = np.array(offsets, dtype=np.int32)
        constraint_lengths = np.array(lengths, dtype=np.int32)
        
        # Create write offsets for sparse COO format
        # Each edge in a constraint generates 3 non-zero entries (for x, y, z residuals)
        entries_per_constraint = constraint_lengths * 3
        write_offsets = np.zeros(len(constraint_lengths) + 1, dtype=np.int32)
        write_offsets[1:] = np.cumsum(entries_per_constraint)
        
        return (constraint_edge_indices,
                constraint_sector_angles,
                constraint_offsets,
                constraint_lengths,
                write_offsets)

    def warmup_numba_jit(self):
        """
        Warm up Numba JIT compilation by calling the fast functions
        with dummy data. This ensures the functions are compiled
        at startup rather than on first use.
        """
        if len(self.constraints) == 0:
            return
        
        # Prepare constraint arrays
        constraint_edge_indices, constraint_sector_angles, constraint_offsets, constraint_lengths, write_offsets = self.prepare_constraint_arrays()
        
        # Create dummy data with correct shapes
        n = self.num_edges
        x_dummy = np.zeros(n, dtype=np.float64)
        
        # Total number of non-zero elements for sparse format
        total_nnz = write_offsets[-1]
        J_data_dummy = np.zeros(total_nnz, dtype=np.float64)
        J_rows_dummy = np.zeros(total_nnz, dtype=np.int32)
        J_cols_dummy = np.zeros(total_nnz, dtype=np.int32)
        
        # Call functions to trigger compilation
        fast_jacobian_sparse(x_dummy, constraint_edge_indices, constraint_sector_angles,
                            constraint_offsets, constraint_lengths, write_offsets,
                            J_data_dummy, J_rows_dummy, J_cols_dummy)
        fast_residual_loop(x_dummy, constraint_edge_indices, constraint_sector_angles,
                          constraint_offsets, constraint_lengths)
        
        # Cache the constraint arrays for future use
        self._constraint_arrays = (constraint_edge_indices, constraint_sector_angles, 
                                   constraint_offsets, constraint_lengths, write_offsets)

    def precompute_folding_tree(self):
        """
        Computes the BFS traversal order once and stores it as flat arrays.
        This eliminates the need to recompute the face traversal order every frame.
        """
        # Initialize with root face
        queue = deque([0])
        visited = {0}
        
        # Store steps as (parent_face, child_face, edge_idx, sign)
        tree_steps = []
        
        while queue:
            parent = queue.popleft()
            for child, e_idx in self.face_adjacency[parent]:
                if child in visited:
                    continue
                
                # Pre-calculate sign logic here
                u, v = self.edges[e_idx]
                pf_verts = self.faces[parent]
                try:
                    idx_u = pf_verts.index(u)
                    sgn = 1.0 if pf_verts[(idx_u + 1) % len(pf_verts)] == v else -1.0
                except ValueError:
                    sgn = 1.0
                
                tree_steps.append((parent, child, e_idx, sgn))
                visited.add(child)
                queue.append(child)
        
        # Convert to numpy arrays for fast iteration
        # Format: [parent_idx, child_idx, edge_idx] per row
        if len(tree_steps) > 0:
            self.folding_tree = np.array([(s[0], s[1], s[2]) for s in tree_steps], dtype=np.int32)
            self.folding_tree_signs = np.array([s[3] for s in tree_steps], dtype=np.float64)
            
            # Pre-compute edge directions and reference points as arrays (not dicts)
            # Initialize arrays for all edges
            self.edge_dirs = np.zeros((self.num_edges, 3), dtype=np.float64)
            self.edge_ref_points = np.zeros((self.num_edges, 3), dtype=np.float64)
            
            # Compute for edges used in the tree
            computed_edges = set()
            for parent, child, e_idx, _ in tree_steps:
                if e_idx not in computed_edges:
                    u, v = self.edges[e_idx]
                    p_u = self.initial_nodes[u]
                    edge_vec = self.initial_nodes[v] - p_u
                    edge_dir = edge_vec / np.linalg.norm(edge_vec)
                    self.edge_dirs[e_idx] = edge_dir
                    self.edge_ref_points[e_idx] = p_u
                    computed_edges.add(e_idx)
            
            # Flatten faces to CSR format for Numba
            flat_faces = []
            face_offsets = [0]
            for face in self.faces:
                flat_faces.extend(face)
                face_offsets.append(len(flat_faces))
            
            self.face_indices_flat = np.array(flat_faces, dtype=np.int32)
            self.face_offsets = np.array(face_offsets, dtype=np.int32)
            self.num_faces = len(self.faces)
            self.num_nodes = len(self.initial_nodes)
        else:
            self.folding_tree = np.array([], dtype=np.int32).reshape(0, 3)
            self.folding_tree_signs = np.array([], dtype=np.float64)
            self.edge_dirs = np.zeros((self.num_edges, 3), dtype=np.float64)
            self.edge_ref_points = np.zeros((self.num_edges, 3), dtype=np.float64)
            # Empty face arrays
            self.face_indices_flat = np.array([], dtype=np.int32)
            self.face_offsets = np.array([0], dtype=np.int32)
            self.num_faces = len(self.faces)
            self.num_nodes = len(self.initial_nodes)

    def reconstruct_surface_fast(self, x):
        """
        Optimized surface reconstruction using precomputed folding tree.
        Uses Numba-compiled function for maximum speed.
        """
        return fast_reconstruct_loop(
            self.folding_tree,
            self.folding_tree_signs,
            self.edge_dirs,
            self.edge_ref_points,
            x,
            self.initial_nodes,
            self.face_indices_flat,
            self.face_offsets,
            self.num_faces,
            self.num_nodes
        )

# ===================================================================
# 2. Residual & Jacobian (Adapters)
# ===================================================================

@njit
def fast_reconstruct_loop(folding_tree, folding_tree_signs, 
                          edge_dirs, edge_ref_points, x, 
                          initial_nodes, 
                          face_indices_flat, face_offsets, 
                          num_faces, num_nodes):
    """
    Numba-compiled surface reconstruction.
    Speed: ~0.1ms for 2000 faces (vs 10ms in Python).
    """
    # 1. Allocate Transforms (Stack/Heap)
    # R: (num_faces, 3, 3), T: (num_faces, 3)
    Rs = np.zeros((num_faces, 3, 3), dtype=np.float64)
    Ts = np.zeros((num_faces, 3), dtype=np.float64)
    
    # Initialize Root (Face 0)
    Rs[0] = np.eye(3)
    # Ts[0] is already zeros
    
    # 2. Propagate Transforms (Linear Pass)
    num_steps = folding_tree.shape[0]
    for i in range(num_steps):
        parent = folding_tree[i, 0]
        child = folding_tree[i, 1]
        edge_idx = folding_tree[i, 2]
        sign = folding_tree_signs[i]
        
        # Get Parent Transform
        R_p = Rs[parent]
        T_p = Ts[parent]
        
        # Edge Geometry
        axis = edge_dirs[edge_idx]
        p_u = edge_ref_points[edge_idx]
        angle = sign * x[edge_idx]
        
        # --- Rotation Matrix (Rodrigues) Inline ---
        # Numba inlines this faster than a function call
        c, s = np.cos(angle), np.sin(angle)
        # Assuming axis is normalized (it usually is in rigid origami)
        kx, ky, kz = axis[0], axis[1], axis[2]
        
        K = np.array([
            [0.0, -kz,  ky],
            [ kz, 0.0, -kx],
            [-ky,  kx, 0.0]
        ])
        
        # R_hinge = I + s*K + (1-c)*(K@K)
        # Manual expansion for speed
        K2 = K @ K
        R_hinge = np.eye(3) + s * K + (1.0 - c) * K2
        
        # 3. Compute Child Transform
        R_c = R_p @ R_hinge
        # T_c = T_p + (R_p @ p_u) - (R_c @ p_u)
        # Factor out p_u: T_c = T_p + (R_p - R_c) @ p_u
        T_c = T_p + (R_p - R_c) @ p_u
        
        Rs[child] = R_c
        Ts[child] = T_c

    # 4. Compute Vertex Positions (Averaging)
    node_accum = np.zeros((num_nodes, 3), dtype=np.float64)
    node_counts = np.zeros(num_nodes, dtype=np.float64)
    
    for f in range(num_faces):
        # Slice the flat array for this face
        start = face_offsets[f]
        end = face_offsets[f+1]
        
        R_f = Rs[f]
        T_f = Ts[f]
        
        for k in range(start, end):
            v_idx = face_indices_flat[k]
            # Apply transform: new_pos = R @ initial_pos + T
            init_pos = initial_nodes[v_idx]
            
            # Manual matmul for vector (3,)
            # new_pos = R_f @ init_pos + T_f
            
            px = R_f[0,0]*init_pos[0] + R_f[0,1]*init_pos[1] + R_f[0,2]*init_pos[2] + T_f[0]
            py = R_f[1,0]*init_pos[0] + R_f[1,1]*init_pos[1] + R_f[1,2]*init_pos[2] + T_f[1]
            pz = R_f[2,0]*init_pos[0] + R_f[2,1]*init_pos[1] + R_f[2,2]*init_pos[2] + T_f[2]
            
            node_accum[v_idx, 0] += px
            node_accum[v_idx, 1] += py
            node_accum[v_idx, 2] += pz
            node_counts[v_idx] += 1.0

    # 5. Normalize
    for n in range(num_nodes):
        if node_counts[n] > 0:
            inv = 1.0 / node_counts[n]
            node_accum[n, 0] *= inv
            node_accum[n, 1] *= inv
            node_accum[n, 2] *= inv
            
    return node_accum

@njit(parallel=True)
def fast_jacobian_sparse(x, constraint_edge_indices, constraint_sector_angles, 
                         constraint_offsets, constraint_lengths, write_offsets,
                         J_data, J_rows, J_cols):
    """
    Computes Jacobian in Sparse Coordinate Format (COO).
    Writes directly into pre-allocated 1D arrays to avoid race conditions.
    """
    num_constraints = len(constraint_offsets)
    MAX_EDGES = 32  # Stack allocation constant
    
    for c_idx in prange(num_constraints):
        offset = constraint_offsets[c_idx]
        num_edges = constraint_lengths[c_idx]
        
        # Safety check for buffer size
        if num_edges > MAX_EDGES:
            num_edges = MAX_EDGES
        
        # Determine where this thread writes in the big 1D arrays
        start_ptr = write_offsets[c_idx]
        
        # Determine the logical row indices for this constraint (0, 1, 2)
        base_row_idx = c_idx * 3
        
        # Fixed-Size Allocation
        Ts = np.zeros((MAX_EDGES, 9), dtype=np.float64)
        dTs = np.zeros((MAX_EDGES, 9), dtype=np.float64)
        
        # --- PRECOMPUTE Ts and dTs ---
        for j in range(num_edges):
            edge_idx = constraint_edge_indices[offset + j]
            fold_angle = x[edge_idx]
            alpha = constraint_sector_angles[offset + j]
            
            # R_fold (Rotation about local X-axis)
            c_f, s_f = np.cos(fold_angle), np.sin(fold_angle)
            R_fold = np.array([1.0, 0.0, 0.0, 0.0, c_f, -s_f, 0.0, s_f, c_f])
            
            # Derivative of R_fold
            dR_fold = np.array([0.0, 0.0, 0.0, 0.0, -s_f, -c_f, 0.0, c_f, -s_f])
            
            # R_sector (Rotation about local Z-axis)
            c_s, s_s = np.cos(alpha), np.sin(alpha)
            R_sector = np.array([c_s, -s_s, 0.0, s_s, c_s, 0.0, 0.0, 0.0, 1.0])
            
            # T = R_sector @ R_fold
            T = np.zeros(9)
            dT = np.zeros(9)
            
            # Manual 3x3 Multiplication
            for i in range(3):
                row_offset = i * 3
                for k in range(3):
                    idx = row_offset + k
                    val_T = 0.0
                    val_dT = 0.0
                    for m in range(3):
                        r_s_val = R_sector[row_offset + m]
                        r_f_idx = m * 3 + k
                        val_T  += r_s_val * R_fold[r_f_idx]
                        val_dT += r_s_val * dR_fold[r_f_idx]
                    T[idx] = val_T
                    dT[idx] = val_dT
            
            Ts[j] = T
            dTs[j] = dT
        
        # --- COMPUTE DERIVATIVES (Chain Rule) ---
        for k in range(num_edges):
            target_edge_idx = constraint_edge_indices[offset + k]
            
            # Compute Left product (T_{N-1} ... T_{k+1})
            Left = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            for j in range(num_edges - 1, k, -1):
                new_Left = np.zeros(9)
                for i in range(3):
                    row_offset = i * 3
                    for col in range(3):
                        acc = 0.0
                        for m in range(3):
                            acc += Left[row_offset + m] * Ts[j][m * 3 + col]
                        new_Left[row_offset + col] = acc
                Left = new_Left
            
            # Compute Right product (T_{k-1} ... T_0)
            Right = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            for j in range(k):
                new_Right = np.zeros(9)
                for i in range(3):
                    row_offset = i * 3
                    for col in range(3):
                        acc = 0.0
                        for m in range(3):
                            acc += Ts[j][row_offset + m] * Right[m * 3 + col]
                        new_Right[row_offset + col] = acc
                Right = new_Right
            
            # dP = Left @ dTs[k] @ Right
            temp = np.zeros(9)
            for i in range(3):
                row_offset = i * 3
                for col in range(3):
                    acc = 0.0
                    for m in range(3):
                        acc += Left[row_offset + m] * dTs[k][m * 3 + col]
                    temp[row_offset + col] = acc
            
            dP = np.zeros(9)
            for i in range(3):
                row_offset = i * 3
                for col in range(3):
                    acc = 0.0
                    for m in range(3):
                        acc += temp[row_offset + m] * Right[m * 3 + col]
                    dP[row_offset + col] = acc
            
            # Map matrix derivative to residual derivative
            val_0 = 0.5 * (dP[2*3 + 1] - dP[1*3 + 2])
            val_1 = 0.5 * (dP[0*3 + 2] - dP[2*3 + 0])
            val_2 = 0.5 * (dP[1*3 + 0] - dP[0*3 + 1])
            
            # --- WRITE TO SPARSE ARRAYS ---
            # We stride by 3 because this edge generates 3 entries
            ptr = start_ptr + (k * 3)
            
            # Entry 1: (base_row, target_edge) -> val_0
            J_rows[ptr + 0] = base_row_idx + 0
            J_cols[ptr + 0] = target_edge_idx
            J_data[ptr + 0] = val_0
            
            # Entry 2: (base_row + 1, target_edge) -> val_1
            J_rows[ptr + 1] = base_row_idx + 1
            J_cols[ptr + 1] = target_edge_idx
            J_data[ptr + 1] = val_1
            
            # Entry 3: (base_row + 2, target_edge) -> val_2
            J_rows[ptr + 2] = base_row_idx + 2
            J_cols[ptr + 2] = target_edge_idx
            J_data[ptr + 2] = val_2

@njit(parallel=True)
def fast_residual_loop(x, constraint_edge_indices, constraint_sector_angles,
                       constraint_offsets, constraint_lengths):
    """
    Numba-compiled fast residual computation (parallelized).
    
    Args:
        x: fold angles array (n,)
        constraint_edge_indices: flat array of edge indices for all constraints
        constraint_sector_angles: flat array of sector angles for all constraints
        constraint_offsets: start indices for each constraint
        constraint_lengths: number of edges in each constraint
    
    Returns:
        res: residual vector (num_constraints * 3,)
    """
    num_constraints = len(constraint_offsets)
    # Output size is 3 * num_constraints
    res = np.empty(num_constraints * 3, dtype=np.float64)
    
    # Parallelize over constraints - each constraint writes to different indices
    for c_idx in prange(num_constraints):
        offset = constraint_offsets[c_idx]
        num_edges = constraint_lengths[c_idx]
        
        # Initialize Product Matrix P as Identity
        P = np.eye(3)
        
        for j in range(num_edges):
            edge_idx = constraint_edge_indices[offset + j]
            fold_angle = x[edge_idx]
            alpha = constraint_sector_angles[offset + j]
            
            cf, sf = np.cos(fold_angle), np.sin(fold_angle)
            cs, ss = np.cos(alpha), np.sin(alpha)
            
            # Construct T = R_z(alpha) @ R_x(fold) manually
            # T = [
            #   [cs, -ss*cf,  ss*sf],
            #   [ss,  cs*cf, -cs*sf],
            #   [ 0,     sf,     cf]
            # ]
            T00, T01, T02 = cs, -ss*cf, ss*sf
            T10, T11, T12 = ss,  cs*cf, -cs*sf
            T20, T21, T22 = 0.0,    sf,    cf
            
            # P_new = T @ P (Manual 3x3 multiplication)
            p00 = T00*P[0,0] + T01*P[1,0] + T02*P[2,0]
            p01 = T00*P[0,1] + T01*P[1,1] + T02*P[2,1]
            p02 = T00*P[0,2] + T01*P[1,2] + T02*P[2,2]
            
            p10 = T10*P[0,0] + T11*P[1,0] + T12*P[2,0]
            p11 = T10*P[0,1] + T11*P[1,1] + T12*P[2,1]
            p12 = T10*P[0,2] + T11*P[1,2] + T12*P[2,2]
            
            p20 = T20*P[0,0] + T21*P[1,0] + T22*P[2,0]
            p21 = T20*P[0,1] + T21*P[1,1] + T22*P[2,1]
            p22 = T20*P[0,2] + T21*P[1,2] + T22*P[2,2]
            
            # Update P
            P[0,0] = p00
            P[0,1] = p01
            P[0,2] = p02
            P[1,0] = p10
            P[1,1] = p11
            P[1,2] = p12
            P[2,0] = p20
            P[2,1] = p21
            P[2,2] = p22
        
        # Extract skew-symmetric error parts
        idx = c_idx * 3
        res[idx + 0] = 0.5 * (P[2, 1] - P[1, 2])
        res[idx + 1] = 0.5 * (P[0, 2] - P[2, 0])
        res[idx + 2] = 0.5 * (P[1, 0] - P[0, 1])
        
    return res

def compute_residual(x, proc):
    """
    Computes the geometric closure error for all internal vertices.
    Uses Numba JIT compilation for speedup.
    """
    if len(proc.constraints) == 0:
        return np.array([], dtype=np.float64)
    
    # Use Numba-accelerated version
    if not hasattr(proc, '_constraint_arrays'):
        proc._constraint_arrays = proc.prepare_constraint_arrays()
    
    constraint_edge_indices, constraint_sector_angles, constraint_offsets, constraint_lengths, _ = proc._constraint_arrays
    
    # Call Numba-compiled function
    return fast_residual_loop(x, constraint_edge_indices, constraint_sector_angles,
                              constraint_offsets, constraint_lengths)

def compute_jacobian_analytical(x, proc):
    """
    Computes the exact Jacobian using the analytical derivative of the rotation chain.
    Uses sparse COO format for efficiency with large grids.
    Uses Numba JIT compilation for speedup.
    """
    m = len(proc.constraints) * 3
    n = len(x)
    
    # Use Numba-accelerated version
    if len(proc.constraints) > 0:
        # Prepare constraint arrays (cache them in proc if not already present)
        if not hasattr(proc, '_constraint_arrays'):
            proc._constraint_arrays = proc.prepare_constraint_arrays()
        
        constraint_edge_indices, constraint_sector_angles, constraint_offsets, constraint_lengths, write_offsets = proc._constraint_arrays
        
        # Total number of non-zero elements (NNZ)
        total_nnz = write_offsets[-1]
        
        # Allocate 1D arrays for COO format
        J_data = np.zeros(total_nnz, dtype=np.float64)
        J_rows = np.zeros(total_nnz, dtype=np.int32)
        J_cols = np.zeros(total_nnz, dtype=np.int32)
        
        # Call Numba-compiled sparse function
        fast_jacobian_sparse(x, constraint_edge_indices, constraint_sector_angles,
                            constraint_offsets, constraint_lengths, write_offsets,
                            J_data, J_rows, J_cols)
        
        # Construct sparse COO matrix and convert to CSR for efficient slicing
        J = coo_matrix((J_data, (J_rows, J_cols)), shape=(m, n))
        return J.tocsr()  # Convert to CSR for efficient column slicing in solver

# ===================================================================
# 3. Newton-Raphson Solver
# ===================================================================

def NewtonSolver(eval_f, eval_Jf, x_start, p, errf=1e-10, errDeltax=1e-8, MaxIter=100, held_edges=None, verbose=True, print_angles=False):
    if verbose:
        print("Starting Newton-Raphson Solver...")
    
    k = 0
    N = len(x_start)
    x_curr = x_start.copy()
    
    if held_edges is None:
        held_edges = np.zeros(N, dtype=bool)
    
    # Initial error
    f = eval_f(x_curr, p)
    
    # Handle case where there are no constraints (empty residual)
    if len(f) == 0:
        if verbose:
            print("No constraints to satisfy. Returning initial state.")
        return x_curr
    
    errf_k = np.linalg.norm(f, np.inf)
    
    if verbose:
        print(f"Iter {k:>3}: ||f(x)|| = {errf_k:.4e}")
        if print_angles:
            angles_deg = np.rad2deg(x_curr)
            print(f"  Fold angles (deg): {angles_deg}")
    
    while k < MaxIter and errf_k > errf:
        # 1. Jacobian
        Jf = eval_Jf(x_curr, p)
        
        # 2. Linear Solve
        # We need to solve J * dx = -f
        # But we must enforce dx[held_edges] = 0.
        # We can zero out columns of J corresponding to held edges,
        # but that makes J singular.
        # Better approach: Remove columns of J for held variables from the solve.
        
        free_indices = np.where(~held_edges)[0]
        J_free = Jf[:, free_indices]
        f_vec = -f
        
        # Use sparse least squares solver for efficiency
        # lsmr (Least Squares Minimum Residual) is generally faster and more stable than lsqr
        # For small systems (<500 edges), dense solver might be faster, but sparse is better for large grids
        # lsmr returns (solution, istop, itn, normr, normar, norma, conda, normx)
        # We only need the solution (first element)
        result = lsmr(J_free, f_vec, atol=1e-8, btol=1e-8)
        dx_free = result[0]

        # Reconstruct full dx
        dx = np.zeros(N)
        dx[free_indices] = dx_free
        
        # 3. Update
        # Simple line search / damping to prevent flipping
        step_scale = 1.0
        max_step = 0.5 # radians
        if np.max(np.abs(dx)) > max_step:
            step_scale = max_step / np.max(np.abs(dx))
            
        x_curr += dx * step_scale
        
        # 4. Check errors
        f = eval_f(x_curr, p)
        
        # Handle case where residual becomes empty (shouldn't happen, but safety check)
        if len(f) == 0:
            if verbose:
                print("No constraints remaining. Converged.")
            break
        
        errf_k = np.linalg.norm(f, np.inf)
        errDx = np.linalg.norm(dx * step_scale, np.inf)
        
        k += 1
        if verbose:
            print(f"Iter {k:>3}: ||f(x)|| = {errf_k:.4e}, ||dx|| = {errDx:.4e}")
            if print_angles:
                angles_deg = np.rad2deg(x_curr)
                print(f"  Fold angles (deg): {angles_deg}")
        
        if errDx < errDeltax:
            if verbose:
                print("Converged by small step.")
            break

    return x_curr

# ===================================================================
# 4. Main & Visualization
# ===================================================================

def visualize_comparison(proc, x_init, x_final):
    n_i = proc.reconstruct_surface(x_init)
    n_f = proc.reconstruct_surface(x_final)
    
    faces_flat = []
    for f in proc.faces:
        faces_flat.append(len(f))
        faces_flat.extend(f)
    
    # Calculate edge midpoints for labeling using initial (flat) nodes
    edge_points = []
    edge_labels = []
    for edge_idx in range(proc.num_edges):
        u, v = proc.edges[edge_idx]
        midpoint = (n_i[u] + n_i[v]) / 2.0
        midpoint[2] += 0.05  # Small z-offset to lift label above mesh
        edge_points.append(midpoint)
        edge_labels.append(str(edge_idx))
    
    edge_points = np.array(edge_points)
    
    p = pv.Plotter(shape=(1, 2))
    
    # Left subplot: Initial (Flat)
    p.subplot(0,0)
    p.add_text("Initial (Flat)", font_size=10)
    mesh_i = pv.PolyData(n_i, faces_flat)
    p.add_mesh(mesh_i, color='lightblue', show_edges=True)
    # Add edge index labels to the initial (before) graph
    if len(edge_points) > 0:
        p.add_point_labels(edge_points, edge_labels, font_size=12, text_color='red', point_size=0, shape=None)
    p.reset_camera()  # Center and fit the mesh in the view
    p.view_isometric()
    
    # Right subplot: Animated Folding
    p.subplot(0,1)
    p.add_text("Folding Animation (Use Slider)", font_size=10)
    
    # Create initial mesh for animation
    mesh_anim = pv.PolyData(n_i.copy(), faces_flat)
    actor = p.add_mesh(mesh_anim, color='lightblue', show_edges=True)
    p.reset_camera()
    p.view_isometric()
    
    # Callback function to update mesh based on slider value
    def update_mesh(value):
        # Interpolate between initial and final fold angles
        t = value / 100.0  # Slider value is 0-100, convert to 0-1
        x_interp = (1 - t) * x_init + t * x_final
        
        # Reconstruct surface at interpolated state
        n_interp = proc.reconstruct_surface(x_interp)
        
        # Update mesh vertices
        mesh_anim.points = n_interp
        mesh_anim.Modified()  # Notify PyVista that mesh has changed
        p.render()
    
    # Add slider widget (positioned at the bottom of the window)
    p.add_slider_widget(
        update_mesh,
        rng=[0, 100],
        value=100,
        title='Folding Progress (%)',
        pointa=(0.02, 0.07),
        pointb=(0.98, 0.07),
        style='modern',
        title_height=0.02,
        fmt='%.0f',
        interaction_event='always'  # Update continuously while dragging
    )
    
    p.link_views()
    p.show()

def visualize_with_solver(proc, x_start_base, constraint_edges, eval_f_wrapper, eval_Jf_wrapper, 
                          held_mask_base, angle_min=-180, angle_max=180, 
                          initial_angle=0, draw_driven_edges=True, draw_labels=True):
    """
    Visualize origami with slider controlling target_angle.
    When slider changes, reruns the solver with the new target_angle value.
    
    Args:
        proc: MeshProcessor instance
        x_start_base: Base initial state (before applying constraints)
        constraint_edges: List of tuples (edge_idx, scale_factor)
                          scale_factor is (fold_angle / 180.0)
        eval_f_wrapper: Function to evaluate residual
        eval_Jf_wrapper: Function to evaluate Jacobian
        held_mask_base: Base held mask (before applying target_angle constraints)
        ...
    """
    n_i = proc.reconstruct_surface(x_start_base)
    
    faces_flat = []
    for f in proc.faces:
        faces_flat.append(len(f))
        faces_flat.extend(f)
    
    # Get list of driven edge indices (for coloring)
    driven_edge_indices = [edge_idx for edge_idx, _ in constraint_edges]
    
    p = pv.Plotter()
    
    # Enable Eye Dome Lighting (EDL) for shadows in creases
    p.enable_eye_dome_lighting()
    
    # Create initial mesh for animation
    mesh_anim = pv.PolyData(n_i.copy(), faces_flat)
    
    # Add mesh with matte paper-like finish
    # Blue for surface, light blue for backface
    actor = p.add_mesh(
        mesh_anim,
        color='#3498DB',  # Blue for front face
        backface_params=dict(color='#85C1E9'),  # Light blue for backface
        specular=0.1,     # Low specularity for matte paper look
        diffuse=0.8,      # Good diffuse reflection
        ambient=0.8,      # Ambient lighting
        smooth_shading=False,  # Smooth lighting across faces
        show_edges=True,
        edge_color='black'
    )
    
    # Create edge lines for driven edges (will be colored red)
    def create_driven_edges(nodes):
        """Create line segments for driven edges."""
        if not driven_edge_indices:
            return None
        
        # Build lines array: each line is [n_points, point1_idx, point2_idx, ...]
        lines = []
        for edge_idx in driven_edge_indices:
            u, v = proc.edges[edge_idx]
            lines.extend([2, u, v])
        
        if lines:
            # Create a PolyData with lines
            edge_mesh = pv.PolyData(nodes, lines=np.array(lines, dtype=np.int32))
            return edge_mesh
        return None
    
    # Helper functions to create vertex and edge labels
    def create_vertex_labels(nodes):
        """Create vertex label points and labels."""
        vertex_points = nodes.copy()
        # Small offset to lift labels above mesh
        vertex_points[:, 2] += 0.02
        vertex_labels = [f"V{i}" for i in range(len(nodes))]
        return vertex_points, vertex_labels
    
    def create_edge_labels(nodes):
        """Create edge label points and labels at edge midpoints."""
        edge_points = []
        edge_labels = []
        for edge_idx in range(proc.num_edges):
            u, v = proc.edges[edge_idx]
            midpoint = (nodes[u] + nodes[v]) / 2.0
            midpoint[2] += 0.02  # Small z-offset to lift label above mesh
            edge_points.append(midpoint)
            edge_labels.append(f"E{edge_idx}")
        return np.array(edge_points), edge_labels
    
    # Initial driven edges - add as red lines directly on the mesh (if enabled)
    driven_edges_actor = None
    if draw_driven_edges:
        driven_edges_mesh = create_driven_edges(n_i)
        if driven_edges_mesh is not None:
            # Add as simple red lines - they'll appear as colored edges on the mesh
            # Note: PyVista doesn't support per-line width, but red color makes them clearly visible
            driven_edges_actor = p.add_mesh(driven_edges_mesh, color='red')
            # Make lines thicker using renderer property
            if hasattr(driven_edges_actor, 'GetProperty'):
                driven_edges_actor.GetProperty().SetLineWidth(4)
    
    # Add vertex labels in orange (hover text) (if enabled)
    vertex_label_actor = None
    edge_label_actor = None
    if draw_labels:
        vertex_points, vertex_labels = create_vertex_labels(n_i)
        vertex_label_actor = p.add_point_labels(
            vertex_points, 
            vertex_labels, 
            font_size=16, 
            text_color='orange',
            point_size=4,  # Small visible points for hover
            shape=None,
            always_visible=False,  # Show labels on hover
            show_points=False,
            point_color='orange'
        )
        
        # Add edge labels in green (hover text)
        edge_points, edge_labels = create_edge_labels(n_i)
        edge_label_actor = p.add_point_labels(
            edge_points,
            edge_labels,
            font_size=16,
            text_color='green',
            point_size=4,  # Small visible points for hover
            shape=None,
            always_visible=False,  # Show labels on hover
            show_points=False,
            point_color='green'
        )
    
    p.reset_camera()
    p.view_isometric()
    
    # Store current solution to avoid recomputing if slider hasn't changed
    current_angle = initial_angle
    current_x_solution = None
    
    # Callback function to update mesh based on slider value (target_angle in degrees)
    def update_mesh(value):
        nonlocal current_angle, current_x_solution, driven_edges_actor, vertex_label_actor, edge_label_actor
        
        current_angle = value
        
        # Helper function to update labels
        def update_labels(nodes):
            nonlocal vertex_label_actor, edge_label_actor
            # Remove old labels
            if vertex_label_actor is not None:
                p.remove_actor(vertex_label_actor)
            if edge_label_actor is not None:
                p.remove_actor(edge_label_actor)
            
            # Create new labels with updated positions
            vertex_points, vertex_labels = create_vertex_labels(nodes)
            vertex_label_actor = p.add_point_labels(
                vertex_points,
                vertex_labels,
                font_size=16,
                text_color='orange',
                point_size=5,  # Small visible points for hover
                shape=None,
                always_visible=False,  # Show labels on hover
                show_points=False,
                point_color='orange'
            )
            
            edge_points, edge_labels = create_edge_labels(nodes)
            edge_label_actor = p.add_point_labels(
                edge_points,
                edge_labels,
                font_size=16,
                text_color='green',
                point_size=5,  # Small visible points for hover
                shape=None,
                always_visible=False,  # Show labels on hover
                show_points=False,
                point_color='green'
            )
        
        # If slider is at 0, reinitialize to flat state
        if abs(value) < 0.1:
            # Reset to flat state
            mesh_anim.points = n_i.copy()
            mesh_anim.Modified()
            
            # Update driven edges visualization for flat state (if enabled)
            if draw_driven_edges:
                if driven_edges_actor is not None:
                    p.remove_actor(driven_edges_actor)
                driven_edges_mesh = create_driven_edges(n_i)
                if driven_edges_mesh is not None:
                    driven_edges_actor = p.add_mesh(driven_edges_mesh, color='red')
                    if hasattr(driven_edges_actor, 'GetProperty'):
                        driven_edges_actor.GetProperty().SetLineWidth(3)
            
            # Update labels for flat state (if enabled)
            if draw_labels:
                update_labels(n_i)
            
            # Reset solution cache so it reinitializes when moving away from 0
            current_x_solution = None
            
            # =========================================================
            # Auto-Focus and Auto-Zoom Logic
            # =========================================================
            
            # 1. Get the new center of the mesh (bounding box center)
            new_center = np.array(mesh_anim.center)
            
            # 2. Get the new size (diagonal length of bounding box)
            #    We use max() to prevent camera crashing into the object if it collapses to a point
            new_size = max(mesh_anim.length, 0.1) 
            
            # 3. Get current camera orientation vectors
            cam_pos = np.array(p.camera.position)
            cam_focus = np.array(p.camera.focal_point)
            
            # Calculate the vector from focus to camera (the "viewing direction" reversed)
            vec = cam_pos - cam_focus
            dist = np.linalg.norm(vec)
            
            # Normalize direction so we can scale it
            if dist > 0:
                direction = vec / dist
            else:
                direction = np.array([0, 0, 1]) # Fallback
                
            # 4. Update Focal Point: Lock onto the new center of the origami
            p.camera.focal_point = new_center
            
            # 5. Update Position: Move camera along the view vector to maintain framing
            #    Factor 2.0 ensures the object fills most of the screen but isn't cut off.
            #    Adjust 2.0 to 2.5 if you want it further away.
            p.camera.position = new_center + (direction * new_size * 2.0)
            
            # 6. Update Clipping Range to prevent the mesh from being sliced 
            #    by the near/far planes as it gets smaller
            p.camera.clipping_range = (new_size * 0.01, new_size * 100)
            
            # =========================================================
            
            p.render()
            return
        
        # Use previous solution as initial guess if available, otherwise use base
        if current_x_solution is not None:
            x_start = current_x_solution.copy()
        else:
            x_start = x_start_base.copy()
        
        held_mask = held_mask_base.copy()
        
        # --- CHANGED LOGIC HERE ---
        # Apply target_angle constraints with scaling factor
        for edge_idx, scale_factor in constraint_edges:
            # angle = scale_factor * input_angle
            angle = np.deg2rad(scale_factor * value)
            x_start[edge_idx] = angle
            held_mask[edge_idx] = True
        # --------------------------
        
        # Rerun solver (quietly, no print statements)
        try:
            x_solution = NewtonSolver(
                eval_f_wrapper,
                eval_Jf_wrapper,
                x_start,
                p=proc,
                held_edges=held_mask,
                errf=1e-6,  # Slightly relaxed for faster convergence
                errDeltax=1e-6,
                MaxIter=25,  # Limit iterations for interactivity
                verbose=False  # Suppress output during interactive updates
            )
            current_x_solution = x_solution
            
            # Reconstruct surface at solved state
            n_solved = proc.reconstruct_surface(x_solution)
            
            # Update mesh vertices
            mesh_anim.points = n_solved
            mesh_anim.Modified()
            
            # Update driven edges visualization (if enabled)
            if draw_driven_edges:
                if driven_edges_actor is not None:
                    p.remove_actor(driven_edges_actor)
                driven_edges_mesh = create_driven_edges(n_solved)
                if driven_edges_mesh is not None:
                    # Add as simple red lines - they'll appear as colored edges on the mesh
                    # Note: PyVista doesn't support per-line width, but red color makes them clearly visible
                    driven_edges_actor = p.add_mesh(driven_edges_mesh, color='red')
                    # Make lines thicker using renderer property
                    if hasattr(driven_edges_actor, 'GetProperty'):
                        driven_edges_actor.GetProperty().SetLineWidth(3)
            
            # Update labels for solved state (if enabled)
            if draw_labels:
                update_labels(n_solved)
            
            # =========================================================
            # Auto-Focus and Auto-Zoom Logic
            # =========================================================
            
            # 1. Get the new center of the mesh (bounding box center)
            new_center = np.array(mesh_anim.center)
            
            # 2. Get the new size (diagonal length of bounding box)
            #    We use max() to prevent camera crashing into the object if it collapses to a point
            new_size = max(mesh_anim.length, 0.1) 
            
            # 3. Get current camera orientation vectors
            cam_pos = np.array(p.camera.position)
            cam_focus = np.array(p.camera.focal_point)
            
            # Calculate the vector from focus to camera (the "viewing direction" reversed)
            vec = cam_pos - cam_focus
            dist = np.linalg.norm(vec)
            
            # Normalize direction so we can scale it
            if dist > 0:
                direction = vec / dist
            else:
                direction = np.array([0, 0, 1]) # Fallback
                
            # 4. Update Focal Point: Lock onto the new center of the origami
            p.camera.focal_point = new_center
            
            # 5. Update Position: Move camera along the view vector to maintain framing
            #    Factor 2.0 ensures the object fills most of the screen but isn't cut off.
            #    Adjust 2.0 to 2.5 if you want it further away.
            p.camera.position = new_center + (direction * new_size * 2.0)
            
            # 6. Update Clipping Range to prevent the mesh from being sliced 
            #    by the near/far planes as it gets smaller
            p.camera.clipping_range = (new_size * 0.01, new_size * 100)
            
            # =========================================================
            
            p.render()
        except Exception as e:
            print(f"Error solving for target_angle={value:.1f}°: {e}")
    
    # Run initial solve
    update_mesh(initial_angle)
    
    # Add slider widget (positioned at the bottom of the window)
    slider_widget = p.add_slider_widget(
        update_mesh,
        rng=[angle_min, angle_max],
        value=initial_angle,
        pointa=(0.02, 0.05),
        pointb=(0.98, 0.05),
        style='modern',
        fmt='%.1f',
        interaction_event='always'  # Update continuously while dragging
    )
    
    # Add key press handler for space to reset slider to 0
    def handle_key_press(caller, event):
        """Handle key press events - space resets slider to 0."""
        key = caller.GetKeySym()
        if key == 'space':
            # Reset slider to 0
            if slider_widget is not None:
                slider_widget.GetRepresentation().SetValue(initial_angle)
                update_mesh(initial_angle)
    
    p.iren.add_observer("KeyPressEvent", handle_key_press)
    
    p.show()

def input_fold(filename, drive_fold_groups=[1], initial_angle=0):
    """
    Parses a .FOLD file. Uses 'edges_foldAngle' to determine which edges to drive.
    Drives edges using a scale factor = edges_foldAngle / 180.0.
    """
    # Add .fold extension if not present
    if not filename.endswith('.fold'):
        filename = filename + '.fold'
    
    with open(filename, 'r') as f:
        content = f.read()
        # Handle Python-style booleans (True/False) by converting to JSON booleans (true/false)
        content = content.replace('True', 'true').replace('False', 'false')
        data = json.loads(content)
    
    # 1. Geometry
    nodes = np.array(data['vertices_coords'], dtype=np.float64)
    
    faces = data['faces_vertices']
    
    # Initialize Processor
    proc = MeshProcessor(nodes, faces)
    
    # Precompute folding tree for fast surface reconstruction
    proc.precompute_folding_tree()
    
    # 2. Extract Edge Data
    file_edges = data.get('edges_vertices', [])
    file_assignments = data.get('edges_assignment', [])
    file_angles = data.get('edges_foldAngle', [])
    
    # Create lookups for assignments and angles, and map file edges to processor edges
    assignment_map = {}
    angle_map = {}
    file_edge_to_proc_edge = {}  # Map file edge index to processor edge index
    
    # First, create a map from edge key (sorted vertex pair) to processor edge index
    proc_edge_key_to_idx = {}
    for proc_edge_idx, (u, v) in enumerate(proc.edges):
        key = tuple(sorted((u, v)))
        proc_edge_key_to_idx[key] = proc_edge_idx
    
    # Now map file edges to processor edges
    for i, file_edge in enumerate(file_edges):
        key = tuple(sorted((file_edge[0], file_edge[1])))
        
        # Store assignment
        if i < len(file_assignments):
            assignment_map[key] = file_assignments[i]
        else:
            assignment_map[key] = 'U'  # Unknown
        
        # Store angle (can be None/null)
        if i < len(file_angles):
            angle_map[key] = file_angles[i] if file_angles[i] is not None else None
        else:
            angle_map[key] = None
        
        # Map to processor edge if it exists
        if key in proc_edge_key_to_idx:
            file_edge_to_proc_edge[i] = proc_edge_key_to_idx[key]
    
    # 3. Determine which edges to drive based on edges_foldAngle
    # Find all groups of consecutive non-null angles with the same value,
    # then select the nth group based on drive_fold_group
    
    # First, find all groups of consecutive non-null angles
    groups = []
    current_group = []
    current_group_value = None
    
    for i in range(len(file_angles)):
        angle = file_angles[i] if i < len(file_angles) else None
        
        if angle is None:
            # Null angle - end current group if exists
            if current_group:
                groups.append((current_group, current_group_value))
                current_group = []
                current_group_value = None
        else:
            # Non-null angle found - normalize to float for consistent comparison
            angle_val = float(angle)
            
            if not current_group:
                # Start a new group
                current_group = [i]
                current_group_value = angle
            else:
                # Compare with current group value (normalize to float)
                current_val = float(current_group_value)
                if angle_val == current_val:
                    # Continue current group (same value)
                    current_group.append(i)
                else:
                    # Value changed - save current group and start new one
                    if current_group:
                        groups.append((current_group, current_group_value))
                    current_group = [i]
                    current_group_value = angle
    
    # Save the last group if it exists
    if current_group:
        groups.append((current_group, current_group_value))
    
    # Normalize drive_fold_groups to a list if a single integer was passed (for backward compatibility)
    if isinstance(drive_fold_groups, int):
        drive_fold_groups = [drive_fold_groups]
    
    # Validate requested groups
    if not drive_fold_groups:
        raise ValueError("drive_fold_groups cannot be empty")
    
    for group_num in drive_fold_groups:
        if group_num < 1:
            raise ValueError(f"drive_fold_groups must contain values >= 1, got {group_num}")
        if len(groups) < group_num:
            raise ValueError(f"Only {len(groups)} group(s) of non-null angles found, but group {group_num} was requested")
    
    # Collect edges from all requested groups
    driven_file_edge_indices = []
    target_angle_values = []  # Store target angles for each group (may be different)
    
    for group_num in drive_fold_groups:
        group_indices, group_value = groups[group_num - 1]
        driven_file_edge_indices.extend(group_indices)
        target_angle_values.append(group_value)
    
    # For sign determination, we'll use the first group's angle value
    # (though in practice, each edge will use its own assignment-based sign)
    target_angle_value = target_angle_values[0] if target_angle_values else None
    
    # Convert driven file edge indices to processor edge indices
    driven_proc_edge_indices = set()
    for file_edge_idx in driven_file_edge_indices:
        if file_edge_idx in file_edge_to_proc_edge:
            driven_proc_edge_indices.add(file_edge_to_proc_edge[file_edge_idx])
    
    # Create reverse mapping from processor edge index to file edge index
    proc_edge_to_file_edge = {}
    for file_edge_idx, proc_edge_idx in file_edge_to_proc_edge.items():
        proc_edge_to_file_edge[proc_edge_idx] = file_edge_idx
    
    # 4. Build Constraints
    x_start = np.zeros(proc.num_edges)
    held_mask = np.zeros(proc.num_edges, dtype=bool)
    constraint_edges = []
    
    for i, (u, v) in enumerate(proc.edges):
        key = tuple(sorted((u, v)))
        assignment = assignment_map.get(key, 'U')
        
        is_driven = i in driven_proc_edge_indices
        
        if is_driven:
            # NEW LOGIC: Use the actual angle from the file to determine scale
            angle_val = angle_map.get(key)
            
            if angle_val is not None:
                # Scale input based on 180 degrees
                # e.g., if angle is 90, scale is 0.5. If input is 180, result is 90.
                scale = angle_val / 180.0
                constraint_edges.append((i, scale))
            else:
                # Fallback if angle is missing (should not happen for driven edges typically)
                if assignment == 'M':
                    constraint_edges.append((i, 1.0))
                elif assignment == 'V':
                    constraint_edges.append((i, -1.0))
                else:
                    constraint_edges.append((i, 1.0))
        else:
            if assignment == 'B' or assignment == 'F':
                held_mask[i] = True
                x_start[i] = 0.0
    
    return {
        'proc': proc,
        'x_start_base': x_start,
        'constraint_edges': constraint_edges,
        'held_mask_base': held_mask,
        'initial_angle': initial_angle,
        'angle_min': -180,
        'angle_max': 180
    }

def generate_miura_ori_grid(m, n, alpha_target, initial_angle=0):
    """
    Sets up the miura-ori example.
    Returns a dictionary with all data needed for visualization and solving.
    """
    
    # 1. Generate Mesh
    nx = n * 2 + 1
    ny = m * 2 + 1
    l1 = 1.0
    l2 = 1.0
    alpha = np.deg2rad(alpha_target)
    psi = np.deg2rad(90 - alpha_target)
    nodes = []
    for j in range(ny):
        y = j * l2
        for i in range(nx):
            x_shift = 0
            if j % 2 != 0:
                x_shift = l2 * np.tan(psi)
            x = i * l1 + x_shift
            nodes.append([x, y, 0.0])
    nodes = np.array(nodes)
    
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j*nx + i
            n1 = j*nx + (i+1)
            n2 = (j+1)*nx + (i+1)
            n3 = (j+1)*nx + i
            faces.append([n0, n1, n2, n3])

    proc = MeshProcessor(nodes, faces)
    
    # Precompute folding tree for fast surface reconstruction
    proc.precompute_folding_tree()
    
    # 2. Initialization

    x_start = np.zeros(proc.num_edges)

    # Find horizontal edges in first row of faces for initial nudges
    # These are edges connecting nodes in row 1 (first row of faces)
    initial_nudge_edges = []
    for i, (u, v) in enumerate(proc.edges):
        u_row = u // nx
        v_row = v // nx
        if u_row == 1 and v_row == 1:
            u_col = u % nx
            v_col = v % nx
            if abs(u_col - v_col) == 1:
                initial_nudge_edges.append(i)
    initial_nudge_edges.sort()

    # Find vertical constraint edges (skip first column, col 0)
    # These connect rows and are used as constraint edges
    constraint_edge_list = []
    for i, (u, v) in enumerate(proc.edges):
        u_row = u // nx
        v_row = v // nx
        if u_row != v_row:
            u_col = u % nx
            v_col = v % nx
            if u_col == v_col and u_col > 0:  # Skip first column
                constraint_edge_list.append((i, min(u_row, v_row), u_col))
    
    # Sort by row transition, then by column
    constraint_edge_list.sort(key=lambda x: (x[1], x[2]))
    constraint_edge_indices = [e[0] for e in constraint_edge_list]

    # 3. Setup base state (without target_angle constraints)
    # x_start_base contains only the epsilon perturbations
    x_start_base = x_start.copy()
    
    # Define which edges use target_angle and their signs
    # Format: (edge_idx, sign) where sign is +1 or -1
    # Use the programmatically found constraint edges with alternating signs
    constraint_edges = []
    for idx, edge_idx in enumerate(constraint_edge_indices):
        sign = -1 if idx % 2 == 0 else 1
        constraint_edges.append((edge_idx, sign))
    
    # Base held_mask (empty, will be populated in visualization callback)
    held_mask_base = np.zeros(proc.num_edges, dtype=bool)
    
    return {
        'proc': proc,
        'x_start_base': x_start_base,
        'constraint_edges': constraint_edges,
        'held_mask_base': held_mask_base,
        'initial_angle': initial_angle,
        'angle_min': -180,
        'angle_max': 180
    }

def generate_waterbomb_grid(n, m, initial_angle=0):
    nodes = []
    faces = []
    index_map = {}
    index_count = 0
    diagonals = set()
    edges = set()
    edge_vertices = set()
    size = 1.0

    def add_node(pos):
        nonlocal nodes, index_map, index_count
        x, y = pos
        nodes.append([x, y, 0])
        index_map[pos] = index_count
        index_count += 1
        return

    def add_face(a, b, c):
        nonlocal faces, index_map
        faces.append([index_map[a], index_map[b], index_map[c]])

    def add_regular_unit(row, col):
        nonlocal edge_vertices, edges, diagonals, index_map
        x = col * 2 * size
        y = row * 2 * size
        
        top_left = (x, y)
        top_midpoint = (x + size, y)
        top_right = (x + 2 * size, y)
        bottom_left = (x, y + 2 * size)
        center = (x + size, y + size)
        bottom_midpoint = (x + size, y + 2 * size)
        bottom_right = (x + 2 * size, y + 2 * size)

        # add vertices
        if row == 0 and col == 0:
            add_node(top_left)
            edge_vertices.add(top_left)

        if row == 0:
            add_node(top_midpoint)
            edge_vertices.add(top_midpoint)
            add_node(top_right)
            edge_vertices.add(top_right)

            edges.add(frozenset([index_map[top_left], index_map[top_midpoint]]))
            edges.add(frozenset([index_map[top_midpoint], index_map[top_right]]))

        if col == 0:
            add_node(bottom_left)
            edge_vertices.add(bottom_left)

            edges.add(frozenset([index_map[top_left], index_map[bottom_left]]))

        add_node(center)
        add_node(bottom_midpoint)
        add_node(bottom_right)

        if row == m - 1:
            edge_vertices.add(bottom_left)
            edge_vertices.add(bottom_midpoint)

            edges.add(frozenset([index_map[bottom_left], index_map[bottom_midpoint]]))
            edges.add(frozenset([index_map[bottom_midpoint], index_map[bottom_right]]))

        if col == n - 1:
            edge_vertices.add(bottom_right)

            edges.add(frozenset([index_map[top_right], index_map[bottom_right]]))

        # track diagonals
        diagonals.add(frozenset([index_map[top_left], index_map[center]]))
        diagonals.add(frozenset([index_map[bottom_left], index_map[center]]))
        diagonals.add(frozenset([index_map[center], index_map[top_right]]))
        diagonals.add(frozenset([index_map[center], index_map[bottom_right]]))

        # add faces - CCW winding
        add_face(top_left, top_midpoint, center)
        add_face(top_left, center, bottom_left)
        add_face(bottom_left, center, bottom_midpoint)
        add_face(bottom_midpoint, center, bottom_right)
        add_face(bottom_right, center, top_right)
        add_face(top_midpoint, top_right, center)

    def add_offset_unit(row, col):
        nonlocal edge_vertices, edges, diagonals, index_map
        x = col * 2 * size
        y = row * 2 * size
        
        top_left = (x, y)
        top_midpoint = (x + size, y)
        top_right = (x + 2 * size, y)
        left_midpoint = (x, y + size)
        bottom_left = (x, y + 2 * size)
        bottom_midpoint = (x + size, y + 2 * size)
        bottom_right = (x + 2 * size, y + 2 * size)
        right_midpoint = (x + 2 * size, y + size)

        # add vertices
        if col == 0:
            add_node(left_midpoint)
            edge_vertices.add(left_midpoint)
            add_node(bottom_left)
            edge_vertices.add(bottom_left)

            edges.add(frozenset([index_map[top_left], index_map[left_midpoint]]))
            edges.add(frozenset([index_map[left_midpoint], index_map[bottom_left]]))

        add_node(bottom_midpoint)
        add_node(bottom_right)
        add_node(right_midpoint)

        if row == m - 1:
            edge_vertices.add(bottom_left)
            edge_vertices.add(bottom_midpoint)

            edges.add(frozenset([index_map[bottom_left], index_map[bottom_midpoint]]))
            edges.add(frozenset([index_map[bottom_midpoint], index_map[bottom_right]]))
        if col == n - 1:
            edge_vertices.add(right_midpoint)
            edge_vertices.add(bottom_right)

            edges.add(frozenset([index_map[top_right], index_map[right_midpoint]]))
            edges.add(frozenset([index_map[right_midpoint], index_map[bottom_right]]))

        # track diagonals
        diagonals.add(frozenset([index_map[left_midpoint], index_map[top_midpoint]]))
        diagonals.add(frozenset([index_map[top_midpoint], index_map[right_midpoint]]))
        diagonals.add(frozenset([index_map[left_midpoint], index_map[bottom_midpoint]]))
        diagonals.add(frozenset([index_map[bottom_midpoint], index_map[right_midpoint]]))

        # add faces - CCW winding
        add_face(top_left, top_midpoint, left_midpoint)
        add_face(left_midpoint, bottom_midpoint, bottom_left)
        add_face(left_midpoint, top_midpoint, bottom_midpoint)
        add_face(bottom_midpoint, right_midpoint, bottom_right)
        add_face(top_midpoint, right_midpoint, bottom_midpoint)
        add_face(top_midpoint, top_right, right_midpoint)

    for row in range(m):
        for col in range(n):
            if row % 2 == 0:
                # even row: waterbomb unit
                add_regular_unit(row, col)
            else:
                # odd row: offset waterbomb unit
                add_offset_unit(row, col)
                
    nodes = np.array(nodes)

    proc = MeshProcessor(nodes, faces)
    
    # Precompute folding tree for fast surface reconstruction
    proc.precompute_folding_tree()

    x_start = np.zeros(proc.num_edges)

    constraint_edges = []
    for i, (u, v) in enumerate(proc.edges):
        if frozenset([u, v]) in diagonals:
            constraint_edges.append((i, 1))

    x_start_base = x_start.copy()

    held_mask_base = np.zeros(proc.num_edges, dtype=bool)

    for i, (u, v) in enumerate(proc.edges):
        if frozenset([u, v]) in edges: # lock the boundary edges of the paper, so the solver doesn't try to fold them
            x_start[i] = 0
            held_mask_base[i] = True
    
    return {
        'proc': proc,
        'x_start_base': x_start_base,
        'constraint_edges': constraint_edges,
        'held_mask_base': held_mask_base,
        'initial_angle': initial_angle,
        'angle_min': -180,
        'angle_max': 180
    }

if __name__ == "__main__":
    # Get example-specific data
    # example_data = generate_miura_ori_grid(1, 1, 70,)
    # example_data = generate_miura_ori_grid(4, 4, 70)
    # example_data = generate_miura_ori_grid(50, 50, 70)
    # example_data = generate_waterbomb_grid(1, 1)
    # example_data = generate_waterbomb_grid(6, 3)
    # example_data = generate_waterbomb_grid(20, 20)
    # example_data = input_fold("squareBase", drive_fold_groups=[1,2])
    # example_data = input_fold("birdBase")
    # example_data = input_fold("waterbombBase", drive_fold_groups)
    # example_data = input_fold("huffmanWaterbomb", drive_fold_groups=[2])
    example_data = input_fold("huffmanRectangularWeave")
    proc = example_data['proc']
    x_start_base = example_data['x_start_base']
    constraint_edges = example_data['constraint_edges']
    held_mask_base = example_data['held_mask_base']
    initial_angle = example_data['initial_angle']
    angle_min = example_data['angle_min']
    angle_max = example_data['angle_max']
    
    # Warm up Numba JIT compilation at startup
    proc.warmup_numba_jit()
    
    # Wrappers to match NewtonSolver signature
    def eval_f_wrapper(x, p):
        return compute_residual(x, p)
        
    def eval_Jf_wrapper(x, p):
        return compute_jacobian_analytical(x, p)

    # Run initial solver for output (with initial angle)
    # Build constraints_initial from constraint_edges
    constraints_initial = {}
    for edge_idx, scale_factor in constraint_edges:
        constraints_initial[edge_idx] = np.deg2rad(scale_factor * initial_angle)
    
    x_start_initial = x_start_base.copy()
    held_mask_initial = held_mask_base.copy()
    for idx, ang in constraints_initial.items():
        x_start_initial[idx] = ang
        held_mask_initial[idx] = True
    
    verbose = False

    x_final = NewtonSolver(
        eval_f_wrapper,
        eval_Jf_wrapper,
        x_start_initial,
        p=proc,
        held_edges=held_mask_initial,
        verbose=verbose
    )
    
    if verbose:
        # Output Results
        print("\n" + "="*50)
        print(f"{'Edge ID':<10} | {'Nodes (u,v)':<15} | {'Final Angle (deg)':<20}")
        print("-" * 50)
        
        # sorted_indices = np.argsort(np.abs(x_final))[::-1]
        sorted_indices = range(proc.num_edges)
        for i in sorted_indices:
            u, v = proc.edges[i]
            angle_deg = np.rad2deg(x_final[i])
            status = "" 
            if held_mask_initial[i]: status = "(Fixed)"
            elif abs(angle_deg) < 1e-4: status = "(Flat)"
            elif angle_deg > 0:       status = "(Valley)"
            else:                     status = "(Mountain)"
                
            print(f"{i:<10} | {f'{u}-{v}':<15} | {angle_deg:<10.4f} {status}")
        print("="*50)

    # Visualize with interactive slider
    visualize_with_solver(
        proc, 
        x_start_base, 
        constraint_edges,
        eval_f_wrapper,
        eval_Jf_wrapper,
        held_mask_base,
        angle_min=angle_min,
        angle_max=angle_max,
        initial_angle=initial_angle,
        draw_driven_edges=False,
        draw_labels=False
    )