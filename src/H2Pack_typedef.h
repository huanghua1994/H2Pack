#ifndef __H2PACK_TYPEDEF_H__
#define __H2PACK_TYPEDEF_H__

#include "H2Pack_config.h"
#include "H2Pack_aux_structs.h"
#include "DAG_task_queue.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pointer to function that evaluates a kernel matrix using given sets of points.
// The kernel function must by symmetric.
// Input parameters:
//   coord0     : Matrix, size pt_dim-by-ld0, coordinates of the 1st point set
//   ld0        : Leading dimension of coord0, should be >= n0
//   n0         : Number of points in coord0 (each column in coord0 is a coordinate)
//   coord1     : Matrix, size pt_dim-by-ld1, coordinates of the 2nd point set
//   ld1        : Leading dimension of coord1, should be >= n1
//   n1         : Number of points in coord1 (each column in coord0 is a coordinate)
//   ldm        : Leading dimension of the kernel matrix
//   krnl_param : Pointer to kernel function parameter array
// Output parameter:
//   mat : Obtained kernel matrix, size n0-by-ld1
typedef void (*kernel_eval_fptr) (
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const void *krnl_param, DTYPE * __restrict mat, const int ldm
);

// Pointer to function that performs kernel matrix matvec using a given set
// of points and a given input vector:
//   x_out = kernel_matrix(coord0, coord1) * x_in
// Input parameters:
//   coord0     : Matrix, size pt_dim-by-ld0, coordinates of the 1st point set
//   ld0        : Leading dimension of coord0, should be >= n0
//   n0         : Number of points in coord0 (each column in coord0 is a coordinate)
//   coord1     : Matrix, size pt_dim-by-ld1, coordinates of the 2nd point set
//   ld1        : Leading dimension of coord1, should be >= n1
//   n1         : Number of points in coord1 (each column in coord0 is a coordinate)
//   krnl_param : Pointer to kernel function parameter array
//   x_in       : Vector, size >= krnl_dim * n1, will be left multiplied by kernel_matrix(coord0, coord1)
// Output parameters:
//   x_out : Vector, size >= krnl_dim * n0, x_out_0 += kernel_matrix(coord0, coord1) * x_in_0
// Performance optimization notes:
//   When calling a kernel_mv_fptr, H2Pack guarantees that:
//     (1) n{0,1} are multiples of SIMD_LEN;
//     (2) The lengths of x_{in,out} are multiples of (SIMD_LEN * krnl_dim)
//     (3) The addresses of coord{0,1}, x_{in,out} are aligned to (SIMD_LEN * sizeof(DTYPE))
typedef void (*kernel_mv_fptr) (
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const void *krnl_param, const DTYPE *x_in, DTYPE * __restrict x_out
);

// Pointer to function that performs kernel matrix bi-matvec using given sets
// of points and given input vectors. The kernel function must be symmetric.
// This function computes:
//   (1) x_out_0 = kernel_matrix(coord0, coord1) * x_in_0,
//   (2) x_out_1 = kernel_matrix(coord1, coord0) * x_in_1,
//   where kernel_matrix(coord0, coord1)^T = kernel_matrix(coord1, coord0).
// Input parameters:
//   coord0     : Matrix, size pt_dim-by-ld0, coordinates of the 1st point set
//   ld0        : Leading dimension of coord0, should be >= n0
//   n0         : Number of points in coord0 (each column in coord0 is a coordinate)
//   coord1     : Matrix, size pt_dim-by-ld1, coordinates of the 2nd point set
//   ld1        : Leading dimension of coord1, should be >= n1
//   n1         : Number of points in coord1 (each column in coord0 is a coordinate)
//   krnl_param : Pointer to kernel function parameter array
//   x_in_0     : Vector, size >= krnl_dim * n1, will be left multiplied by kernel_matrix(coord0, coord1)
//   x_in_1     : Vector, size >= krnl_dim * n0, will be left multiplied by kernel_matrix(coord1, coord0).
// Output parameters:
//   x_out_0 : Vector, size >= krnl_dim * n0, x_out_0 += kernel_matrix(coord0, coord1) * x_in_0
//   x_out_1 : Vector, size >= krnl_dim * n1, x_out_1 += kernel_matrix(coord1, coord0) * x_in_1
// Performance optimization notes:
//   When calling a kernel_bimv_fptr, H2Pack guarantees that:
//     (1) n{0,1} are multiples of SIMD_LEN;
//     (2) The lengths of x_{in,out}_{0,1} are multiples of (SIMD_LEN * krnl_dim)
//     (3) The addresses of coord{0,1}, x_{in,out}_{0,1} are aligned to (SIMD_LEN * sizeof(DTYPE))
typedef void (*kernel_bimv_fptr) (
    const DTYPE *coord0, const int ld0, const int n0,
    const DTYPE *coord1, const int ld1, const int n1,
    const void *krnl_param, const DTYPE *x_in_0, const DTYPE *x_in_1, 
    DTYPE * __restrict x_out_0, DTYPE * __restrict x_out_1
);

// Structure of H2 matrix tree flatten representation
struct H2Pack
{
    // H2 matrix tree flatten representation
    int    n_thread;                // Number of threads
    int    pt_dim;                  // Dimension of point coordinate
    int    xpt_dim;                 // Dimension of extended point coordinate (for RPY)
    int    krnl_dim;                // Dimension of tensor kernel's return
    int    QR_stop_type;            // Partial QR stop criteria
    int    QR_stop_rank;            // Partial QR maximum rank
    int    n_point;                 // Number of points for the kernel matrix
    int    krnl_mat_size;           // Size of the kernel matrix
    int    krnl_bimv_flops;         // FLOPs needed in kernel bi-matvec
    int    max_leaf_points;         // Maximum point in a leaf node's box
    int    n_node;                  // Number of nodes in this H2 tree
    int    root_idx;                // Index of the root node (== n_node - 1, save it for convenience)
    int    n_leaf_node;             // Number of leaf nodes in this H2 tree
    int    max_child;               // Maximum number of children per node, == 2^pt_dim
    int    max_neighbor;            // Maximum number of neighbor nodes per node, == 2^pt_dim
    int    max_level;               // Maximum level of this H2 tree, (root = 0, total max_level + 1 levels)
    int    min_adm_level;           // Minimum level of reduced admissible pair
    int    HSS_min_adm_level;       // Minimum level of reduced admissible pair in HSS mode
    int    n_r_inadm_pair;          // Number of reduced inadmissible pairs 
    int    n_r_adm_pair;            // Number of reduced admissible pairs 
    int    HSS_n_r_inadm_pair;      // Number of reduced inadmissible pairs in HSS mode
    int    HSS_n_r_adm_pair;        // Number of reduced admissible pairs in HSS mode
    int    n_UJ;                    // Number of projection matrices & skeleton row sets, == n_node
    int    n_B;                     // Number of generator matrices
    int    n_D;                     // Number of dense blocks
    int    mm_max_n_vec;            // Maximum number of vectors that can be multiplied in matmul
    int    BD_JIT;                  // If B and D matrices are computed just-in-time in matvec
    int    is_H2ERI;                // If H2Pack is called from H2ERI
    int    is_HSS;                  // If H2Pack is running in HSS mode
    int    is_RPY;                  // If H2Pack is running RPY kernel
    int    is_RPY_Ewald;            // If H2Pack is running RPY Ewald summation kernel
    int    is_HSS_SPD;              // If H2Pack in HSS mode is SPD 
    int    n_lattice;               // Number of periodic lattices, == 3^pt_dim
    int    print_timers;            // If H2Pack prints internal timers for performance analysis
    int    print_dbginfo;           // If H2Pack prints debug information
    int    *parent;                 // Size n_node, parent index of each node
    int    *children;               // Size n_node * max_child, indices of a node's children nodes
    int    *pt_cluster;             // Size n_node * 2, start and end (included) indices of points belong to each node
    int    *mat_cluster;            // Size n_node * 2, start and end (included) indices of matvec vector elements belong to each node
    int    *n_child;                // Size n_node, number of children nodes of each node
    int    *node_level;             // Size n_node, level of each node
    int    *node_height;            // Size n_node, height of each node
    int    *level_n_node;           // Size max_level+1, number of nodes in each level
    int    *level_nodes;            // Size (max_level+1) * n_leaf_node, indices of nodes on each level
    int    *height_n_node;          // Size max_level+1, number of nodes of each height
    int    *height_nodes;           // Size (max_level+1) * n_leaf_node, indices of nodes of each height
    int    *r_inadm_pairs;          // Size unknown, reduced inadmissible pairs 
    int    *r_adm_pairs;            // Size unknown, reduced admissible pairs 
    int    *HSS_r_inadm_pairs;      // Size unknown, reduced inadmissible pairs in HSS mode
    int    *HSS_r_adm_pairs;        // Size unknown, reduced admissible pairs in HSS mode
    int    *node_inadm_lists;       // Size n_node * max_neighbor, lists of each node's inadmissible nodes
    int    *node_n_r_inadm;         // Size n_node, numbers of each node's reduced inadmissible nodes
    int    *node_n_r_adm;           // Size n_node, numbers of each node's reduced admissible nodes
    int    *coord_idx;              // Size n_point, original index of each sorted point
    int    *fwd_pmt_idx;            // Size krnl_mat_size, multiplicand vector/matrix forward permutation indices 
    int    *bwd_pmt_idx;            // Size krnl_mat_size, output       vector/matrix forward permutation indices
    int    *B_p2i_rowptr;           // Size n_node+1, row_ptr array of the CSR matrix for mapping B{i, j} to a B block index
    int    *B_p2i_colidx;           // Size n_B, col_idx array of the CSR matrix for mapping B{i, j} to a B block index
    int    *B_p2i_val;              // Size n_B, val array of the CSR matrix for mapping B{i, j} to a B block index
    int    *D_p2i_rowptr;           // Size n_node+1, row_ptr array of the CSR matrix for mapping D{i, j} to a D block index
    int    *D_p2i_colidx;           // Size n_D, col_idx array of the CSR matrix for mapping D{i, j} to a D block index
    int    *D_p2i_val;              // Size n_D, val array of the CSR matrix for mapping D{i, j} to a D block index
    int    *ULV_Ls;                 // Size n_node, splitting point of each ULV_L[i], (1 : ULV_Ls[i], 1 : ULV_Ls[i]) are I
    int    *B_nrow;                 // Size n_B, numbers of rows of generator matrices
    int    *B_ncol;                 // Size n_B, numbers of columns of generator matrices
    int    *D_nrow;                 // Size n_D, numbers of rows of dense blocks in the original matrix
    int    *D_ncol;                 // Size n_D, numbers of columns of dense blocks in the original matrix
    size_t *B_ptr;                  // Size n_B, offset of each generator matrix's data in B_data
    size_t *D_ptr;                  // Size n_D, offset of each dense block's data in D_data
    void   *krnl_param;             // Pointer to kernel function parameter array
    void   *pkrnl_param;            // Pointer to periodic system kernel function parameter array
    DTYPE  max_leaf_size;           // Maximum size of a leaf node's box
    DTYPE  QR_stop_tol;             // Partial QR stop column norm tolerance
    DTYPE  HSS_logdet;              // log(abs(det(H2/HSS representation of the kernel matrix)))
    DTYPE  *coord;                  // Size n_point * xpt_dim, sorted point coordinates
    DTYPE  *coord0;                 // Size n_point * xpt_dim, original (not sorted) point coordinates
    DTYPE  *enbox;                  // Size n_node * (2*pt_dim), enclosing box data of each node
    DTYPE  *root_enbox;             // Size 2 * pt_dim, enclosing box of the root node
    DTYPE  *per_lattices;           // Size n_lattice     * pt_dim, for periodic system, each row is a periodic lattice
    DTYPE  *per_adm_shifts;         // Size r_adm_pairs   * pt_dim, for periodic system, each row is a j node's shift in a admissible pair (i, j)
    DTYPE  *per_inadm_shifts;       // Size r_inadm_pairs * pt_dim, for periodic system, each row is a j node's shift in a inadmissible pair (i, j)
    DTYPE  *B_data;                 // Size unknown, data of generator matrices
    DTYPE  *D_data;                 // Size unknown, data of dense blocks in the original matrix
    DTYPE  *per_blk;                // Size unknown, periodic system matvec periodic block 
    DTYPE  *xT;                     // Size krnl_mat_size, for transposing matvec input  "matrix" when krnl_dim > 1
    DTYPE  *yT;                     // Size krnl_mat_size, for transposing matvec output "matrix" when krnl_dim > 1
    DTYPE  *pmt_x;                  // Size krnl_mat_size( * mm_max_n_vec), storing the permuted input vector/matrix (the input need to be permuted)
    DTYPE  *pmt_y;                  // Size krnl_mat_size( * mm_max_n_vec), storing the permuted output vector/matrix (the final output need to be revered)
    H2P_int_vec_p     B_blk;        // Size BD_NTASK_THREAD * n_thread, B matrices task partitioning
    H2P_int_vec_p     D_blk0;       // Size BD_NTASK_THREAD * n_thread, diagonal blocks in D matrices task partitioning
    H2P_int_vec_p     D_blk1;       // Size BD_NTASK_THREAD * n_thread, inadmissible blocks in D matrices task partitioning
    H2P_int_vec_p     *J;           // Size n_node, skeleton row sets
    H2P_int_vec_p     *ULV_idx;     // Size n_node, indices of the sub-matrix which ULV_Q and ULV_L performs on for each node in global sense
    H2P_int_vec_p     *ULV_p;       // Size n_node, HSS ULV LU pivot indices of each node
    H2P_dense_mat_p   *J_coord;     // Size n_node, coordinate of J points
    H2P_dense_mat_p   *pp;          // Size max_level+1, proxy points on each level for generating U and J
    H2P_dense_mat_p   *U;           // Size n_node, Projection matrices
    H2P_dense_mat_p   *y0;          // Size n_node, temporary arrays used in matvec
    H2P_dense_mat_p   *y1;          // Size n_node, temporary arrays used in matvec
    H2P_dense_mat_p   *ULV_Q;       // Size n_node, HSS ULV factorization orthogonal matrix w.r.t. each node's basis
    H2P_dense_mat_p   *ULV_L;       // Size n_node, HSS ULV factorization Cholesky / LU factor w.r.t. each node's diagonal block
    H2P_thread_buf_p  *tb;          // Size n_thread, thread-local buffer
    kernel_eval_fptr  krnl_eval;    // Pointer to kernel matrix evaluation function
    kernel_eval_fptr  pkrnl_eval;   // Pointer to periodic system kernel matrix evaluation function
    kernel_mv_fptr    krnl_mv;      // Pointer to kernel matrix matvec function, only used in periodic system
    kernel_bimv_fptr  krnl_bimv;    // Pointer to kernel matrix bi-matvec function
    DAG_task_queue_p  upward_tq;    // Upward sweep DAG task queue

    // Statistic data
    int    n_matvec;                // Number of performed matvec
    int    n_ULV_solve;             // Number of performed ULV solve
    size_t mat_size[11];            // See below macros
    double timers[11];              // See below macros
    double JIT_flops[2];            // See below macros
};
typedef struct H2Pack  H2Pack_s;
typedef struct H2Pack* H2Pack_p;

// For H2Pack_t->mat_size
typedef enum 
{
    U_SIZE_IDX = 0,     // Total size of U matrices
    B_SIZE_IDX,         // Total size of B matrices
    D_SIZE_IDX,         // Total size of D matrices
    MV_FWD_SIZE_IDX,    // Total memory footprint of H2 matvec forward transformation
    MV_MID_SIZE_IDX,    // Total memory footprint of H2 matvec intermediate multiplication
    MV_BWD_SIZE_IDX,    // Total memory footprint of H2 matvec backward transformation
    MV_DEN_SIZE_IDX,    // Total memory footprint of H2 matvec dense multiplication
    MV_VOP_SIZE_IDX,    // Total memory footprint of H2 matvec OpenMP vector operations
    ULV_Q_SIZE_IDX,     // Total size of ULV Q matrices
    ULV_L_SIZE_IDX,     // Total size of ULV L matrices
    ULV_I_SIZE_IDX      // Total size of ULV integer arrays
} size_idx_t;

// For H2Pack_t->timers
typedef enum
{
    PT_TIMER_IDX = 0,   // Hierarchical partitioning
    U_BUILD_TIMER_IDX,  // U matrices construction
    B_BUILD_TIMER_IDX,  // B matrices construction
    D_BUILD_TIMER_IDX,  // D matrices construction
    MV_FWD_TIMER_IDX,   // H2 matvec forward transformation
    MV_MID_TIMER_IDX,   // H2 matvec intermediate multiplication
    MV_BWD_TIMER_IDX,   // H2 matvec backward transformation
    MV_DEN_TIMER_IDX,   // H2 matvec dense multiplication
    MV_VOP_TIMER_IDX,   // H2 matvec OpenMP vector operations
    ULV_FCT_TIMER_IDX,  // ULV factorization
    ULV_SLV_TIMER_IDX   // ULV solve
} timer_idx_t;

// For H2Pack_t->JIT_flops
typedef enum
{
    JIT_B_FLOPS_IDX = 0,    // JIT H2 matvec intermediate multiplication FLOPS
    JIT_D_FLOPS_IDX         // JIT H2 matvec dense multiplication FLOPS
} jit_flops_idx_t;

// Initialize an H2Pack structure
// Input parameters:
//   pt_dim        : Dimension of point coordinate
//   krnl_dim      : Dimension of tensor kernel's return
//   QR_stop_rank  : Partial QR stop criteria: QR_RANK, QR_REL_NRM, or QR_ABS_NRM
//   QR_stop_param : Pointer to partial QR stop parameter
// Output parameter:
//   h2pack_ : Initialized H2Pack structure
void H2P_init(
    H2Pack_p *h2pack_, const int pt_dim, const int krnl_dim, 
    const int QR_stop_type, void *QR_stop_param
);

// Run H2Pack in HSS mode (by default, H2Pack runs in H2 mode), conflict with 
// H2P_run_RPY_Ewald(). This function should be called after H2P_init() and before 
// H2P_partition_points().
// Input & output parameter:
//   h2pack : H2Pack structure to be configured (h2pack->is_HSS = 1)
void H2P_run_HSS(H2Pack_p h2pack);

// Run the RPY kernel in H2Pack, conflict with H2P_run_RPY_Ewald(). This function 
// should be called after H2P_init() and before H2P_partition_points().
// Input & output parameter:
//   h2pack : H2Pack structure to be configured (h2pack->is_RPY = 1)
void H2P_run_RPY(H2Pack_p h2pack);

// Run the RPY Ewald summation kernel in H2Pack, conflict with H2P_run_HSS()
// and H2P_run_RPY(). This function should be called after H2P_init() and before 
// H2P_partition_points().
// Input & output parameter:
//   h2pack : H2Pack structure to be configured (h2pack->is_RPY_Ewald = 1)
void H2P_run_RPY_Ewald(H2Pack_p h2pack);

// Destroy an H2Pack structure
// Input parameter:
//   *h2pack : H2Pack structure to be destroyed
void H2P_destroy(H2Pack_p *h2pack_);

// Print statistical info of an H2Pack structure
// Input parameter:
//   h2pack : H2Pack structure whose statistical info to be printed
void H2P_print_statistic(H2Pack_p h2pack);

// Reset timing statistical info of an H2Pack structure
// Input parameter:
//   h2pack : H2Pack structure whose timing statistical info to be reset
void H2P_reset_timers(H2Pack_p h2pack);

#ifdef __cplusplus
}
#endif

#endif
