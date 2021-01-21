#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#ifdef __linux__
#include <malloc.h>
#endif

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_utils.h"
#include "H2Pack_SPDHSS_H2.h"
#include "utils.h"

// Build explicit U matrices from nested U matrices
// Input parameter:
//   h2pack : H2Pack data structure after H2P_build()
// Output parameter:
//   exU : Size h2pack->n_node, explicit U matrices for each node
void H2P_build_explicit_U(H2Pack_p h2pack, H2P_dense_mat_p **exU_)
{
    int n_node        = h2pack->n_node;
    int n_leaf_node   = h2pack->n_leaf_node;
    int n_thread      = h2pack->n_thread;
    int max_level     = h2pack->max_level;
    int max_child     = h2pack->max_child;
    int *n_child      = h2pack->n_child;
    int *children     = h2pack->children;
    int *level_nodes  = h2pack->level_nodes;
    int *level_n_node = h2pack->level_n_node;
    int *mat_cluster  = h2pack->mat_cluster;
    H2P_dense_mat_p  *U = h2pack->U;
    H2P_thread_buf_p *thread_buf = h2pack->tb;

    H2P_dense_mat_p *exU = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    ASSERT_PRINTF(exU != NULL, "Failed to allocate %d explicit U matrices\n", n_node);
    memset(exU, 0, sizeof(H2P_dense_mat_p) * n_node);

    for (int i = max_level; i >= 1; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);

        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();
            H2P_int_vec_p idx = thread_buf[tid]->idx0;
            H2P_int_vec_set_capacity(idx, max_child);

            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                H2P_dense_mat_p U_node = U[node];

                int s_row = mat_cluster[2 * node];
                int n_row = mat_cluster[2 * node + 1] - s_row + 1;

                if (U_node->nrow == 0) continue;
                if (n_child_node == 0)
                {
                    H2P_dense_mat_init(&exU[node], U_node->nrow, U_node->ncol);
                    ASSERT_PRINTF(n_row == exU[node]->nrow, "Node %d exU got %d rows, expected %d rows\n", node, exU[node]->nrow, n_row);
                    copy_matrix_block(sizeof(DTYPE), U_node->nrow, U_node->ncol, U_node->data, U_node->ld, exU[node]->data, exU[node]->ld);
                } else {
                    // Multiple each child nodes' exU with U{node} directly
                    int *node_children = children + node * max_child;
                    int bd_U_nrow = 0, bd_U_ncol = 0;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = node_children[k];
                        bd_U_nrow += exU[child_k]->nrow;
                        bd_U_ncol += exU[child_k]->ncol;
                    }
                    ASSERT_PRINTF(n_row == bd_U_nrow, "Node %d exU got %d rows, expected %d rows\n", node, bd_U_nrow, n_row);
                    ASSERT_PRINTF(U_node->nrow == bd_U_ncol, "Node %d U has %d rows, but diagblk(exU) has %d columns\n", node, U_node->nrow, bd_U_ncol);
                    H2P_dense_mat_init(&exU[node], bd_U_nrow, U_node->ncol);
                    bd_U_nrow = 0;
                    bd_U_ncol = 0;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = node_children[k];
                        H2P_dense_mat_p exU_k = exU[child_k];
                        DTYPE *exU_node_k_row = exU[node]->data + bd_U_nrow * exU[node]->ld;
                        DTYPE *U_node_k_col   = U_node->data    + bd_U_ncol * U_node->ld;
                        CBLAS_GEMM(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans, exU_k->nrow, U_node->ncol, exU_k->ncol,
                            1.0, exU_k->data, exU_k->ld, U_node_k_col, U_node->ld, 0.0, exU_node_k_row, exU[node]->ld
                        );
                        bd_U_nrow += exU_k->nrow;
                        bd_U_ncol += exU_k->ncol;
                    }
                }  // End of "if (n_child_node == 0)"
            }  // End of j loop
        }  // End of "#pragma omp parallel"
    }  // End of i loop

    for (int i = 0; i < n_node; i++)
    {
        if (exU[i] != NULL) continue;
        H2P_dense_mat_init(&exU[i], 8, 1);
        exU[i]->nrow = 0;
        exU[i]->ncol = 0;
        exU[i]->ld   = 0;
    }

    *exU_ = exU;
}

// Compute the level of two node's lowest common ancestor
// Input parameters:
//   parent     : Size n_node, parent of each node
//   node_level : Size n_node, level of each node
//   n_level    : Total number of levels (max_level+1 since root is level 0)
//   node{0, 1} : Target node pair
//   work       : Work buffer, size >= 2 * n_level
int H2P_tree_common_ancestor_level(
    const int *parent, const int *node_level, const int n_level, 
    const int node0, const int node1, int *work
)
{
    int *path0 = work;
    int *path1 = work + n_level;
    memset(path0, 0, sizeof(int) * n_level);
    memset(path1, 0, sizeof(int) * n_level);
    int p0 = node0, p1 = node1;
    int level0 = node_level[node0], level1 = node_level[node1];
    for (int i = level0; i >= 0; i--)
    {
        path0[i] = p0;
        p0 = parent[p0];
    }
    for (int i = level1; i >= 0; i--)
    {
        path1[i] = p1;
        p1 = parent[p1];
    }
    int min_level_01 = level0 < level1 ? level0 : level1;
    int level = 0;
    for (int i = 0; i <= min_level_01; i++)
    {
        if (path0[i] != path1[i])
        {
            level = i - 1;
            break;
        }
    }
    return level;
}

// Accumulate partial H2 matvec results for H2P_SPDHSS_H2_build()
// Input parameters:
//   h2mat : Source H2 matrix structure
//   n_vec : Use n_vec Gaussian random vectors
// Output parameter:
//   *Yk_ : Matrix, size h2mat->n_node * h2mat->max_level, each non-empty element is 
//          a matrix of n_vec columns
void H2P_SPDHSS_H2_acc_matvec(H2Pack_p h2mat, const int n_vec, H2P_dense_mat_p **Yk_)
{
    int n_node          = h2mat->n_node;
    int n_leaf_node     = h2mat->n_leaf_node;
    int n_thread        = h2mat->n_thread;
    int max_level       = h2mat->max_level;
    int max_child       = h2mat->max_child;
    int min_adm_level   = h2mat->min_adm_level;
    int *parent         = h2mat->parent;
    int *n_child        = h2mat->n_child;
    int *children       = h2mat->children;
    int *level_nodes    = h2mat->level_nodes;
    int *level_n_node   = h2mat->level_n_node;
    int *leaf_nodes     = h2mat->height_nodes;
    int *node_level     = h2mat->node_level;
    int *mat_cluster    = h2mat->mat_cluster;
    int *B_p2i_rowptr   = h2mat->B_p2i_rowptr;
    int *B_p2i_colidx   = h2mat->B_p2i_colidx;
    int *D_p2i_rowptr   = h2mat->D_p2i_rowptr;
    int *D_p2i_colidx   = h2mat->D_p2i_colidx;
    H2P_dense_mat_p  *U = h2mat->U;
    H2P_thread_buf_p *thread_buf = h2mat->tb;

    // 1. Build explicit U matrix for each node
    H2P_dense_mat_p *exU;
    H2P_build_explicit_U(h2mat, &exU);

    // 2. Prepare the Gaussian random matrix vec and Yk_mat
    // Yk_mat(:, n_vec*(i-1) + 1:n_vec) stores the matvec results for nodes at the i-th level
    const int kms = h2mat->krnl_mat_size;
    const int Yk_mat_ld = n_vec * max_level;
    size_t vec_msize = sizeof(DTYPE) * (size_t) kms * (size_t) n_vec;
    DTYPE *vec    = (DTYPE*) malloc(vec_msize);
    DTYPE *Yk_mat = (DTYPE*) malloc(vec_msize * max_level);  // Note: we have max_level+1 levels in total
    ASSERT_PRINTF(vec != NULL && Yk_mat != NULL, "Failed to allocate space for accu_matvec\n");
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        int s_row, n_row;
        calc_block_spos_len(kms, n_thread, tid, &s_row, &n_row);
        H2P_gen_normal_distribution(0.0, 1.0, (size_t) n_row * (size_t) n_vec, vec + (size_t) s_row * (size_t) n_vec);
        size_t Yk_mat_offset = (size_t) s_row * (size_t) n_vec * (size_t) max_level;
        size_t Yk_mat_nelem  = (size_t) n_row * (size_t) n_vec * (size_t) max_level;
        memset(Yk_mat + Yk_mat_offset, 0, sizeof(DTYPE) * Yk_mat_nelem);
    }

    // 3. H2 matvec upward sweep
    H2P_dense_mat_p *y0 = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    ASSERT_PRINTF(y0 != NULL, "Failed to allocate %d working matrices\n", n_node);
    for (int i = 0; i < n_node; i++) y0[i] = NULL;
    for (int i = max_level; i >= min_adm_level; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);

        #pragma omp parallel num_threads(n_thread_i)
        {
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                H2P_dense_mat_p U_node = U[node];

                H2P_dense_mat_init(&y0[node], U_node->ncol, n_vec);
                if (n_child_node == 0)
                {
                    // Leaf node, directly multiply x_j with U_j^T
                    int s_row = mat_cluster[2 * node];
                    int e_row = mat_cluster[2 * node + 1];
                    int nrow = e_row - s_row + 1;
                    DTYPE *vec_blk = vec + (size_t)s_row * (size_t)n_vec;
                    CBLAS_GEMM(
                        CblasRowMajor, CblasTrans, CblasNoTrans, U_node->ncol, n_vec, nrow,
                        1.0, U_node->data, U_node->ld, vec_blk, n_vec, 0.0, y0[node]->data, y0[node]->ld
                    );
                } else {
                    // Non-leaf node, concatenate y0 in the children nodes and multiply it with U_j^T
                    // Multiple U{node} with each child nodes' y0 directly
                    int *node_children = children + node * max_child;
                    int y0_tmp_nrow = 0;
                    DTYPE beta = 0.0;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = node_children[k];
                        H2P_dense_mat_p y0_k = y0[child_k];
                        DTYPE *U_node_k_row = U_node->data + y0_tmp_nrow * U_node->ld;
                        CBLAS_GEMM(
                            CblasRowMajor, CblasTrans, CblasNoTrans, U_node->ncol, n_vec, y0_k->nrow,
                            1.0, U_node_k_row, U_node->ld, y0_k->data, y0_k->ld, beta, y0[node]->data, y0[node]->ld
                        );
                        beta = 1.0;
                        y0_tmp_nrow += y0[child_k]->nrow;
                    }  // End of k loop
                }  // End of "if (n_child_node == 0)"
            }  // End of j loop
        }  // End of "#pragma omp parallel"
    }  // End of i loop

    // 4. For each pair of siblings (i, j), compute
    //    Yk{i} += Aij  * vec_j
    //    Yk{j} += Aij' * vec_i
    // Yk{i/j} are stored in the corresponding columns of the actual Yk
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_int_vec_p   work = thread_buf[tid]->idx0;
        H2P_dense_mat_p Dij  = thread_buf[tid]->mat0;
        H2P_dense_mat_p Bij  = thread_buf[tid]->mat0;
        H2P_dense_mat_p tmpM = thread_buf[tid]->mat1;
        H2P_int_vec_set_capacity(work, 2 * max_level + 4);

        // 4.1 Inadmissible pairs
        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int s_row0 = mat_cluster[2 * node0];
            int e_row0 = mat_cluster[2 * node0 + 1];
            int n_row0 = e_row0 - s_row0 + 1;
            for (int i = D_p2i_rowptr[node0]; i < D_p2i_rowptr[node0 + 1]; i++)
            {
                int node1 = D_p2i_colidx[i];
                if (node0 == node1) continue;
                int ca_level = H2P_tree_common_ancestor_level(parent, node_level, max_level+1, node0, node1, work->data) + 1;
                int s_col  = n_vec * (ca_level - 1);
                int s_row1 = mat_cluster[2 * node1];
                int e_row1 = mat_cluster[2 * node1 + 1];
                int n_row1 = e_row1 - s_row1 + 1;
                size_t Yk_mat_offset = (size_t) s_row0 * (size_t) Yk_mat_ld + (size_t) s_col;
                DTYPE *Yk_mat_blk0 = Yk_mat + Yk_mat_offset;
                DTYPE *vec_blk1    = vec + (size_t)s_row1 * (size_t)n_vec;

                int Dij_nrow, Dij_ncol, Dij_ld, Dij_trans;
                H2P_get_Dij_block(h2mat, node0, node1, Dij);
                DTYPE *Dij_data = Dij->data;
                if (Dij->ld > 0)
                {
                    Dij_nrow  = Dij->nrow;
                    Dij_ncol  = Dij->ncol;
                    Dij_ld    = Dij->ld;
                    Dij_trans = 0;
                } else {
                    Dij_nrow  = Dij->ncol;
                    Dij_ncol  = Dij->nrow;
                    Dij_ld    = -Dij->ld;
                    Dij_trans = 1;
                }

                // We only handle: 
                //     Yk_mat(idx1, col_idx) = Yk_mat(idx1, col_idx) + D{D_idx}  * vec(idx2, :);
                // it's symmetric operation:
                //     Yk_mat(idx2, col_idx) = Yk_mat(idx2, col_idx) + D{D_idx}' * vec(idx1, :);
                // is handled by double counting node inadmissible pairs
                ASSERT_PRINTF(
                    Dij_nrow == n_row0 && Dij_ncol == n_row1,
                    "D{%d, %d} has size %d * %d, expected %d * %d\n", 
                    node0, node1, Dij_nrow, Dij_ncol, n_row0, n_row1
                );
                CBLAS_TRANSPOSE Dij_trans_ = (Dij_trans == 0) ? CblasNoTrans : CblasTrans;
                CBLAS_GEMM(
                    CblasRowMajor, Dij_trans_, CblasNoTrans, n_row0, n_vec, n_row1,
                    1.0, Dij_data, Dij_ld, vec_blk1, n_vec, 1.0, Yk_mat_blk0, Yk_mat_ld
                );
            }  // End of i loop
        }  // End of node0 loop
        
        // 4.2 Admissible pairs
        #pragma omp barrier
        for (int i = max_level; i >= min_adm_level; i--)
        {
            int *level_i_nodes = level_nodes + i * n_leaf_node;
            int level_i_n_node = level_n_node[i];

            #pragma omp barrier
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node0 = level_i_nodes[j];
                int s_row0 = mat_cluster[2 * node0];
                int e_row0 = mat_cluster[2 * node0 + 1];
                int n_row0 = e_row0 - s_row0 + 1;
                int level0 = node_level[node0];
                H2P_dense_mat_p exU_0 = exU[node0];
                for (int k = B_p2i_rowptr[node0]; k < B_p2i_rowptr[node0 + 1]; k++)
                {
                    int node1 = B_p2i_colidx[k];
                    if (node0 == node1) continue;
                    int ca_level = H2P_tree_common_ancestor_level(parent, node_level, max_level+1, node0, node1, work->data) + 1;
                    int s_col  = n_vec * (ca_level - 1);
                    int s_row1 = mat_cluster[2 * node1];
                    int e_row1 = mat_cluster[2 * node1 + 1];
                    int n_row1 = e_row1 - s_row1 + 1;
                    int level1 = node_level[node1];

                    int Bij_nrow, Bij_ncol, Bij_ld, Bij_trans;
                    H2P_get_Bij_block(h2mat, node0, node1, Bij);
                    DTYPE *Bij_data = Bij->data;
                    if (Bij->ld > 0)
                    {
                        Bij_nrow  = Bij->nrow;
                        Bij_ncol  = Bij->ncol;
                        Bij_ld    = Bij->ld;
                        Bij_trans = 0;
                    } else {
                        Bij_nrow  = Bij->ncol;
                        Bij_ncol  = Bij->nrow;
                        Bij_ld    = -Bij->ld;
                        Bij_trans = 1;
                    }
                    H2P_dense_mat_p y0_1 = y0[node1];
                    size_t Yk_mat_offset = (size_t) s_row0 * (size_t) Yk_mat_ld + (size_t) s_col;
                    DTYPE *Yk_mat_blk0 = Yk_mat + Yk_mat_offset;
                    DTYPE *vec_blk1    = vec + (size_t)s_row1 * (size_t) n_vec;

                    // We only handle the update on Yk_mat_blk0, the symmetric operation for 
                    // updating Yk_mat_blk1 is handled by double counting the admissible pairs

                    // A. Two nodes are of the same level, compress on both side
                    if (level0 == level1)
                    {
                        // Yk_mat(idx1, col_idx) = Yk_mat(idx1, col_idx) + exU{c1} * (Bij  * y0{c2});
                        ASSERT_PRINTF(
                            exU_0->ncol == Bij_nrow && Bij_ncol == y0_1->nrow,
                            "Pair (%d, %d) GEMM size mismatch: [%d, %d] * [%d, %d] * [%d, %d]\n",
                            node0, node1, exU_0->nrow, exU_0->ncol, Bij_nrow, Bij_ncol, y0_1->nrow, y0_1->ncol
                        );
                        ASSERT_PRINTF(
                            n_row0 == exU_0->nrow && n_vec == y0_1->ncol, 
                            "Pair (%d, %d) matrix addition size mismatch: expected [%d, %d], got [%d, %d]\n",
                            node0, node1, n_row0, n_vec, exU_0->nrow, y0_1->ncol
                        );
                        H2P_dense_mat_resize(tmpM, Bij_nrow, y0_1->ncol);
                        CBLAS_TRANSPOSE Bij_trans_ = (Bij_trans == 0) ? CblasNoTrans : CblasTrans;
                        CBLAS_GEMM(
                            CblasRowMajor, Bij_trans_, CblasNoTrans, Bij_nrow, y0_1->ncol, Bij_ncol,
                            1.0, Bij_data, Bij_ld, y0_1->data, y0_1->ld, 0.0, tmpM->data, tmpM->ld
                        );
                        CBLAS_GEMM(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans, exU_0->nrow, tmpM->ncol, exU_0->ncol,
                            1.0, exU_0->data, exU_0->ld, tmpM->data, tmpM->ld, 1.0, Yk_mat_blk0, Yk_mat_ld
                        );
                    }  // End of "if (level0 == level1)"

                    // B. node1 is a leaf node and its level is larger than node0,
                    //    only compress on node0's side
                    if (level0 > level1)
                    {
                        // Yk_mat(idx1, col_idx) = Yk_mat(idx1, col_idx) + exU{c1} * (Bij * vec(idx2, :));
                        ASSERT_PRINTF(
                            exU_0->ncol == Bij_nrow && Bij_ncol == n_row1,
                            "Pair (%d, %d) GEMM size mismatch: [%d, %d] * [%d, %d] * [%d, %d]\n",
                            node0, node1, exU_0->nrow, exU_0->ncol, Bij_nrow, Bij_ncol, n_row1, n_vec
                        );
                        ASSERT_PRINTF(
                            n_row0 == exU_0->nrow, 
                            "Pair (%d, %d) matrix addition size mismatch: expected [%d, %d], got [%d, %d]\n",
                            node0, node1, n_row0, n_vec, exU_0->nrow, n_vec
                        );
                        H2P_dense_mat_resize(tmpM, Bij_nrow, n_vec);
                        CBLAS_TRANSPOSE Bij_trans_ = (Bij_trans == 0) ? CblasNoTrans : CblasTrans;
                        CBLAS_GEMM(
                            CblasRowMajor, Bij_trans_, CblasNoTrans, Bij_nrow, n_vec, Bij_ncol,
                            1.0, Bij_data, Bij_ld, vec_blk1, n_vec, 0.0, tmpM->data, tmpM->ld
                        );
                        CBLAS_GEMM(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans, exU_0->nrow, tmpM->ncol, exU_0->ncol,
                            1.0, exU_0->data, exU_0->ld, tmpM->data, tmpM->ld, 1.0, Yk_mat_blk0, Yk_mat_ld
                        );
                    }  // End of "if (level0 > level1)"

                    // C. node0 is a leaf node and its level is larger than node1,
                    //    only compress on node1's side
                    if (level0 < level1)
                    {
                        // Yk_mat(idx1, col_idx) = Yk_mat(idx1, col_idx) + Bij * y0{c2};
                        ASSERT_PRINTF(
                            n_row0 == Bij_nrow && Bij_ncol == y0_1->nrow && y0_1->ncol == n_vec,
                            "Pair (%d, %d) GEMM & matrix addition size mismatch: [%d, %d] + [%d, %d] * [%d, %d]\n",
                            node0, node1, n_row0, n_vec, Bij_nrow, Bij_ncol, y0_1->nrow, y0_1->ncol
                        );
                        CBLAS_TRANSPOSE Bij_trans_ = (Bij_trans == 0) ? CblasNoTrans : CblasTrans;
                        CBLAS_GEMM(
                            CblasRowMajor, Bij_trans_, CblasNoTrans, Bij_nrow, y0_1->ncol, Bij_ncol, 
                            1.0, Bij_data, Bij_ld, y0_1->data, y0_1->ld, 1.0, Yk_mat_blk0, Yk_mat_ld
                        );
                    }  // End of "if (level0 < level1)"
                }  // End of k loop (admissible pairs)
            }  // End of j loop (nodes in this level)
        }  // End of i loop (level)
    }  // End of "#pragma omp parallel"

    // 5. Accumulate the results in Yk_mat to lead nodes
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        int s_row, n_row;
        calc_block_spos_len(kms, n_thread, tid, &s_row, &n_row);
        for (int level = 1; level < max_level; level++)
        {
            int s_col = n_vec * (level - 1);
            for (int i = s_row; i < s_row + n_row; i++)
            {
                size_t Yk_mat_offset = (size_t) i * (size_t) Yk_mat_ld;
                DTYPE *Yk_mat_i = Yk_mat + Yk_mat_offset;
                #pragma omp simd
                for (int j = s_col; j < s_col + n_vec; j++)
                    Yk_mat_i[j + n_vec] += Yk_mat_i[j];
            }
        }
    }  // End of "#pragma omp parallel"

    // 6. Repack Yk_mat into Yk
    H2P_dense_mat_p *Yk = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node * max_level);
    ASSERT_PRINTF(Yk != NULL, "Failed to allocate %d * %d Yk matrices\n", n_node, max_level);
    for (int i = 0; i < n_node * max_level; i++) Yk[i] = NULL;
    #pragma omp parallel num_threads(n_thread)
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_leaf_node; i++)
        {
            int node  = leaf_nodes[i];
            int level = node_level[node];
            int s_row = mat_cluster[2 * node];
            int e_row = mat_cluster[2 * node + 1];
            int n_row = e_row - s_row + 1;
            for (int j = level - 1; j >= 0; j--)
            {
                int s_col = j * n_vec;
                size_t Yk_mat_offset = (size_t) s_row * (size_t) Yk_mat_ld + (size_t) s_col;
                DTYPE *Yk_mat_blk = Yk_mat + Yk_mat_offset;
                int Yk_idx = node * max_level + (level - 1 - j);
                H2P_dense_mat_init(&Yk[Yk_idx], n_row, n_vec);
                ASSERT_PRINTF(Yk[Yk_idx] != NULL, "Failed to allocate %d * %d Yk[idx] matrices\n", n_row, n_vec);
                H2P_dense_mat_p Yk_ij = Yk[Yk_idx];
                copy_matrix_block(sizeof(DTYPE), n_row, n_vec, Yk_mat_blk, Yk_mat_ld, Yk_ij->data, Yk_ij->ld);
            }
        }  // End of i loop
    }  // End of "#pragma omp parallel"
    for (int i = 0; i < n_node * max_level; i++) 
    {
        if (Yk[i] != NULL) continue;
        H2P_dense_mat_init(&Yk[i], 8, 8);
        Yk[i]->nrow = 0;
        Yk[i]->ncol = 0;
        Yk[i]->ld   = 0;
    }

    // 7. Free intermediate arrays 
    for (int i = 0; i < n_node; i++)
    {
        H2P_dense_mat_destroy(&exU[i]);
        H2P_dense_mat_destroy(&y0[i]);
    }
    free(exU);
    free(y0);
    free(vec);
    free(Yk_mat);
    *Yk_ = Yk;
}

// Gather matrices in HSS_B into a large matrix tmpB s.t. the i-th row j-th column 
// block of tmpB is HSS_B[HSS_B_pair2idx(blk0[i], blk1[j])]
// Input parameters:
//   n_blk{0, 1}    : Number of row & column blocks
//   blk{0, 1}      : Node indices of row & column blocks
//   n_node         : Number of nodes
//   HSS_B_p2i_{*}  : CSR matrix array triple, convert (i, j) pair to an index for HSS_B
//   HSS_B          : Source HSS_B matrices
// Output parameter:
//   tmpB : Result matrix
void H2P_SPDHSS_H2_gather_HSS_B(
    const int n_blk0, const int n_blk1, const int *blk0, const int *blk1, const int n_node, 
    const int *HSS_B_p2i_rowptr, const int *HSS_B_p2i_colidx, const int *HSS_B_p2i_val,
    H2P_dense_mat_p *HSS_B, H2P_dense_mat_p tmpB
)
{
    int nrow = 0, ncol = 0;
    int B_idx_00 = H2P_get_int_CSR_elem(HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, blk0[0], blk1[0]);
    ASSERT_PRINTF(B_idx_00 > 0, "SPDHSS_B{%d, %d} does not exist!\n", blk0[0], blk1[0]);
    B_idx_00--;
    int nrow0 = HSS_B[B_idx_00]->nrow;
    int ncol0 = HSS_B[B_idx_00]->ncol;
    
    // Count the total number of rows
    for (int i = 0; i < n_blk0; i++)
    {
        int B_idx_i0 = H2P_get_int_CSR_elem(HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, blk0[i], blk1[0]);
        ASSERT_PRINTF(B_idx_i0 > 0, "SPDHSS_B{%d, %d} does not exist!\n", blk0[i], blk1[0]);
        B_idx_i0--;
        if (HSS_B[B_idx_i0]->ncol != ncol0)
        {
            ERROR_PRINTF(
                "SPDHSS_B{%d, %d} ncol = %d, expected %d\n", 
                blk0[i], blk1[0], HSS_B[B_idx_i0]->ncol, ncol0
            );
            return;
        }
        nrow += HSS_B[B_idx_i0]->nrow;
    }  // End of i loop

    // Count the total number of columns
    for (int j = 0; j < n_blk1; j++)
    {
        int B_idx_0j = H2P_get_int_CSR_elem(HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, blk0[0], blk1[j]);
        ASSERT_PRINTF(B_idx_0j > 0, "SPDHSS_B{%d, %d} does not exist!\n", blk0[0], blk1[j]);
        B_idx_0j--;
        if (HSS_B[B_idx_0j]->nrow != nrow0)
        {
            ERROR_PRINTF(
                "SPDHSS_B{%d, %d} nrow = %d, expected %d\n", 
                blk0[0], blk1[j], HSS_B[B_idx_0j]->nrow, nrow0
            );
            return;
        }
        ncol += HSS_B[B_idx_0j]->ncol;
    }  // End of j loop

    // Gather each block
    H2P_dense_mat_resize(tmpB, nrow, ncol);
    int s_row = 0;
    for (int i = 0; i < n_blk0; i++)
    {
        int s_col = 0, nrow_i = 0, ncol_j;
        for (int j = 0; j < n_blk1; j++)
        {
            int B_idx_ij = H2P_get_int_CSR_elem(HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, blk0[i], blk1[j]);
            ASSERT_PRINTF(B_idx_ij > 0, "SPDHSS_B{%d, %d} does not exist!\n", blk0[i], blk1[j]);
            B_idx_ij--;
            DTYPE *tmpB_ij = tmpB->data + s_row * tmpB->ld + s_col;
            H2P_dense_mat_p HSS_Bij = HSS_B[B_idx_ij];
            nrow_i = HSS_Bij->nrow;
            ncol_j = HSS_Bij->ncol;
            copy_matrix_block(sizeof(DTYPE), nrow_i, ncol_j, HSS_Bij->data, HSS_Bij->ld, tmpB_ij, tmpB->ld);
            s_col += ncol_j;
        }  // End of j loop
        s_row += nrow_i;
    }  // End of i loop
}

// Remove unused HSS_B matrices
// Input parameters:
//   n_blk{0, 1}    : Number of row & column blocks
//   blk{0, 1}      : Node indices of row & column blocks
//   n_node         : Number of nodes
//   HSS_B_p2i_{*}  : CSR matrix array triple, convert (i, j) pair to an index for HSS_B
//   HSS_B          : Source HSS_B matrices
// Output parameter:
//   HSS_B_p2i_{*} : Updated CSR matrix array triple
//   HSS_B         : Updated HSS_B matrices (some unused Bij matrices will be deleted)
void H2P_SPDHSS_H2_clean_HSS_B(
    const int n_blk0, const int n_blk1, const int *blk0, const int *blk1, const int n_node, 
    const int *HSS_B_p2i_rowptr, const int *HSS_B_p2i_colidx, int *HSS_B_p2i_val, H2P_dense_mat_p *HSS_B
)
{
    for (int i = 0; i < n_blk0; i++)
    {
        for (int j = 0; j < n_blk1; j++)
        {
            int B_idx_ij = H2P_get_int_CSR_elem(HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, blk0[i], blk1[j]);
            ASSERT_PRINTF(B_idx_ij > 0, "SPDHSS_B{%d, %d} does not exist!\n", blk0[i], blk1[j]);
            B_idx_ij--;
            H2P_dense_mat_destroy(&HSS_B[B_idx_ij]);
            H2P_set_int_CSR_elem(HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, blk0[i], blk1[j], 0);
        }
    }
}

// Calculate a new HSS Bij matrix for pair (node0, node1)
// Input parameters:
//   h2mat          : Source H2 matrix structure
//   node{0, 1}     : Node pair
//   S, V, W, Minv  : Arrays, size h2mat->n_node, intermediate matrices used in H2P_SPDHSS_H2_build()
//   HSS_B_p2i_{*}  : CSR matrix array triple, convert (i, j) pair to an index for HSS_B
//   HSS_B          : New HSS Bij matrices
// Output parameters:
//   HSS_B_p2i_{*} : Updated CSR matrix array triple
//   HSS_B         : Updated HSS_B matrices (some unused Bij matrices will be deleted)
void H2P_SPDHSS_H2_calc_HSS_Bij(
    H2Pack_p h2mat, const int node0, const int node1, const int tid, 
    H2P_dense_mat_p *S, H2P_dense_mat_p *V, H2P_dense_mat_p *W, H2P_dense_mat_p *Minv, 
    const int *HSS_B_p2i_rowptr, const int *HSS_B_p2i_colidx, int *HSS_B_p2i_val, H2P_dense_mat_p *HSS_B
)
{
    int   pt_dim      = h2mat->pt_dim;
    int   max_child   = h2mat->max_child;
    int   n_node      = h2mat->n_node;
    int   *node_level = h2mat->node_level;
    int   *n_child    = h2mat->n_child;
    int   *children   = h2mat->children;
    DTYPE *enbox      = h2mat->enbox;

    int   level0   = node_level[node0];
    int   level1   = node_level[node1];
    int   n_child0 = n_child[node0];
    int   n_child1 = n_child[node1];
    int   *child0  = children + node0 * max_child;
    int   *child1  = children + node1 * max_child;
    DTYPE *enbox0  = enbox + node0 * 2 * pt_dim;
    DTYPE *enbox1  = enbox + node1 * 2 * pt_dim;
    int   is_adm   = H2P_check_box_admissible(enbox0, enbox1, pt_dim, ALPHA_H2);
    H2P_thread_buf_p thread_buf = h2mat->tb[tid];

    int HSS_B_idx = H2P_get_int_CSR_elem(HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, node0, node1);
    ASSERT_PRINTF(HSS_B_idx > 0, "SPDHSS_B{%d, %d} does not exist!\n", node0, node1);
    HSS_B_idx--;
    H2P_dense_mat_init(&HSS_B[HSS_B_idx], 0, 0);
    H2P_dense_mat_p HSS_Bij = HSS_B[HSS_B_idx];

    if (level0 == level1)
    {
        // 1.1: node0 and node1 are admissible
        if (is_adm)
        {
            H2P_dense_mat_p H2_Bij = thread_buf->mat0;
            H2P_dense_mat_p tmpM   = thread_buf->mat1;
            H2_Bij->nrow = 0;
            H2P_get_Bij_block(h2mat, node0, node1, H2_Bij);
            if (H2_Bij->nrow == 0)
            {
                ERROR_PRINTF("Bug in case 1.1, node pair (%d, %d)\n", node0, node1);
                H2P_dense_mat_reset(H2_Bij);
                H2P_dense_mat_reset(tmpM);
                return;
            }
            H2P_dense_mat_p W0 = W[node0];
            H2P_dense_mat_p W1 = W[node1];
            // Bij = W{node1} * H2_B{H2_B_idx} * W{node2}';
            H2P_dense_mat_resize(tmpM, H2_Bij->nrow, W1->nrow);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasTrans, H2_Bij->nrow, W1->nrow, H2_Bij->ncol,
                1.0, H2_Bij->data, H2_Bij->ld, W1->data, W1->ld, 0.0, tmpM->data, tmpM->ld
            );
            H2P_dense_mat_resize(HSS_Bij, W0->nrow, tmpM->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, W0->nrow, tmpM->ncol, W0->ncol,
                1.0, W0->data, W0->ld, tmpM->data, tmpM->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            return;
        }  // End of "if (is_adm)"

        // Otherwise: node0 and node1 are inadmissible

        // 1.2: Both nodes are leaf nodes
        if (n_child0 == 0 && n_child1 == 0)
        {
            H2P_dense_mat_p H2_Dij = thread_buf->mat0;
            H2P_dense_mat_p tmpM   = thread_buf->mat1;
            H2_Dij->nrow = 0;
            H2P_get_Dij_block(h2mat, node0, node1, H2_Dij);
            if (H2_Dij->nrow == 0)
            {
                ERROR_PRINTF("Bug in case 1.2, node pair (%d, %d)\n", node0, node1);
                H2P_dense_mat_reset(H2_Dij);
                H2P_dense_mat_reset(tmpM);
                return;
            }
            H2P_dense_mat_p S0 = S[node0];
            H2P_dense_mat_p S1 = S[node1];
            H2P_dense_mat_p V0 = V[node0];
            H2P_dense_mat_p V1 = V[node1];
            // tmpM = V{node1}' * linsolve(S{node1}, H2_D{H2_D_idx}, struct('LT', true));
            CBLAS_TRSM(
                CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 
                H2_Dij->nrow, H2_Dij->ncol, 1.0, S0->data, S0->ld, H2_Dij->data, H2_Dij->ld
            );
            H2P_dense_mat_resize(tmpM, V0->ncol, H2_Dij->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, H2_Dij->ncol, V0->nrow,
                1.0, V0->data, V0->ld, H2_Dij->data, H2_Dij->ld, 0.0, tmpM->data, tmpM->ld
            );
            // Bij = linsolve(S{node2}, tmpM', struct('LT', true))' * V{node2};
            // S{node2} * X = tmpM', we need Bij = X' * V{node2}
            // Solve X' * S{node2}' = tmpM to obtain X' directly
            CBLAS_TRSM(
                CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, 
                tmpM->nrow, tmpM->ncol, 1.0, S1->data, S1->ld, tmpM->data, tmpM->ld
            );
            H2P_dense_mat_resize(HSS_Bij, tmpM->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpM->nrow, V1->ncol, tmpM->ncol,
                1.0, tmpM->data, tmpM->ld, V1->data, V1->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            return;
        }  // End of "if (n_child0 == 0 && n_child1 == 0)"

        // 1.3: Both nodes are non-leaf nodes
        if (n_child0 > 0 && n_child1 > 0)
        {
            H2P_dense_mat_p tmpB  = thread_buf->mat0;
            H2P_dense_mat_p tmpM0 = thread_buf->mat1;
            H2P_dense_mat_p tmpM1 = thread_buf->mat2;
            tmpB->nrow = 0;
            H2P_SPDHSS_H2_gather_HSS_B(
                n_child0, n_child1, child0, child1, n_node, 
                HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, HSS_B, tmpB
            );
            if (tmpB->nrow == 0)
            {
                ERROR_PRINTF("Bug in case 1.3, node pair (%d, %d)\n", node0, node1);
                H2P_dense_mat_reset(tmpB);
                H2P_dense_mat_reset(tmpM0);
                H2P_dense_mat_reset(tmpM1);
                return;
            }
            H2P_dense_mat_p V0    = V[node0];
            H2P_dense_mat_p V1    = V[node1];
            H2P_dense_mat_p Minv0 = Minv[node0];
            H2P_dense_mat_p Minv1 = Minv[node1];
            // Bij = V{node1}' * Minv{node1} * tmpB * Minv{node2} * V{node2};
            H2P_dense_mat_resize(tmpM0, Minv1->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv1->nrow, V1->ncol, Minv1->ncol, 
                1.0, Minv1->data, Minv1->ld, V1->data, V1->ld, 0.0, tmpM0->data, tmpM0->ld
            );
            H2P_dense_mat_resize(tmpM1, tmpB->nrow, tmpM0->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpB->nrow, tmpM0->ncol, tmpB->ncol,
                1.0, tmpB->data, tmpB->ld, tmpM0->data, tmpM0->ld, 0.0, tmpM1->data, tmpM1->ld
            );
            H2P_dense_mat_resize(tmpM0, Minv0->nrow, tmpM1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv0->nrow, tmpM1->ncol, Minv0->ncol,
                1.0, Minv0->data, Minv0->ld, tmpM1->data, tmpM1->ld, 0.0, tmpM0->data, tmpM0->ld
            );
            H2P_dense_mat_resize(HSS_Bij, V0->ncol, tmpM0->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, tmpM0->ncol, V0->nrow, 
                1.0, V0->data, V0->ld, tmpM0->data, tmpM0->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_SPDHSS_H2_clean_HSS_B(
                n_child0, n_child1, child0, child1, n_node, 
                HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, HSS_B
            );
            return;
        }  // End of "if (n_child0 > 0 && n_child1 > 0)"

        // 1.4: node0 is non-leaf, node1 is leaf
        if (n_child0 > 0 && n_child1 == 0)
        {
            H2P_dense_mat_p tmpB  = thread_buf->mat0;
            H2P_dense_mat_p tmpM0 = thread_buf->mat1;
            H2P_dense_mat_p tmpM1 = thread_buf->mat2;
            tmpB->nrow = 0;
            H2P_SPDHSS_H2_gather_HSS_B(
                n_child0, 1, child0, &node1, n_node, 
                HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, HSS_B, tmpB
            );
            if (tmpB->nrow == 0)
            {
                ERROR_PRINTF("Bug in case 1.4, node pair (%d, %d)\n", node0, node1);
                H2P_dense_mat_reset(tmpB);
                H2P_dense_mat_reset(tmpM0);
                H2P_dense_mat_reset(tmpM1);
                return;
            }
            H2P_dense_mat_p V0    = V[node0];
            H2P_dense_mat_p V1    = V[node1];
            H2P_dense_mat_p S1    = S[node1];
            H2P_dense_mat_p Minv0 = Minv[node0];
            // tmpM1 = V{node1}' * Minv{node1} * tmpB;
            H2P_dense_mat_resize(tmpM0, Minv0->nrow, tmpB->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv0->nrow, tmpB->ncol, Minv0->ncol,
                1.0, Minv0->data, Minv0->ld, tmpB->data, tmpB->ld, 0.0, tmpM0->data, tmpM0->ld
            );
            H2P_dense_mat_resize(tmpM1, V0->ncol, tmpM0->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, tmpM0->ncol, V0->nrow,
                1.0, V0->data, V0->ld, tmpM0->data, tmpM0->ld, 0.0, tmpM1->data, tmpM1->ld
            );
            // Bij = linsolve(S{node2}, tmpM1', struct('LT', true))' * V{node2};
            // S{node2} * X = tmpM1', we need Bij = X' * V{node2}
            // Solve X' * S{node2}' = tmpM1 to obtain X' directly
            CBLAS_TRSM(
                CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, 
                tmpM1->nrow, tmpM1->ncol, 1.0, S1->data, S1->ld, tmpM1->data, tmpM1->ld
            );
            H2P_dense_mat_resize(HSS_Bij, tmpM1->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpM1->nrow, V1->ncol, tmpM1->ncol,
                1.0, tmpM1->data, tmpM1->ld, V1->data, V1->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_SPDHSS_H2_clean_HSS_B(
                n_child0, 1, child0, &node1, n_node, 
                HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, HSS_B
            );
            return;
        }  // End of "if (n_child0 > 0 && n_child1 == 0)"

        // 1.5: node0 is leaf, node1 is non-leaf
        if (n_child0 == 0 && n_child1 > 0)
        {
            H2P_dense_mat_p tmpB  = thread_buf->mat0;
            H2P_dense_mat_p tmpM0 = thread_buf->mat1;
            H2P_dense_mat_p tmpM1 = thread_buf->mat2;
            tmpB->nrow = 0;
            H2P_SPDHSS_H2_gather_HSS_B(
                1, n_child1, &node0, child1, n_node, 
                HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, HSS_B, tmpB
            );
            if (tmpB->nrow == 0)
            {
                ERROR_PRINTF("Bug in case 1.5, node pair (%d, %d)\n", node0, node1);
                H2P_dense_mat_reset(tmpB);
                H2P_dense_mat_reset(tmpM0);
                H2P_dense_mat_reset(tmpM1);
                return;
            }
            H2P_dense_mat_p V0    = V[node0];
            H2P_dense_mat_p V1    = V[node1];
            H2P_dense_mat_p S0    = S[node0];
            H2P_dense_mat_p Minv1 = Minv[node1];
            // tmpM1 = tmpB * Minv{node2} * V{node2};
            H2P_dense_mat_resize(tmpM0, Minv1->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv1->nrow, V1->ncol, Minv1->ncol,
                1.0, Minv1->data, Minv1->ld, V1->data, V1->ld, 0.0, tmpM0->data, tmpM0->ld
            );
            H2P_dense_mat_resize(tmpM1, tmpB->nrow, tmpM0->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpB->nrow, tmpM0->ncol, tmpB->ncol,
                1.0, tmpB->data, tmpB->ld, tmpM0->data, tmpM0->ld, 0.0, tmpM1->data, tmpM1->ld
            );
            // Bij = V{node1}' * linsolve(S{node1}, tmpM1, struct('LT', true));
            CBLAS_TRSM(
                CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 
                tmpM1->nrow, tmpM1->ncol, 1.0, S0->data, S0->ld, tmpM1->data, tmpM1->ld
            );
            H2P_dense_mat_resize(HSS_Bij, V0->ncol, tmpM1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, tmpM1->ncol, V0->nrow, 
                1.0, V0->data, V0->ld, tmpM1->data, tmpM1->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_SPDHSS_H2_clean_HSS_B(
                1, n_child1, &node0, child1, n_node, 
                HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, HSS_B
            );
            return;
        }  // End of "if (n_child0 == 0 && n_child1 > 0)"
    }  // End of "if (level0 == level1)"

    if (level0 > level1)
    {
        // Note: node1 must be a leaf node
        if (n_child1 > 0)
        {
            ERROR_PRINTF("Bug in case 2, node pair (%d, %d)\n", node0, node1);
            return;
        }  // End of "if (n_child1 > 0)"

        // 2.1: node0 and node1 are admissible
        if (is_adm)
        {
            H2P_dense_mat_p H2_Bij = thread_buf->mat0;
            H2_Bij->nrow = 0;
            H2P_get_Bij_block(h2mat, node0, node1, H2_Bij);
            if (H2_Bij->nrow == 0)
            {
                ERROR_PRINTF("Bug in case 2.1, node pair (%d, %d)\n", node0, node1);
                H2P_dense_mat_reset(H2_Bij);
                return;
            }
            H2P_dense_mat_p W0 = W[node0];
            // Bij = W{node1} * H2_B{H2_B_idx};
            H2P_dense_mat_resize(HSS_Bij, W0->nrow, H2_Bij->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, W0->nrow, H2_Bij->ncol, W0->ncol,
                1.0, W0->data, W0->ld, H2_Bij->data, H2_Bij->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            return;
        }  // End of "if (is_adm)"

        // Otherwise: node0 and node1 are inadmissible

        // 2.2: node0 is a leaf node
        if (n_child0 == 0)
        {
            H2P_dense_mat_p H2_Dij = thread_buf->mat0;
            H2_Dij->nrow = 0;
            H2P_get_Dij_block(h2mat, node0, node1, H2_Dij);
            if (H2_Dij->nrow == 0)
            {
                ERROR_PRINTF("Bug in case 2.2, node pair (%d, %d)\n", node0, node1);
                H2P_dense_mat_reset(H2_Dij);
                return;
            }
            H2P_dense_mat_p V0 = V[node0];
            H2P_dense_mat_p S0 = S[node0];
            // Bij = V{node1}' * linsolve(S{node1}, H2_Dij, struct('LT', true));
            CBLAS_TRSM(
                CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                H2_Dij->nrow, H2_Dij->ncol, 1.0, S0->data, S0->ld, H2_Dij->data, H2_Dij->ld
            );
            H2P_dense_mat_resize(HSS_Bij, V0->ncol, H2_Dij->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, H2_Dij->ncol, V0->nrow,
                1.0, V0->data, V0->ld, H2_Dij->data, H2_Dij->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            return;
        }  // End of "if (n_child0 == 0)"

        // 2.3: node0 is a non-leaf node
        if (n_child0 > 0)
        {
            H2P_dense_mat_p tmpB = thread_buf->mat0;
            H2P_dense_mat_p tmpM = thread_buf->mat1;
            tmpB->nrow = 0;
            H2P_SPDHSS_H2_gather_HSS_B(
                n_child0, 1, child0, &node1, n_node, 
                HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, HSS_B, tmpB
            );
            if (tmpB->nrow == 0)
            {
                ERROR_PRINTF("Bug in case 2.3, node pair (%d, %d)\n", node0, node1);
                H2P_dense_mat_reset(tmpB);
                H2P_dense_mat_reset(tmpM);
                return;
            }
            H2P_dense_mat_p V0    = V[node0];
            H2P_dense_mat_p Minv0 = Minv[node0];
            // Bij = V{node1}' * Minv{node1} * tmpB;
            H2P_dense_mat_resize(tmpM, Minv0->nrow, tmpB->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv0->nrow, tmpB->ncol, Minv0->ncol,
                1.0, Minv0->data, Minv0->ld, tmpB->data, tmpB->ld, 0.0, tmpM->data, tmpM->ld
            );
            H2P_dense_mat_resize(HSS_Bij, V0->ncol, tmpM->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasTrans, CblasNoTrans, V0->ncol, tmpM->ncol, V0->nrow,
                1.0, V0->data, V0->ld, tmpM->data, tmpM->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_SPDHSS_H2_clean_HSS_B(
                n_child0, 1, child0, &node1, n_node, 
                HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, HSS_B
            );
            return;
        }  // End of "if (n_child0 > 0)"
    }  // End of "if (level0 > level1)"

    if (level0 < level1)
    {
        // Note: node0 must be a leaf node
        if (n_child0 > 0)
        {
            ERROR_PRINTF("Bug in case 3, node pair (%d, %d)\n", node0, node1);
            return;
        }

        // 3.1: node0 and node1 are admissable
        if (is_adm)
        {
            H2P_dense_mat_p H2_Bij = thread_buf->mat0;
            H2P_dense_mat_resize(H2_Bij, 128, 128);
            H2_Bij->nrow = 0;
            H2P_get_Bij_block(h2mat, node0, node1, H2_Bij);
            if (H2_Bij->nrow == 0)
            {
                ERROR_PRINTF("Bug in case 3.1, node pair (%d, %d)\n", node0, node1);
                H2P_dense_mat_reset(H2_Bij);
                return;
            }
            H2P_dense_mat_p W1 = W[node1];
            // Bij = H2_B{H2_B_idx} * W{node2}';
            H2P_dense_mat_resize(HSS_Bij, H2_Bij->nrow, W1->nrow);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasTrans, H2_Bij->nrow, W1->nrow, H2_Bij->ncol,
                1.0, H2_Bij->data, H2_Bij->ld, W1->data, W1->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            return;
        }  // End of "if (is_adm)"

        // Otherwise: node0 and node1 are inadmissible

        // 3.2: node1 is a leaf node
        if (n_child1 == 0)
        {
            H2P_dense_mat_p H2_Dij = thread_buf->mat0;
            H2_Dij->nrow = 0;
            H2P_get_Dij_block(h2mat, node0, node1, H2_Dij);
            if (H2_Dij->nrow == 0)
            {
                ERROR_PRINTF("Bug in case 3.2, node pair (%d, %d)\n", node0, node1);
                H2P_dense_mat_reset(H2_Dij);
                return;
            }
            H2P_dense_mat_p V1 = V[node1];
            H2P_dense_mat_p S1 = S[node1];
            // Bij = linsolve(S{node2}, H2_Dij', struct('LT', true))' * V{node2};
            // S{node2} * X = H2_Dij', we need Bij = X' * V{node2}
            // Solve X' * S{node2}' = H2_Dij to obtain X' directly
            CBLAS_TRSM(
                CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, 
                H2_Dij->nrow, H2_Dij->ncol, 1.0, S1->data, S1->ld, H2_Dij->data, H2_Dij->ld
            );
            H2P_dense_mat_resize(HSS_Bij, H2_Dij->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, H2_Dij->nrow, V1->ncol, H2_Dij->ncol,
                1.0, H2_Dij->data, H2_Dij->ld, V1->data, V1->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            return;
        }  // End of "if (n_child1 == 0)"

        // 3.3: node1 is a non-leaf node
        if (n_child1 > 0)
        {
            H2P_dense_mat_p tmpB = thread_buf->mat0;
            H2P_dense_mat_p tmpM = thread_buf->mat1;
            tmpB->nrow = 0;
            H2P_SPDHSS_H2_gather_HSS_B(
                1, n_child1, &node0, child1, n_node, 
                HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, HSS_B, tmpB
            );
            if (tmpB->nrow == 0)
            {
                ERROR_PRINTF("Bug in case 3.3, node pair (%d, %d)\n", node0, node1);
                H2P_dense_mat_reset(tmpB);
                H2P_dense_mat_reset(tmpM);
                return;
            }
            H2P_dense_mat_p V1    = V[node1];
            H2P_dense_mat_p Minv1 = Minv[node1];
            // Bij = tmpB * Minv{node2} * V{node2};
            H2P_dense_mat_resize(tmpM, Minv1->nrow, V1->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv1->nrow, V1->ncol, Minv1->ncol,
                1.0, Minv1->data, Minv1->ld, V1->data, V1->ld, 0.0, tmpM->data, tmpM->ld
            );
            H2P_dense_mat_resize(HSS_Bij, tmpB->nrow, tmpM->ncol);
            CBLAS_GEMM(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpB->nrow, tmpM->ncol, tmpB->ncol,
                1.0, tmpB->data, tmpB->ld, tmpM->data, tmpM->ld, 0.0, HSS_Bij->data, HSS_Bij->ld
            );
            H2P_SPDHSS_H2_clean_HSS_B(
                1, n_child1, &node0, child1, n_node, 
                HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, HSS_B
            );
            return;
        }  // End of "if (n_child1 > 0)"
    }  // End of "if (level0 < level1)"
}

// Construct the list of blocks (i, j) at each level satisfying 
//   (1) (i, j) are inadmissible pairs;
//   (2) (i, j) are admissible but their parents are inadmissible.
// If (i, j) are at different levels, block (i, j) is contained and 
// processed in the lower level (closer to leaf level) of the two.
// Input parameter:
//   h2mat : Source H2 matrix
// Output parameters:
//   *level_HSS_Bij_pairs_ : Array, size h2mat->max_level+1, new HSS Bij pairs on each level
//   *n_HSS_Bij_pair_      : Total number of new HSS Bij pairs
void H2P_SPDHSS_H2_get_level_HSS_Bij_pairs(H2Pack_p h2mat, H2P_int_vec_p **level_HSS_Bij_pairs_, int *n_HSS_Bij_pair_)
{
    int n_node              = h2mat->n_node;
    int max_level           = h2mat->max_level;
    int H2_n_r_adm_pairs    = h2mat->n_r_adm_pair;
    int H2_n_r_inadm_pairs  = h2mat->n_r_inadm_pair;
    int *parent             = h2mat->parent;
    int *node_level         = h2mat->node_level;
    int *H2_r_adm_pairs     = h2mat->r_adm_pairs;
    int *H2_r_inadm_pairs   = h2mat->r_inadm_pairs;

    int n_level = max_level + 1;  // This is the total number of levels

    H2P_int_vec_p *level_HSS_Bij_pairs = (H2P_int_vec_p*) malloc(sizeof(H2P_int_vec_p) * n_level);
    int *inadm_max_level = (int*) malloc(sizeof(int) * H2_n_r_inadm_pairs);
    int *adm_max_level   = (int*) malloc(sizeof(int) * H2_n_r_adm_pairs);
    ASSERT_PRINTF(
        level_HSS_Bij_pairs != NULL && inadm_max_level != NULL && adm_max_level != NULL,
        "Failed to allocate arrays for storing new SPDHSS Bij pairs\n"
    );
    // inadm_max_lvl = max(node_lvl(H2_r_near_pair), [], 2);
    // adm_max_lvl   = max(node_lvl(H2_r_far_pair), [], 2);
    for (int i = 0; i < H2_n_r_inadm_pairs; i++)
    {
        int node0 = H2_r_inadm_pairs[2 * i];
        int node1 = H2_r_inadm_pairs[2 * i + 1];
        inadm_max_level[i] = MAX(node_level[node0], node_level[node1]);
    }
    for (int i = 0; i < H2_n_r_adm_pairs; i++)
    {
        int node0 = H2_r_adm_pairs[2 * i];
        int node1 = H2_r_adm_pairs[2 * i + 1];
        adm_max_level[i] = MAX(node_level[node0], node_level[node1]);
    }

    // inadm_pairs = H2_r_near_pair(inadm_max_lvl == max_level, :);
    // adm_pairs   = H2_r_far_pair(adm_max_lvl == max_level, :);
    int n_HSS_Bij_pair = 0;
    H2P_int_vec_init(&level_HSS_Bij_pairs[max_level], 1024);
    H2P_int_vec_p level_pairs = level_HSS_Bij_pairs[max_level];
    for (int i = 0; i < H2_n_r_inadm_pairs; i++)
    {
        if (inadm_max_level[i] != max_level) continue;
        H2P_int_vec_push_back(level_pairs, H2_r_inadm_pairs[2 * i]);
        H2P_int_vec_push_back(level_pairs, H2_r_inadm_pairs[2 * i + 1]);
    }
    for (int i = 0; i < H2_n_r_adm_pairs; i++)
    {
        if (adm_max_level[i] != max_level) continue;
        H2P_int_vec_push_back(level_pairs, H2_r_adm_pairs[2 * i]);
        H2P_int_vec_push_back(level_pairs, H2_r_adm_pairs[2 * i + 1]);
    }
    n_HSS_Bij_pair += level_pairs->length / 2;

    H2P_int_vec_p prev_pairs, prev_pairs1, work_buf;
    H2P_int_vec_init(&prev_pairs,  1024);
    H2P_int_vec_init(&prev_pairs1, 1024);
    H2P_int_vec_init(&work_buf,    1024);
    for (int i = max_level - 1; i >= 1; i--)
    {
        H2P_int_vec_p prev_pairs0 = level_HSS_Bij_pairs[i + 1];
        H2P_int_vec_set_capacity(prev_pairs, prev_pairs0->length);
        memcpy(prev_pairs->data, prev_pairs0->data, sizeof(int) * prev_pairs0->length);
        prev_pairs->length = prev_pairs0->length;

        // Handling partial admissible pairs
        for (int k = 0; k < prev_pairs->length; k++)
        {
            if (node_level[prev_pairs->data[k]] == i + 1) 
                prev_pairs->data[k] = parent[prev_pairs->data[k]];
        }

        // prev_pairs = prev_pairs(prev_pairs(:, 1) ~= prev_pairs(:, 2), :);
        // prev_pairs = unique(prev_pairs, 'rows');
        H2P_int_vec_set_capacity(work_buf, prev_pairs->length);
        int n_prev_pair = prev_pairs->length / 2;
        int *key = work_buf->data;
        int *val = work_buf->data + n_prev_pair;
        int *pp_data = prev_pairs->data;
        int valid_cnt = 0;
        for (int k = 0; k < n_prev_pair; k++)
        {
            int node0 = pp_data[2 * k];
            int node1 = pp_data[2 * k + 1];
            if (node0 == node1) continue;
            key[valid_cnt] = node0 * n_node + node1;
            val[valid_cnt] = k;
            valid_cnt++;
        }
        H2P_qsort_int_key_val(key, val, 0, valid_cnt - 1);
        H2P_int_vec_set_capacity(prev_pairs1, valid_cnt * 2);
        int *pp1_data = prev_pairs1->data;
        int cnt = 0, curr_key = -19241112;
        for (int k = 0; k < valid_cnt; k++)
        {
            if (curr_key != key[k])
            {
                curr_key = key[k];
                int pair_k = val[k];
                int node0 = pp_data[2 * pair_k];
                int node1 = pp_data[2 * pair_k + 1];
                pp1_data[2 * cnt]     = node0;
                pp1_data[2 * cnt + 1] = node1;
                cnt++;
            }
        }  // End of k loop
        prev_pairs1->length = 2 * cnt;

        // level_HSS_Bij_pairs{i} = [prev_pairs; inadm_blks; adm_blks];
        H2P_int_vec_init(&level_HSS_Bij_pairs[i], 1024);
        level_pairs = level_HSS_Bij_pairs[i];
        for (int k = 0; k < prev_pairs1->length; k++)
            H2P_int_vec_push_back(level_pairs, prev_pairs1->data[k]);
        for (int k = 0; k < H2_n_r_inadm_pairs; k++)
        {
            if (inadm_max_level[k] != i) continue;
            H2P_int_vec_push_back(level_pairs, H2_r_inadm_pairs[2 * k]);
            H2P_int_vec_push_back(level_pairs, H2_r_inadm_pairs[2 * k + 1]);
        }
        for (int k = 0; k < H2_n_r_adm_pairs; k++)
        {
            if (adm_max_level[k] != i) continue;
            H2P_int_vec_push_back(level_pairs, H2_r_adm_pairs[2 * k]);
            H2P_int_vec_push_back(level_pairs, H2_r_adm_pairs[2 * k + 1]);
        }
        n_HSS_Bij_pair += level_pairs->length / 2;
    }  // End of i loop
    level_HSS_Bij_pairs[0] = NULL;

    H2P_int_vec_destroy(&prev_pairs);
    H2P_int_vec_destroy(&prev_pairs1);
    H2P_int_vec_destroy(&work_buf);
    free(inadm_max_level);
    free(adm_max_level);
    *level_HSS_Bij_pairs_ = level_HSS_Bij_pairs;
    *n_HSS_Bij_pair_ = n_HSS_Bij_pair;
}

// Wrap up the new HSS matrix with calculated HSS_{U, B, D} and existing hierarchical tree information
// Input parameters:
//   h2mat          : Source H2 matrix
//   HSS_{U, B, D}  : New U/B/D matrices calculated in H2P_SPDHSS_H2_build()
//   HSS_B_p2i_{*}  : CSR matrix array triple, convert (i, j) pair to an index for HSS_B
//   HSS_D_pair2idx : Size h2mat->n_node, convert (i, i) pair to an index for HSS_D
// Output parameter:
//   *hssmat_ : New HSS matrix
void H2P_SPDHSS_H2_wrap_new_HSS(
    H2Pack_p h2mat, H2P_dense_mat_p *HSS_U, H2P_dense_mat_p *HSS_B, H2P_dense_mat_p *HSS_D, 
    const int *HSS_B_p2i_rowptr, const int *HSS_B_p2i_colidx, const int *HSS_B_p2i_val, 
    const int *HSS_D_pair2idx, H2Pack_p *hssmat_
)
{
    H2Pack_p hssmat;
    H2P_init(&hssmat, h2mat->pt_dim, h2mat->krnl_dim, h2mat->QR_stop_type, &h2mat->QR_stop_tol);
    
    int pt_dim    = h2mat->pt_dim;
    int n_point   = h2mat->n_point;
    int n_node    = h2mat->n_node;
    int max_child = h2mat->max_child;
    int max_level = h2mat->max_level;

    // 1. Copy point coordinates
    hssmat->n_point         = n_point;
    hssmat->max_leaf_points = h2mat->max_leaf_points;
    hssmat->max_leaf_size   = h2mat->max_leaf_size;
    size_t coord_msize = sizeof(DTYPE) * n_point * pt_dim;
    hssmat->coord_idx = (int*)   malloc(sizeof(int) * n_point);
    hssmat->coord     = (DTYPE*) malloc(coord_msize);
    memcpy(hssmat->coord_idx, h2mat->coord_idx, sizeof(int) * n_point);
    memcpy(hssmat->coord, h2mat->coord, coord_msize);
    
    // 2. Copy hierarchical partition tree information
    hssmat->n_node        = n_node;
    hssmat->root_idx      = n_node - 1;
    hssmat->n_leaf_node   = h2mat->n_leaf_node;
    hssmat->max_level     = h2mat->max_level;
    hssmat->krnl_mat_size = h2mat->krnl_mat_size;
    size_t int_n_node_msize  = sizeof(int)   * n_node;
    size_t int_n_level_msize = sizeof(int)   * (max_level + 1);
    size_t enbox_msize       = sizeof(DTYPE) * n_node * 2 * pt_dim;
    hssmat->parent        = malloc(int_n_node_msize);
    hssmat->children      = malloc(int_n_node_msize * max_child);
    hssmat->pt_cluster    = malloc(int_n_node_msize * 2);
    hssmat->mat_cluster   = malloc(int_n_node_msize * 2);
    hssmat->n_child       = malloc(int_n_node_msize);
    hssmat->node_level    = malloc(int_n_node_msize);
    hssmat->node_height   = malloc(int_n_node_msize);
    hssmat->level_n_node  = malloc(int_n_level_msize);
    hssmat->level_nodes   = malloc(int_n_level_msize * h2mat->n_leaf_node);
    hssmat->height_n_node = malloc(int_n_level_msize);
    hssmat->height_nodes  = malloc(int_n_level_msize * h2mat->n_leaf_node);
    hssmat->enbox         = malloc(enbox_msize);
    ASSERT_PRINTF(hssmat->parent        != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    ASSERT_PRINTF(hssmat->children      != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    ASSERT_PRINTF(hssmat->pt_cluster    != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    ASSERT_PRINTF(hssmat->mat_cluster   != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    ASSERT_PRINTF(hssmat->n_child       != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    ASSERT_PRINTF(hssmat->node_level    != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    ASSERT_PRINTF(hssmat->node_height   != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    ASSERT_PRINTF(hssmat->level_n_node  != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    ASSERT_PRINTF(hssmat->level_nodes   != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    ASSERT_PRINTF(hssmat->height_n_node != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    ASSERT_PRINTF(hssmat->height_nodes  != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    ASSERT_PRINTF(hssmat->enbox         != NULL, "Failed to allocate arrays for storing hierarchical partioning tree in SPDHSS\n");
    memcpy(hssmat->parent       , h2mat->parent       , int_n_node_msize);
    memcpy(hssmat->children     , h2mat->children     , int_n_node_msize * max_child);
    memcpy(hssmat->pt_cluster   , h2mat->pt_cluster   , int_n_node_msize * 2);
    memcpy(hssmat->mat_cluster  , h2mat->mat_cluster  , int_n_node_msize * 2);
    memcpy(hssmat->n_child      , h2mat->n_child      , int_n_node_msize);
    memcpy(hssmat->node_level   , h2mat->node_level   , int_n_node_msize);
    memcpy(hssmat->node_height  , h2mat->node_height  , int_n_node_msize);
    memcpy(hssmat->level_n_node , h2mat->level_n_node , int_n_level_msize);
    memcpy(hssmat->level_nodes  , h2mat->level_nodes  , int_n_level_msize * h2mat->n_leaf_node);
    memcpy(hssmat->height_n_node, h2mat->height_n_node, int_n_level_msize);
    memcpy(hssmat->height_nodes , h2mat->height_nodes , int_n_level_msize * h2mat->n_leaf_node);
    memcpy(hssmat->enbox        , h2mat->enbox        , enbox_msize);

    // 3. Copy H2 & HSS reduced (in)admissible pairs
    hssmat->min_adm_level      = h2mat->min_adm_level;
    hssmat->HSS_min_adm_level  = h2mat->HSS_min_adm_level;
    hssmat->n_r_inadm_pair     = h2mat->n_r_inadm_pair;
    hssmat->n_r_adm_pair       = h2mat->n_r_adm_pair;
    hssmat->HSS_n_r_inadm_pair = h2mat->HSS_n_r_inadm_pair;
    hssmat->HSS_n_r_adm_pair   = h2mat->HSS_n_r_adm_pair;
    size_t r_inadm_pairs_msize     = sizeof(int) * h2mat->n_r_inadm_pair * 2;
    size_t r_adm_pairs_msize       = sizeof(int) * h2mat->n_r_adm_pair   * 2;
    size_t HSS_r_inadm_pairs_msize = sizeof(int) * h2mat->HSS_n_r_inadm_pair * 2;
    size_t HSS_r_adm_pairs_msize   = sizeof(int) * h2mat->HSS_n_r_adm_pair   * 2;
    hssmat->r_inadm_pairs     = (int*) malloc(r_inadm_pairs_msize);
    hssmat->r_adm_pairs       = (int*) malloc(r_adm_pairs_msize);
    hssmat->HSS_r_inadm_pairs = (int*) malloc(HSS_r_inadm_pairs_msize);
    hssmat->HSS_r_adm_pairs   = (int*) malloc(HSS_r_adm_pairs_msize);
    hssmat->node_inadm_lists  = (int*) malloc(int_n_node_msize * h2mat->max_neighbor);
    hssmat->node_n_r_inadm    = (int*) malloc(int_n_node_msize);
    hssmat->node_n_r_adm      = (int*) malloc(int_n_node_msize);
    ASSERT_PRINTF(hssmat->r_inadm_pairs     != NULL, "Failed to allocate arrays for storing (in)admissible pairs in SPDHSS\n");
    ASSERT_PRINTF(hssmat->r_adm_pairs       != NULL, "Failed to allocate arrays for storing (in)admissible pairs in SPDHSS\n");
    ASSERT_PRINTF(hssmat->HSS_r_inadm_pairs != NULL, "Failed to allocate arrays for storing (in)admissible pairs in SPDHSS\n");
    ASSERT_PRINTF(hssmat->HSS_r_adm_pairs   != NULL, "Failed to allocate arrays for storing (in)admissible pairs in SPDHSS\n");
    ASSERT_PRINTF(hssmat->node_inadm_lists  != NULL, "Failed to allocate arrays for storing (in)admissible pairs in SPDHSS\n");
    ASSERT_PRINTF(hssmat->node_n_r_inadm    != NULL, "Failed to allocate arrays for storing (in)admissible pairs in SPDHSS\n");
    ASSERT_PRINTF(hssmat->node_n_r_adm      != NULL, "Failed to allocate arrays for storing (in)admissible pairs in SPDHSS\n");
    memcpy(hssmat->r_inadm_pairs    , h2mat->r_inadm_pairs    , r_inadm_pairs_msize);
    memcpy(hssmat->r_adm_pairs      , h2mat->r_adm_pairs      , r_adm_pairs_msize);
    memcpy(hssmat->HSS_r_inadm_pairs, h2mat->HSS_r_inadm_pairs, HSS_r_inadm_pairs_msize);
    memcpy(hssmat->HSS_r_adm_pairs  , h2mat->HSS_r_adm_pairs  , HSS_r_adm_pairs_msize);
    memcpy(hssmat->node_inadm_lists , h2mat->node_inadm_lists , int_n_node_msize * h2mat->max_neighbor);
    memcpy(hssmat->node_n_r_inadm   , h2mat->node_n_r_inadm   , int_n_node_msize);
    memcpy(hssmat->node_n_r_adm     , h2mat->node_n_r_adm     , int_n_node_msize);

    // 4. Initialize thread-local buffer
    hssmat->tb = (H2P_thread_buf_p*) malloc(sizeof(H2P_thread_buf_p) * hssmat->n_thread);
    ASSERT_PRINTF(hssmat->tb != NULL, "Failed to allocate %d thread buffers in SPDHSS\n", hssmat->n_thread);
    for (int i = 0; i < hssmat->n_thread; i++)
        H2P_thread_buf_init(&hssmat->tb[i], hssmat->krnl_mat_size);

    // 5. Set up kernel pointers and U/B/D info
    hssmat->BD_JIT          = 0;
    hssmat->is_HSS          = 1;
    hssmat->krnl_param      = h2mat->krnl_param;
    hssmat->krnl_eval       = h2mat->krnl_eval;
    hssmat->krnl_bimv       = h2mat->krnl_bimv;
    hssmat->krnl_bimv_flops = h2mat->krnl_bimv_flops;

    int    n_thread         = hssmat->n_thread;
    int    n_leaf_node      = hssmat->n_leaf_node;
    int    *leaf_nodes      = hssmat->height_nodes;
    size_t *mat_size        = hssmat->mat_size;
    int    BD_ntask_thread  = (hssmat->BD_JIT == 1) ? BD_NTASK_THREAD : 1;

    // 5.1 Copy U matrices directly
    hssmat->n_UJ = h2mat->n_UJ;
    hssmat->U    = HSS_U;
    for (int i = 0; i < n_node; i++)
    {
        if (HSS_U[i] != NULL)
        {
            mat_size[U_SIZE_IDX]      += HSS_U[i]->nrow * HSS_U[i]->ncol;
            mat_size[MV_FWD_SIZE_IDX] += HSS_U[i]->nrow * HSS_U[i]->ncol;
            mat_size[MV_FWD_SIZE_IDX] += HSS_U[i]->nrow + HSS_U[i]->ncol;
            mat_size[MV_BWD_SIZE_IDX] += HSS_U[i]->nrow * HSS_U[i]->ncol;
            mat_size[MV_BWD_SIZE_IDX] += HSS_U[i]->nrow + HSS_U[i]->ncol;
        } else {
            H2P_dense_mat_init(&HSS_U[i], 0, 0);
            HSS_U[i]->nrow = 0;
            HSS_U[i]->ncol = 0;
            HSS_U[i]->ld   = 0;
        }
    }

    // 5.2 Copy B matrices
    int HSS_n_r_adm_pair = hssmat->HSS_n_r_adm_pair;
    int *HSS_r_adm_pairs = hssmat->HSS_r_adm_pairs;
    size_t int_r_adm_pairs_msize = sizeof(int) * HSS_n_r_adm_pair;
    int    *B_pair_i = (int*)    malloc(int_r_adm_pairs_msize * 2);
    int    *B_pair_j = (int*)    malloc(int_r_adm_pairs_msize * 2);
    int    *B_pair_v = (int*)    malloc(int_r_adm_pairs_msize * 2);
    int    *B_nrow   = (int*)    malloc(int_r_adm_pairs_msize);
    int    *B_ncol   = (int*)    malloc(int_r_adm_pairs_msize);
    size_t *B_ptr    = (size_t*) malloc(sizeof(size_t) * (HSS_n_r_adm_pair + 1));
    ASSERT_PRINTF(
        B_nrow   != NULL && B_ncol   != NULL && B_ptr    != NULL,
        "Failed to allocate %d SPDHSS B matrices infomation array\n", HSS_n_r_adm_pair
    );
    ASSERT_PRINTF(
        B_pair_i != NULL && B_pair_j != NULL && B_pair_v != NULL,
        "Failed to allocate working buffer for SPDHSS B matrices indexing\n"
    );
    hssmat->n_B    = HSS_n_r_adm_pair;
    hssmat->B_nrow = B_nrow;
    hssmat->B_ncol = B_ncol;
    hssmat->B_ptr  = B_ptr;

    int    B_pair_cnt   = 0;
    size_t B_total_size = 0;
    B_ptr[0] = 0;
    for (int i = 0; i < HSS_n_r_adm_pair; i++)
    {
        int node0  = HSS_r_adm_pairs[2 * i];
        int node1  = HSS_r_adm_pairs[2 * i + 1];
        int HSS_B_idx = H2P_get_int_CSR_elem(HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, node0, node1);
        ASSERT_PRINTF(HSS_B_idx > 0, "SPDHSS_B{%d, %d} does not exist!\n", node0, node1);
        HSS_B_idx--;
        H2P_dense_mat_p HSS_Bi = HSS_B[HSS_B_idx];
        B_nrow[i] = HSS_Bi->nrow;
        B_ncol[i] = HSS_Bi->ncol;
        size_t Bi_size = (size_t) B_nrow[i] * (size_t) B_ncol[i];
        B_total_size += Bi_size;
        B_ptr[i + 1] = Bi_size;
        B_pair_i[B_pair_cnt] = node0;
        B_pair_j[B_pair_cnt] = node1;
        B_pair_v[B_pair_cnt] = i + 1;
        B_pair_cnt++;
        mat_size[MV_MID_SIZE_IDX] += B_nrow[i] * B_ncol[i];
        mat_size[MV_MID_SIZE_IDX] += 2 * (B_nrow[i] + B_ncol[i]);
    }
    H2P_int_vec_p B_blk = hssmat->B_blk;
    H2P_partition_workload(HSS_n_r_adm_pair, B_ptr + 1, B_total_size, n_thread * BD_ntask_thread, B_blk);
    for (int i = 1; i <= HSS_n_r_adm_pair; i++) B_ptr[i] += B_ptr[i - 1];
    mat_size[B_SIZE_IDX] = B_total_size;

    hssmat->B_p2i_rowptr = (int*) malloc(sizeof(int) * (n_node + 1));
    hssmat->B_p2i_colidx = (int*) malloc(int_r_adm_pairs_msize * 2);
    hssmat->B_p2i_val    = (int*) malloc(int_r_adm_pairs_msize * 2);
    ASSERT_PRINTF(hssmat->B_p2i_rowptr != NULL, "Failed to allocate arrays for SPDHSS B matrices indexing\n");
    ASSERT_PRINTF(hssmat->B_p2i_colidx != NULL, "Failed to allocate arrays for SPDHSS B matrices indexing\n");
    ASSERT_PRINTF(hssmat->B_p2i_val    != NULL, "Failed to allocate arrays for SPDHSS B matrices indexing\n");
    H2P_int_COO_to_CSR(
        n_node, B_pair_cnt, B_pair_i, B_pair_j, B_pair_v, 
        hssmat->B_p2i_rowptr, hssmat->B_p2i_colidx, hssmat->B_p2i_val
    );

    hssmat->B_data = (DTYPE*) malloc_aligned(sizeof(DTYPE) * B_total_size, 64);
    ASSERT_PRINTF(hssmat->B_data != NULL, "Failed to allocate space for storing all %zu SPDHSS B matrices elements\n", B_total_size);
    DTYPE *B_data = hssmat->B_data;
    const int n_B_blk = B_blk->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();

        //#pragma omp for schedule(dynamic) nowait
        //for (int i_blk = 0; i_blk < n_B_blk; i_blk++)
        int i_blk = tid;    // Use first-touch policy for better NUMA memory access performance
        {
            int B_blk_s = B_blk->data[i_blk];
            int B_blk_e = B_blk->data[i_blk + 1];
            if (i_blk >= n_B_blk)
            {
                B_blk_s = 0; 
                B_blk_e = 0;
            }
            for (int i = B_blk_s; i < B_blk_e; i++)
            {
                int node0 = HSS_r_adm_pairs[2 * i];
                int node1 = HSS_r_adm_pairs[2 * i + 1];
                int HSS_B_idx = H2P_get_int_CSR_elem(HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, node0, node1);
                ASSERT_PRINTF(HSS_B_idx != 0, "SPDHSS_B{%d, %d} does not exist!\n", node0, node1);
                HSS_B_idx--;
                H2P_dense_mat_p HSS_Bi = HSS_B[HSS_B_idx];
                int Bi_nrow = HSS_Bi->nrow;
                int Bi_ncol = HSS_Bi->ncol;
                DTYPE *Bi = B_data + B_ptr[i];
                copy_matrix_block(sizeof(DTYPE), Bi_nrow, Bi_ncol, HSS_Bi->data, HSS_Bi->ld, Bi, Bi_ncol);
            }
        }  // End of i_blk loop
    }  // End of "#pragma omp parallel"

    // 5.3 Copy D matrices
    size_t int_n_leaf_node_msize = sizeof(int) * n_leaf_node;
    int    *D_pair_i = (int*)    malloc(int_n_leaf_node_msize * 2);
    int    *D_pair_j = (int*)    malloc(int_n_leaf_node_msize * 2);
    int    *D_pair_v = (int*)    malloc(int_n_leaf_node_msize * 2);
    int    *D_nrow   = (int*)    malloc(int_n_leaf_node_msize);
    int    *D_ncol   = (int*)    malloc(int_n_leaf_node_msize);
    size_t *D_ptr    = (size_t*) malloc(sizeof(size_t) * (n_leaf_node + 1));
    ASSERT_PRINTF(
        D_nrow   != NULL && D_ncol   != NULL && D_ptr    != NULL,
        "Failed to allocate %d SPDHSS D matrices information array\n", n_leaf_node
    );
    ASSERT_PRINTF(
        D_pair_i != NULL && D_pair_j != NULL && D_pair_v != NULL,
        "Failed to allocate working buffer for SPDHSS D matrices indexing\n"
    );
    hssmat->n_D    = n_leaf_node;
    hssmat->D_nrow = D_nrow;
    hssmat->D_ncol = D_ncol;
    hssmat->D_ptr  = D_ptr;

    int    D_pair_cnt   = 0;
    size_t D_total_size = 0;
    D_ptr[0] = 0;
    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        int HSS_D_idx = HSS_D_pair2idx[node];
        ASSERT_PRINTF(HSS_D_idx >= 0, "SPDHSS_D{%d, %d} does not exist!\n", node, node);
        H2P_dense_mat_p HSS_Di = HSS_D[HSS_D_idx];
        D_nrow[i] = HSS_Di->nrow;
        D_ncol[i] = HSS_Di->ncol;
        size_t Di_size = (size_t) D_nrow[i] * (size_t) D_ncol[i];
        D_total_size += Di_size;
        D_ptr[i + 1] = Di_size;
        D_pair_i[D_pair_cnt] = node;
        D_pair_j[D_pair_cnt] = node;
        D_pair_v[D_pair_cnt] = i + 1;
        D_pair_cnt++;
        mat_size[MV_DEN_SIZE_IDX] += D_nrow[i] * D_ncol[i];
        mat_size[MV_DEN_SIZE_IDX] += 2 * (D_nrow[i] + D_ncol[i]);
    }
    H2P_int_vec_p D_blk0 = hssmat->D_blk0;
    H2P_int_vec_p D_blk1 = hssmat->D_blk1;
    H2P_partition_workload(n_leaf_node, D_ptr + 1, D_total_size, n_thread * BD_ntask_thread, D_blk0);
    for (int i = 1; i <= n_leaf_node; i++) D_ptr[i] += D_ptr[i - 1];
    D_blk1->length  = 1;
    D_blk1->data[0] = 0;
    mat_size[D_SIZE_IDX] = D_total_size;

    hssmat->D_p2i_rowptr = (int*) malloc(sizeof(int) * (n_node + 1));
    hssmat->D_p2i_colidx = (int*) malloc(int_n_leaf_node_msize * 2);
    hssmat->D_p2i_val    = (int*) malloc(int_n_leaf_node_msize * 2);
    ASSERT_PRINTF(hssmat->D_p2i_rowptr != NULL, "Failed to allocate arrays for SPDHSS D matrices indexing\n");
    ASSERT_PRINTF(hssmat->D_p2i_colidx != NULL, "Failed to allocate arrays for SPDHSS D matrices indexing\n");
    ASSERT_PRINTF(hssmat->D_p2i_val    != NULL, "Failed to allocate arrays for SPDHSS D matrices indexing\n");
    H2P_int_COO_to_CSR(
        n_node, D_pair_cnt, D_pair_i, D_pair_j, D_pair_v, 
        hssmat->D_p2i_rowptr, hssmat->D_p2i_colidx, hssmat->D_p2i_val
    );

    hssmat->D_data = (DTYPE*) malloc_aligned(sizeof(DTYPE) * D_total_size, 64);
    ASSERT_PRINTF(hssmat->D_data != NULL, "Failed to allocate space for storing all %zu SPDHSS B matrices elements\n", D_total_size);
    DTYPE *D_data = hssmat->D_data;
    const int n_D0_blk = D_blk0->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        //#pragma omp for schedule(dynamic) nowait
        //for (int i_blk0 = 0; i_blk0 < n_D0_blk; i_blk0++)
        int i_blk0 = tid;    // Use first-touch policy for better NUMA memory access performance
        {
            int D_blk0_s = D_blk0->data[i_blk0];
            int D_blk0_e = D_blk0->data[i_blk0 + 1];
            if (i_blk0 >= n_D0_blk)
            {
                D_blk0_s = 0;
                D_blk0_e = 0;
            }
            for (int i = D_blk0_s; i < D_blk0_e; i++)
            {
                int node = leaf_nodes[i];
                int HSS_D_idx = HSS_D_pair2idx[node];
                H2P_dense_mat_p HSS_Di = HSS_D[HSS_D_idx];
                int Di_nrow = HSS_Di->nrow;
                int Di_ncol = HSS_Di->ncol;
                DTYPE *Di = D_data + D_ptr[i];
                copy_matrix_block(sizeof(DTYPE), Di_nrow, Di_ncol, HSS_Di->data, HSS_Di->ld, Di, Di_ncol);
            }
        }  // End of i_blk0 loop
    }  // End of "#pragma omp parallel"

    // 6. Copy environment variables & permutation indices
    hssmat->mm_max_n_vec  = h2mat->mm_max_n_vec;
    hssmat->print_timers  = h2mat->print_timers;
    hssmat->print_dbginfo = h2mat->print_dbginfo;
    size_t pmt_idx_msize = sizeof(int) * h2mat->krnl_mat_size;
    int *fwd_pmt_idx = (int*) malloc(pmt_idx_msize);
    int *bwd_pmt_idx = (int*) malloc(pmt_idx_msize);
    memcpy(fwd_pmt_idx, h2mat->fwd_pmt_idx, pmt_idx_msize);
    memcpy(bwd_pmt_idx, h2mat->bwd_pmt_idx, pmt_idx_msize);
    hssmat->fwd_pmt_idx = fwd_pmt_idx;
    hssmat->bwd_pmt_idx = bwd_pmt_idx;

    size_t krnl_mat_msize = sizeof(DTYPE) * hssmat->krnl_mat_size;
    hssmat->xT    = (DTYPE*) malloc(krnl_mat_msize);
    hssmat->yT    = (DTYPE*) malloc(krnl_mat_msize);
    hssmat->pmt_x = (DTYPE*) malloc(krnl_mat_msize * hssmat->mm_max_n_vec);
    hssmat->pmt_y = (DTYPE*) malloc(krnl_mat_msize * hssmat->mm_max_n_vec);
    ASSERT_PRINTF(
        hssmat->xT != NULL && hssmat->yT != NULL && hssmat->pmt_x != NULL && hssmat->pmt_y != NULL,
        "Failed to allocate working arrays of size %d for matvec & matmul\n", 2 * hssmat->krnl_mat_size * (hssmat->mm_max_n_vec+1)
    );


    *hssmat_ = hssmat;
}

// Build an SPD HSS matrix A_{HSS} from an H2 matrix s.t. A_{HSS} ~= A_{H2}
void H2P_SPDHSS_H2_build(
    const int max_rank, const DTYPE reltol, const DTYPE shift, 
    H2Pack_p h2mat, H2Pack_p *hssmat_
)
{
    if (h2mat == NULL || h2mat->U == NULL || h2mat->is_HSS)
    {
        ERROR_PRINTF("Input h2mat is not constructed or configured as HSS\n");
        return;
    }

    if (h2mat->is_RPY_Ewald)
    {
        ERROR_PRINTF("Cannot construct SPDHSS for RPY Ewald kernel\n");
        return;
    }

    #ifdef __linux__
    // Any H2P_dense_mat_t->data allocation > 1KB will use mmap instead of sbrk and can be released later
    mallopt(M_MMAP_THRESHOLD, 1024);
    #endif

    int n_node          = h2mat->n_node;
    int n_thread        = h2mat->n_thread;
    int n_leaf_node     = h2mat->n_leaf_node;
    int max_child       = h2mat->max_child;
    int max_level       = h2mat->max_level;
    int *children       = h2mat->children;
    int *n_child        = h2mat->n_child;
    int *level_n_node   = h2mat->level_n_node;
    int *level_nodes    = h2mat->level_nodes;
    int *node_level     = h2mat->node_level;
    int *leaf_nodes     = h2mat->height_nodes;
    H2P_dense_mat_p  *H2_U = h2mat->U;
    H2P_thread_buf_p *thread_buf = h2mat->tb;

    int n_level = max_level + 1;  // This is the total number of levels

    double st, et, build_U_t = 0.0, build_B_t = 0.0, build_D_t = 0.0;

    // 1. Accumulate off-diagonal block row H2 matvec results
    st = get_wtime_sec();
    int n_vec = max_rank + 10;
    H2P_dense_mat_p *Yk;
    H2P_SPDHSS_H2_acc_matvec(h2mat, n_vec, &Yk);
    et = get_wtime_sec();
    build_U_t += et - st;
    printf("SPDHSS build: accumulative matvec finished %f.\n", et - st);
    fflush(stdout);

    // 2. Get the new HSS Bij pairs on each level
    st = get_wtime_sec();
    H2P_int_vec_p *level_HSS_Bij_pairs;
    int n_HSS_Bij_pair;
    H2P_SPDHSS_H2_get_level_HSS_Bij_pairs(h2mat, &level_HSS_Bij_pairs, &n_HSS_Bij_pair);
    et = get_wtime_sec();
    build_B_t += et - st;

    // 3. Prepare auxiliary matrices
    H2P_dense_mat_p *S     = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    H2P_dense_mat_p *V     = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    H2P_dense_mat_p *W     = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    H2P_dense_mat_p *Minv  = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    H2P_dense_mat_p *HSS_U = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    H2P_dense_mat_p *HSS_B = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_HSS_Bij_pair);
    H2P_dense_mat_p *HSS_D = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_leaf_node);
    ASSERT_PRINTF(
        S != NULL && V != NULL && W != NULL && Minv != NULL,
        "Failed to allocate %d working arrays for SPDHSS construction\n", 4 * n_node
    );
    ASSERT_PRINTF(HSS_U != NULL, "Failed to allocate %d SPDHSS U matrices\n", n_node);
    ASSERT_PRINTF(HSS_B != NULL, "Failed to allocate %d SPDHSS B matrices\n", n_HSS_Bij_pair);
    ASSERT_PRINTF(HSS_D != NULL, "Failed to allocate %d SPDHSS D matrices\n", n_leaf_node);
    for (int i = 0; i < n_node; i++)
    {
        S[i]     = NULL;
        V[i]     = NULL;
        W[i]     = NULL;
        Minv[i]  = NULL;
        HSS_U[i] = NULL;
    }
    for (int i = 0; i < n_HSS_Bij_pair; i++) HSS_B[i] = NULL;
    for (int i = 0; i < n_leaf_node;    i++) HSS_D[i] = NULL;

    // 4. Construct all HSS_{B,D}_pair2idx in advance. Some HSS_B_pair2idx(i, j) 
    //    will be set as 0 but no new (i, j) pair will be added later
    int *HSS_D_pair2idx = (int*) malloc(sizeof(int) * n_node);
    memset(HSS_D_pair2idx, 0, sizeof(int) * n_node);
    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        HSS_D_pair2idx[node] = i;
    }
    st = get_wtime_sec();
    int HSS_B_n_pair = 0;
    for (int i = max_level; i >= 1; i--)
    {
        H2P_int_vec_p level_i_HSS_Bij_pairs = level_HSS_Bij_pairs[i];
        HSS_B_n_pair += level_i_HSS_Bij_pairs->length / 2;
    }
    int *HSS_B_pair_i = (int*) malloc(sizeof(int) * HSS_B_n_pair);
    int *HSS_B_pair_j = (int*) malloc(sizeof(int) * HSS_B_n_pair);
    int *HSS_B_pair_v = (int*) malloc(sizeof(int) * HSS_B_n_pair);
    int *HSS_B_p2i_rowptr = (int*) malloc(sizeof(int) * (n_node + 1));
    int *HSS_B_p2i_colidx = (int*) malloc(sizeof(int) * HSS_B_n_pair);
    int *HSS_B_p2i_val    = (int*) malloc(sizeof(int) * HSS_B_n_pair);
    ASSERT_PRINTF(
        HSS_B_pair_i != NULL && HSS_B_pair_j != NULL && HSS_B_pair_v,
        "Failed to allocate work arrays for indexing SPDHSS Bij pairs\n"
    );
    ASSERT_PRINTF(
        HSS_B_p2i_rowptr != NULL && HSS_B_p2i_colidx != NULL && HSS_B_p2i_val,
        "Failed to allocate arrays for indexing SPDHSS Bij pairs\n"
    );
    int HSS_B_idx = 0;
    for (int i = max_level; i >= 1; i--)
    {
        H2P_int_vec_p level_i_HSS_Bij_pairs = level_HSS_Bij_pairs[i];
        for (int j = 0; j < level_i_HSS_Bij_pairs->length / 2; j++)
        {
            int node0 = level_i_HSS_Bij_pairs->data[2 * j];
            int node1 = level_i_HSS_Bij_pairs->data[2 * j + 1];
            HSS_B_pair_i[HSS_B_idx] = node0;
            HSS_B_pair_j[HSS_B_idx] = node1;
            HSS_B_pair_v[HSS_B_idx] = HSS_B_idx + 1;
            HSS_B_idx++;
        }
    }
    H2P_int_COO_to_CSR(
        n_node, HSS_B_idx, HSS_B_pair_i, HSS_B_pair_j, HSS_B_pair_v,
        HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val
    );
    free(HSS_B_pair_i);
    free(HSS_B_pair_j);
    free(HSS_B_pair_v);
    et = get_wtime_sec();
    build_B_t += et - st;

    // 5. Loop over all leaf nodes to construct new D matrices
    st = get_wtime_sec();
    H2P_int_vec_p D_blk0 = h2mat->D_blk0;
    const int n_D0_blk = D_blk0->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        #pragma omp for schedule(dynamic)
        for (int i_blk0 = 0; i_blk0 < n_D0_blk; i_blk0++)
        {
            int D_blk0_s = D_blk0->data[i_blk0];
            int D_blk0_e = D_blk0->data[i_blk0 + 1];
            if (i_blk0 >= n_D0_blk)
            {
                D_blk0_s = 0;
                D_blk0_e = 0;
            }
            for (int i = D_blk0_s; i < D_blk0_e; i++)
            {
                // HSS_D{HSS_D_idx} = H2_D{H2_D_idx} + shift * eye(size(H2_D{H2_D_idx}));
                int node = leaf_nodes[i];
                int HSS_D_idx = HSS_D_pair2idx[node];
                H2P_dense_mat_init(&HSS_D[HSS_D_idx], 8, 8);
                H2P_dense_mat_p HSS_Dij = HSS_D[HSS_D_idx];
                H2P_get_Dij_block(h2mat, node, node, HSS_Dij);
                for (int k = 0; k < HSS_Dij->nrow; k++)
                {
                    int idx_kk = k * (HSS_Dij->nrow + 1);
                    HSS_Dij->data[idx_kk] += shift;
                }
            }  // End of i loop
        }  // End of i_blk0 loop
    }  // End of pragma omp parallel 
    et = get_wtime_sec();
    build_D_t += et - st;

    // 6. Level by level hierarchical construction for U and B matrices
    int is_SPD = 1;
    for (int i = max_level; i >= 1; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);

        if (!is_SPD) continue;

        int level_i_HSS_Bij_n_pair = level_HSS_Bij_pairs[i]->length / 2;
        int *level_i_HSS_Bij_pairs = level_HSS_Bij_pairs[i]->data;
        
        st = get_wtime_sec();
        // Build new U matrices
        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();
            H2P_int_vec_p   idx0 = thread_buf[tid]->idx0;
            H2P_dense_mat_p mat0 = thread_buf[tid]->mat0;
            H2P_dense_mat_p mat1 = thread_buf[tid]->mat1;
            H2P_dense_mat_p mat2 = thread_buf[tid]->mat2;

            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                if (!is_SPD) continue;
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                int *node_children = children + node * max_child;

                int info;
                if (n_child_node == 0)
                {
                    // HSS_D_idx = HSS_D_pair2idx(node, node);
                    int HSS_D_idx = HSS_D_pair2idx[node];
                    H2P_dense_mat_p HSS_Dij = HSS_D[HSS_D_idx];
                    // [S{node}, chol_flag] = chol(HSS_D{HSS_D_idx}, 'lower');
                    H2P_dense_mat_init(&S[node], HSS_Dij->nrow, HSS_Dij->ncol);
                    copy_matrix_block(sizeof(DTYPE), HSS_Dij->nrow, HSS_Dij->ncol, HSS_Dij->data, HSS_Dij->ld, S[node]->data, S[node]->ld);
                    info = LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', S[node]->nrow, S[node]->data, S[node]->ld);
                    for (int k = 0; k < S[node]->nrow; k++)
                    {
                        DTYPE *S_kk1 = S[node]->data + k * S[node]->nrow + (k + 1);
                        int n_zero_row = S[node]->nrow - (k + 1);
                        memset(S_kk1, 0, sizeof(DTYPE) * n_zero_row);
                    }
                    if (info != 0)
                    {
                        ERROR_PRINTF("Node %d potrf() returned %d, target matrix with shifting %.2lf is not SPD\n", node, info, shift);
                        is_SPD = 0;
                        continue;
                    }
                    // tmpY = linsolve(S{node}, Yk{node}{1}, struct('LT', true));
                    H2P_dense_mat_p *node_Yk = Yk + node * max_level;
                    H2P_dense_mat_p tmpY = mat0;
                    H2P_dense_mat_resize(tmpY, node_Yk[0]->nrow + 1, node_Yk[0]->ncol);
                    DTYPE *tau = tmpY->data + node_Yk[0]->nrow * node_Yk[0]->ncol;
                    tmpY->nrow--;
                    copy_matrix_block(sizeof(DTYPE), tmpY->nrow, tmpY->ncol, node_Yk[0]->data, node_Yk[0]->ld, tmpY->data, tmpY->ld);
                    ASSERT_PRINTF(
                        tmpY->nrow == S[node]->nrow, 
                        "Node %d: tmpY->nrow (%d) mismatch S->nrow (%d)\n",
                        node, tmpY->nrow, S[node]->nrow
                    );
                    CBLAS_TRSM(
                        CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                        tmpY->nrow, tmpY->ncol, 1.0, S[node]->data, S[node]->ld, tmpY->data, tmpY->ld
                    );
                    // V_ncol = min([size(tmpY), max_rank]);
                    // [tmpQ, ~, ~] = qr(tmpY, 0);
                    // V{node} = tmpQ(:, 1 : V_ncol);
                    int tmpQ_ncol = MIN(tmpY->nrow, tmpY->ncol);
                    int V_ncol = MIN(tmpQ_ncol, max_rank);
                    H2P_dense_mat_p tmpQ = tmpY;
                    H2P_int_vec_p   jpvt = idx0;
                    H2P_int_vec_set_capacity(jpvt, tmpQ->ncol);
                    memset(jpvt->data, 0, sizeof(int) * tmpQ->ncol);
                    LAPACK_GEQPF(LAPACK_ROW_MAJOR, tmpQ->nrow, tmpQ->ncol, tmpQ->data, tmpQ->ld, jpvt->data, tau);
                    int V_ncol1 = -1;
                    DTYPE stop_diag = DABS(tmpQ->data[0]) * reltol;
                    for (int k = 0; k < V_ncol; k++)
                    {
                        if (DABS(tmpQ->data[k * tmpQ->ld + k]) < stop_diag)
                        {
                            V_ncol1 = k - 1;
                            break;
                        }
                    }
                    if (V_ncol1 > 0) V_ncol = V_ncol1;
                    LAPACK_ORGQR(LAPACK_ROW_MAJOR, tmpQ->nrow, tmpQ_ncol, tmpQ_ncol, tmpQ->data, tmpQ->ld, tau);
                    H2P_dense_mat_init(&V[node], tmpQ->nrow, V_ncol);
                    copy_matrix_block(sizeof(DTYPE), tmpQ->nrow, V_ncol, tmpQ->data, tmpQ->ld, V[node]->data, V[node]->ld);
                    // HSS_U{node} = S{node} * V{node};
                    H2P_dense_mat_init(&HSS_U[node], S[node]->nrow, V[node]->ncol);
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, S[node]->nrow, V[node]->ncol, S[node]->ncol,
                        1.0, S[node]->data, S[node]->ld, V[node]->data, V[node]->ld, 0.0, HSS_U[node]->data, HSS_U[node]->ld
                    );
                    // Yk{node}(1) = [];
                    // for k = 1 : length(Yk{node})
                    //    Yk{node}{k} = V{node}' * linsolve(S{node}, Yk{node}{k}, struct('LT', true));
                    // end
                    int last_k = 0;
                    for (int k = 1; k < max_level; k++)
                    {
                        if (node_Yk[k]->ld == 0) break;  // Empty Yk{node}{k}
                        H2P_dense_mat_p node_Yk_k0 = node_Yk[k - 1];
                        H2P_dense_mat_p node_Yk_k  = node_Yk[k];
                        CBLAS_TRSM(
                            CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 
                            node_Yk_k->nrow, node_Yk_k->ncol, 1.0, S[node]->data, S[node]->ld, node_Yk_k->data, node_Yk_k->ld
                        );
                        H2P_dense_mat_resize(node_Yk_k0, V[node]->ncol, node_Yk_k->ncol);
                        CBLAS_GEMM(
                            CblasRowMajor, CblasTrans, CblasNoTrans, V[node]->ncol, node_Yk_k->ncol, V[node]->nrow,
                            1.0, V[node]->data, V[node]->ld, node_Yk_k->data, node_Yk_k->ld, 0.0, node_Yk_k0->data, node_Yk_k0->ld
                        );
                        last_k = k;
                    }  // End of k loop
                    H2P_dense_mat_destroy(&node_Yk[last_k]);
                    // if (~isempty(H2_U{node}))
                    //     W{node} = V{node}' * linsolve(S{node}, H2_U{node}, struct('LT', true));
                    // end
                    H2P_dense_mat_p H2_U_node = H2_U[node];
                    if (H2_U_node->ld > 0)
                    {
                        H2P_dense_mat_p tmpM = tmpQ;
                        H2P_dense_mat_resize(tmpM, H2_U_node->nrow, H2_U_node->ncol);
                        copy_matrix_block(sizeof(DTYPE), H2_U_node->nrow, H2_U_node->ncol, H2_U_node->data, H2_U_node->ld, tmpM->data, tmpM->ld);
                        ASSERT_PRINTF(
                            tmpM->nrow == S[node]->nrow, 
                            "Node %d: H2_U->nrow (%d) mismatch S->nrow (%d)\n",
                            node, tmpM->nrow, S[node]->nrow
                        );
                        CBLAS_TRSM(
                            CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                            tmpM->nrow, tmpM->ncol, 1.0, S[node]->data, S[node]->ld, tmpM->data, tmpM->ld
                        );
                        H2P_dense_mat_init(&W[node], V[node]->ncol, tmpM->ncol);
                        CBLAS_GEMM(
                            CblasRowMajor, CblasTrans, CblasNoTrans, V[node]->ncol, tmpM->ncol, V[node]->nrow,
                            1.0, V[node]->data, V[node]->ld, tmpM->data, tmpM->ld, 0.0, W[node]->data, W[node]->ld
                        );
                    }  // End of "if (H2_U_node->ld > 0)"
                } else {  // Else of "if (n_child_node == 0)"
                    // (1) Construct the intermediate blocks defined by its children nodes
                    H2P_int_vec_set_capacity(idx0, n_child_node + 1);
                    int *offset = idx0->data;
                    offset[0] = 0;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = node_children[k];
                        offset[k + 1] = offset[k] + HSS_U[child_k]->ncol;
                    }
                    int tmpB_nrow = offset[n_child_node];
                    H2P_dense_mat_p tmpB = mat0;
                    H2P_dense_mat_resize(tmpB, tmpB_nrow + 1, tmpB_nrow);
                    tmpB->nrow--;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = node_children[k];
                        // idx_k = offset(k) : offset(k+1)-1;
                        int idx_k_s = offset[k];
                        int idx_k_len = offset[k + 1] - idx_k_s;
                        for (int l = k + 1; l < n_child_node; l++)
                        {
                            int child_l = node_children[l];
                            // idx_l = offset(l) : offset(l+1)-1;
                            int idx_l_s = offset[l];
                            // B_idx = B_pair2idx(child_k, child_l);
                            int HSS_B_idx = H2P_get_int_CSR_elem(HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, child_k, child_l);
                            ASSERT_PRINTF(HSS_B_idx != 0, "SPDHSS_B{%d, %d} does not exist!\n", child_k, child_l);
                            HSS_B_idx--;
                            H2P_dense_mat_p HSS_B_kl = HSS_B[HSS_B_idx];
                            // tmpB(idx_k, idx_l) = HSS_B{B_idx};
                            DTYPE *tmpB_kl = tmpB->data + idx_k_s * tmpB->ld + idx_l_s;
                            copy_matrix_block(sizeof(DTYPE), HSS_B_kl->nrow, HSS_B_kl->ncol, HSS_B_kl->data, HSS_B_kl->ld, tmpB_kl, tmpB->ld);
                            // tmpB(idx_l, idx_k) = HSS_B{B_idx}';
                            // LAPACK_SYEVD only need uppertriangle, no need to fill the lower triangle part
                            //DTYPE *tmpB_lk = tmpB->data + idx_l_s * tmpB->ld + idx_k_s;
                            //H2P_transpose_dmat(1, HSS_B_kl->nrow, HSS_B_kl->ncol, HSS_B_kl->data, HSS_B_kl->ld, tmpB_lk, tmpB->ld);
                        }
                        // Set the diagonal block to zero
                        for (int l = idx_k_s; l < idx_k_s + idx_k_len; l++)
                        {
                            DTYPE *tmpB_l_ks = tmpB->data + l * tmpB->ld + idx_k_s;
                            memset(tmpB_l_ks, 0, sizeof(DTYPE) * idx_k_len);
                        }
                    }  // End of k loop

                    // (2) Decompose the diagonal matrix
                    H2P_dense_mat_p tmpQ = tmpB;
                    DTYPE *tmpE_diag = tmpB->data + tmpQ->nrow * tmpQ->nrow;
                    // [tmpQ, tmpE] = eig(tmpB);
                    // tmpE_diag = diag(tmpE);
                    info = LAPACK_SYEVD(LAPACK_ROW_MAJOR, 'V', 'U', tmpQ->nrow, tmpQ->data, tmpQ->ld, tmpE_diag);
                    if (info != 0)
                    {
                        ERROR_PRINTF("Node %d intermediate diagonal matrix cannot be diagonalized\n", node);
                        is_SPD = 0;
                        continue;
                    }
                    DTYPE min_diag = 19241112.0;
                    for (int k = 0; k < tmpQ->nrow; k++) min_diag = MIN(min_diag, tmpE_diag[k]);
                    if (min_diag <= -1.0)
                    {
                        ERROR_PRINTF("Node %d intermediate diagonal matrix has eigenvalue %e < -1\n", node, min_diag);
                        ERROR_PRINTF("Source H2 matrix with shifting %.3lf is not SPD\n", shift);
                        is_SPD = 0;
                        continue;
                    }
                    H2P_dense_mat_p tmpM  = mat1;  // tmpM need to be reused later!
                    H2P_dense_mat_p tmpQ1 = mat2;
                    // tmpM = tmpQ * diag((1 + tmpE_diag).^0.5) * tmpQ';
                    #pragma omp simd
                    for (int k = 0; k < tmpQ->nrow; k++)
                        tmpE_diag[k] = DSQRT(1.0 + tmpE_diag[k]);
                    H2P_dense_mat_resize(tmpQ1, tmpQ->nrow, tmpQ->ncol);
                    for (int k = 0; k < tmpQ->nrow; k++)
                    {
                        DTYPE *tmpQ_k  = tmpQ->data  + k * tmpQ->ncol;
                        DTYPE *tmpQ1_k = tmpQ1->data + k * tmpQ->ncol;
                        #pragma omp simd
                        for (int l = 0; l < tmpQ->ncol; l++)
                            tmpQ1_k[l] = tmpQ_k[l] * tmpE_diag[l];
                    }
                    H2P_dense_mat_resize(tmpM, tmpQ->nrow, tmpQ->nrow);
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasTrans, tmpQ->nrow, tmpQ->nrow, tmpQ->nrow,
                        1.0, tmpQ1->data, tmpQ1->ld, tmpQ->data, tmpQ->ld, 0.0, tmpM->data, tmpM->ld
                    );
                    // Minv{node} = tmpQ * diag((1 + tmpE_diag).^-0.5) * tmpQ';
                    #pragma omp simd
                    for (int k = 0; k < tmpQ->nrow; k++)
                        tmpE_diag[k] = 1.0 / tmpE_diag[k];
                    H2P_dense_mat_resize(tmpQ1, tmpQ->nrow, tmpQ->ncol);
                    for (int k = 0; k < tmpQ->nrow; k++)
                    {
                        DTYPE *tmpQ_k  = tmpQ->data  + k * tmpQ->ncol;
                        DTYPE *tmpQ1_k = tmpQ1->data + k * tmpQ->ncol;
                        #pragma omp simd
                        for (int l = 0; l < tmpQ->ncol; l++)
                            tmpQ1_k[l] = tmpQ_k[l] * tmpE_diag[l];
                    }
                    H2P_dense_mat_init(&Minv[node], tmpQ->nrow, tmpQ->nrow);
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasTrans, tmpQ->nrow, tmpQ->nrow, tmpQ->nrow,
                        1.0, tmpQ1->data, tmpQ1->ld, tmpQ->data, tmpQ->ld, 0.0, Minv[node]->data, Minv[node]->ld
                    );
                    // Now mat0 and mat2 can be reused

                    // (3) Construct basis matrix
                    H2P_dense_mat_p *node_Yk = Yk + node * max_level;
                    H2P_int_vec_p tmpYk_idx = idx0;
                    H2P_int_vec_set_capacity(tmpYk_idx, n_child_node);
                    tmpYk_idx->length = n_child_node;
                    for (int l = 0; l < node_level[node]; l++)
                    {
                        for (int k = 0; k < n_child_node; k++)
                        {
                            int child_k = node_children[k];
                            tmpYk_idx->data[k] = child_k * max_level + l;
                        }
                        H2P_dense_mat_p node_Yk_l = node_Yk[l];
                        H2P_dense_mat_vertcat(Yk, tmpYk_idx, node_Yk_l);
                        for (int k = 0; k < n_child_node; k++)
                        {
                            int child_k = node_children[k];
                            int Yk_idx = child_k * max_level + l;
                            H2P_dense_mat_destroy(&Yk[Yk_idx]);
                        }
                    }

                    // tmpY = Minv{node} * Yk{node}{1};
                    H2P_dense_mat_p tmpY = mat0;
                    H2P_dense_mat_resize(tmpY, Minv[node]->nrow + 1, node_Yk[0]->ncol);
                    DTYPE *tau = tmpY->data + Minv[node]->nrow * node_Yk[0]->ncol;
                    tmpY->nrow--;
                    ASSERT_PRINTF(
                        Minv[node]->ncol == node_Yk[0]->nrow,
                        "Node %d: Minv->ncol (%d) mismatch node_Yk[0]->nrow (%d)\n",
                        node, Minv[node]->ncol, node_Yk[0]->nrow
                    );
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv[node]->nrow, tmpY->ncol, Minv[node]->ncol,
                        1.0, Minv[node]->data, Minv[node]->ld, node_Yk[0]->data, node_Yk[0]->ld, 0.0, tmpY->data, tmpY->ld
                    );
                    // tmpQ_ncol = min([size(tmpY), max_rank]);
                    // [tmpQ, ~, ~] = qr(tmpY, 0);
                    // V{node} = tmpQ(:, 1 : tmpQ_ncol);
                    int tmpQ_ncol = MIN(tmpY->nrow, tmpY->ncol);
                    int V_ncol = MIN(tmpQ_ncol, max_rank);
                    tmpQ = tmpY;
                    H2P_int_vec_p jpvt = idx0;
                    H2P_int_vec_set_capacity(jpvt, tmpQ->ncol);
                    memset(jpvt->data, 0, sizeof(int) * tmpQ->ncol);
                    LAPACK_GEQPF(LAPACK_ROW_MAJOR, tmpQ->nrow, tmpQ->ncol, tmpQ->data, tmpQ->ld, jpvt->data, tau);
                    int V_ncol1 = -1;
                    DTYPE stop_diag = DABS(tmpQ->data[0]) * reltol;
                    for (int k = 0; k < V_ncol; k++)
                    {
                        if (DABS(tmpQ->data[k * tmpQ->ld + k]) < stop_diag)
                        {
                            V_ncol1 = k - 1;
                            break;
                        }
                    }
                    if (V_ncol1 > 0) V_ncol = V_ncol1;
                    LAPACK_ORGQR(LAPACK_ROW_MAJOR, tmpQ->nrow, tmpQ_ncol, tmpQ_ncol, tmpQ->data, tmpQ->ld, tau);
                    H2P_dense_mat_init(&V[node], tmpQ->nrow, V_ncol);
                    copy_matrix_block(sizeof(DTYPE), tmpQ->nrow, V_ncol, tmpQ->data, tmpQ->ld, V[node]->data, V[node]->ld);
                    // HSS_U{node} = tmpM * V{node};
                    H2P_dense_mat_init(&HSS_U[node], tmpM->nrow, V[node]->ncol);
                    CBLAS_GEMM(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, tmpM->nrow, V[node]->ncol, tmpM->ncol,
                        1.0, tmpM->data, tmpM->ld, V[node]->data, V[node]->ld, 0.0, HSS_U[node]->data, HSS_U[node]->ld
                    );
                    // Now mat1 can be reused

                    // Yk{node}(1) = [];
                    // for k = 1 : length(Yk{node})
                    //     Yk{node}{k} = V{node}' * Minv{node} * Yk{node}{k};
                    // end
                    int last_k = 0;
                    for (int k = 1; k < max_level; k++)
                    {
                        if (node_Yk[k]->ld == 0) break;  // Empty Yk{node}{k}
                        H2P_dense_mat_p node_Yk_k0 = node_Yk[k - 1];
                        H2P_dense_mat_p node_Yk_k  = node_Yk[k];
                        H2P_dense_mat_resize(tmpM, Minv[node]->nrow, node_Yk_k->ncol);
                        ASSERT_PRINTF(
                            Minv[node]->ncol == node_Yk_k->nrow,
                            "Node %d: Minv->ncol (%d) mismatch node_Yk[%d]->nrow (%d)",
                            node, Minv[node]->ncol, k, node_Yk_k->nrow
                        );
                        CBLAS_GEMM(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv[node]->nrow, node_Yk_k->ncol, Minv[node]->ncol,
                            1.0, Minv[node]->data, Minv[node]->ld, node_Yk_k->data, node_Yk_k->ld, 0.0, tmpM->data, tmpM->ld
                        );
                        H2P_dense_mat_resize(node_Yk_k0, V[node]->ncol, tmpM->ncol);
                        ASSERT_PRINTF(
                            V[node]->nrow == Minv[node]->nrow,
                            "Node %d: V->nrow (%d) mismatch tmpM->ncol (%d)\n", 
                            node, V[node]->nrow, Minv[node]->nrow
                        );
                        CBLAS_GEMM(
                            CblasRowMajor, CblasTrans, CblasNoTrans, V[node]->ncol, tmpM->ncol, V[node]->nrow,
                            1.0, V[node]->data, V[node]->ld, tmpM->data, tmpM->ld, 0.0, node_Yk_k0->data, node_Yk_k0->ld
                        );
                        last_k = k;
                    }  // End of k loop
                    H2P_dense_mat_destroy(&node_Yk[last_k]);
                    // if (~isempty(H2_U{node}))
                    //     child_node = children(node, 1 : n_child_node);
                    //     tmpW = blkdiag(W{child_node});
                    //     W{node} = V{node}' * (Minv{node} * (tmpW * H2_U{node}));
                    // end
                    H2P_dense_mat_p H2_U_node = H2_U[node];
                    if (H2_U_node->ld > 0)
                    {
                        H2P_dense_mat_p tmpM0 = mat1;
                        H2P_dense_mat_p tmpM1 = mat2;
                        // Don't use blkdiag, directly multiple each child node's W with H2_U{node}
                        int tmpW_nrow = 0, tmpW_ncol = 0;
                        for (int k = 0; k < n_child_node; k++)
                        {
                            int child_k = node_children[k];
                            tmpW_nrow += W[child_k]->nrow;
                            tmpW_ncol += W[child_k]->ncol;
                        }
                        ASSERT_PRINTF(
                            tmpW_ncol == H2_U_node->nrow,
                            "Node %d: tmpW->ncol (%d) mismatch H2_U->nrow (%d)\n",
                            node, tmpW_ncol, H2_U_node->nrow
                        );
                        H2P_dense_mat_resize(tmpM0, tmpW_nrow, H2_U_node->ncol);
                        tmpW_nrow = 0;
                        tmpW_ncol = 0;
                        for (int k = 0; k < n_child_node; k++)
                        {
                            int child_k = node_children[k];
                            H2P_dense_mat_p W_k = W[child_k];
                            DTYPE *tmpM0_k_row = tmpM0->data + tmpW_nrow * tmpM0->ld;
                            DTYPE *H2_U_k_col  = H2_U_node->data + tmpW_ncol * H2_U_node->ld;
                            CBLAS_GEMM(
                                CblasRowMajor, CblasNoTrans, CblasNoTrans, W_k->nrow, H2_U_node->ncol, W_k->ncol,
                                1.0, W_k->data, W_k->ld, H2_U_k_col, H2_U_node->ld, 0.0, tmpM0_k_row, tmpM0->ld
                            );
                            tmpW_nrow += W_k->nrow;
                            tmpW_ncol += W_k->ncol;
                            H2P_dense_mat_destroy(&W[child_k]);
                        }
                        // The rest GEMM 
                        H2P_dense_mat_resize(tmpM1, Minv[node]->nrow, tmpM0->ncol);
                        ASSERT_PRINTF(
                            Minv[node]->ncol == tmpM0->nrow,
                            "Node %d: Minv->ncol (%d) mismatch tmpM0->nrow (%d)\n",
                            node, Minv[node]->ncol, tmpM0->nrow
                        );
                        CBLAS_GEMM(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans, Minv[node]->nrow, tmpM0->ncol, Minv[node]->ncol, 
                            1.0, Minv[node]->data, Minv[node]->ld, tmpM0->data, tmpM0->ld, 0.0, tmpM1->data, tmpM1->ld
                        );
                        H2P_dense_mat_init(&W[node], V[node]->ncol, tmpM1->ncol);
                        ASSERT_PRINTF(
                            V[node]->nrow == tmpM1->nrow,
                            "Node %d: V->nrow (%d) mismatch tmpM1->nrow (%d)\n",
                            node, V[node]->nrow, tmpM1->nrow
                        );
                        CBLAS_GEMM(
                            CblasRowMajor, CblasTrans, CblasNoTrans, V[node]->ncol, tmpM1->ncol, V[node]->nrow,
                            1.0, V[node]->data, V[node]->ld, tmpM1->data, tmpM1->ld, 0.0, W[node]->data, W[node]->ld
                        );
                    }  // End of "if (H2_U_node->ld > 0)"
                }  // End of "if (n_child_node == 0)"
            }  // End of j loop
        }  // End of "#pragma omp parallel"
        et = get_wtime_sec();
        build_U_t += et - st;
        printf("SPDHSS build at %d level: U build %f.\n", i, et - st);
        fflush(stdout);

        st = get_wtime_sec();
        // Build new B matrices
        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_HSS_Bij_n_pair; j++)
            {
                int node0 = level_i_HSS_Bij_pairs[2 * j];
                int node1 = level_i_HSS_Bij_pairs[2 * j + 1];
                H2P_SPDHSS_H2_calc_HSS_Bij(
                    h2mat, node0, node1, tid, S, V, W, Minv, 
                    HSS_B_p2i_rowptr, HSS_B_p2i_colidx, HSS_B_p2i_val, HSS_B
                );
            }  // End of j loop
        }  // End of "#pragma omp parallel"
        et = get_wtime_sec();
        build_B_t += et - st;
        printf("SPDHSS build at %d level: B build %f.\n", i, et - st);
        fflush(stdout);
    }  // End of i loop

    // 7. Wrap the new SPD HSS matrix
    H2P_SPDHSS_H2_wrap_new_HSS(
        h2mat, HSS_U, HSS_B, HSS_D, HSS_B_p2i_rowptr, HSS_B_p2i_colidx,
        HSS_B_p2i_val, HSS_D_pair2idx, hssmat_
    );
    (*hssmat_)->timers[U_BUILD_TIMER_IDX] = build_U_t;
    (*hssmat_)->timers[B_BUILD_TIMER_IDX] = build_B_t;
    (*hssmat_)->timers[D_BUILD_TIMER_IDX] = build_D_t;

    #ifdef __linux__
    // Restore default value
    mallopt(M_MMAP_THRESHOLD, 128 * 1024);
    #endif

    // 8. Delete intermediate arrays and matrices
    for (int i = 0; i < n_level; i++)
        H2P_int_vec_destroy(&level_HSS_Bij_pairs[i]);
    for (int i = 0; i < n_node; i++)
    {
        H2P_dense_mat_destroy(&S[i]);
        H2P_dense_mat_destroy(&V[i]);
        H2P_dense_mat_destroy(&W[i]);
        H2P_dense_mat_destroy(&Minv[i]);
    }
    for (int i = 0; i < n_HSS_Bij_pair; i++)
        H2P_dense_mat_destroy(&HSS_B[i]);
    for (int i = 0; i < n_leaf_node; i++)
        H2P_dense_mat_destroy(&HSS_D[i]);
    for (int i = 0; i < n_node * max_level; i++)
        H2P_dense_mat_destroy(&Yk[i]);
    free(level_HSS_Bij_pairs);
    free(S);
    free(V);
    free(W);
    free(Minv);
    free(HSS_B);
    free(HSS_D);
    free(HSS_D_pair2idx);
    free(HSS_B_p2i_rowptr);
    free(HSS_B_p2i_colidx);
    free(HSS_B_p2i_val);
}
