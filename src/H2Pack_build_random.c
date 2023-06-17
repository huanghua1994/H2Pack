#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_build_random.h"
#include "H2Pack_ID_compress.h"
#include "H2Pack_utils.h"
#include "utils.h"

// Based on: https://github.com/xinxing02/H2Pack-Matlab/blob/master/auxiliary_tools/H2__admissible_nodelist.m
// For each node i at the k-th level, find admissible node pairs containing node j if 
// it satisfies either of the following conditions:
// 1. j is at k-th level and (i, j) is an admissible pair --> one pair of ancestors of i
//    and j are in the reduced admissible pair set.
// 2. j is a leaf node at a higher level (i's parent level).
// 3. j is at lower level, this only happens when i is a leaf node. In this case, (i, j)
//    should be in the reduced admissible pair set.
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameters:
//   *node_adm_cnt_  : Size h2pack->n_node, number of admissible pairs for each node
//   *nod_adm_pairs_ : Size h2pack->n_node * h2pack->n_node, admissible pairs for each node
void H2P_calc_adm_pairs(H2Pack_p h2pack, int **node_adm_cnt_, int **nod_adm_pairs_)
{
    int n_node        = h2pack->n_node;
    int max_child     = h2pack->max_child;
    int max_level     = h2pack->max_level;
    int n_leaf_node   = h2pack->n_leaf_node;
    int *parent       = h2pack->parent;
    int *children     = h2pack->children;
    int *n_child      = h2pack->n_child;
    int *level_nodes  = h2pack->level_nodes;
    int *level_n_node = h2pack->level_n_node;
    int min_adm_level, n_r_adm_pair, *r_adm_pairs; 
    if (h2pack->is_HSS)
    {
        min_adm_level = h2pack->HSS_min_adm_level;
        n_r_adm_pair  = h2pack->HSS_n_r_adm_pair;
        r_adm_pairs   = h2pack->HSS_r_adm_pairs;
    } else {
        min_adm_level = h2pack->min_adm_level;
        n_r_adm_pair  = h2pack->n_r_adm_pair;
        r_adm_pairs   = h2pack->r_adm_pairs;
    }

    int *node_adm_cnt   = (int *) malloc(sizeof(int) * n_node);
    int *node_adm_pairs = (int *) malloc(sizeof(int) * n_node * n_node);
    ASSERT_PRINTF(
        node_adm_cnt != NULL && node_adm_pairs != NULL, 
        "Failed to allocate work buffers for H2P_calc_adm_pairs().\n"
    );

    // 1. Put all reduced admissible pairs into node_adm_pairs (for cases 2 and 3)
    memset(node_adm_cnt, 0, sizeof(int) * n_node);
    for (int i = 0; i < n_r_adm_pair; i++)
    {
        int node0 = r_adm_pairs[i * 2];
        int node1 = r_adm_pairs[i * 2 + 1];
        node_adm_pairs[node0 * n_node + node_adm_cnt[node0]] = node1;
        node_adm_pairs[node1 * n_node + node_adm_cnt[node1]] = node0;
        node_adm_cnt[node0]++;
        node_adm_cnt[node1]++;
    }

    // 2. Find admissible pairs inherent from parent (case 1)
    for (int i = min_adm_level; i <= max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            int node_parent = parent[node];
            for (int k = 0; k < node_adm_cnt[node_parent]; k++)
            {
                int node_k = node_adm_pairs[node_parent * n_node + k];
                if (n_child[node_k] == 0)
                {
                    node_adm_pairs[node * n_node + node_adm_cnt[node]] = node_k;
                    node_adm_cnt[node]++;
                } else {
                    int *node_k_children = children + node_k * max_child;
                    for (int l = 0; l < n_child[node_k]; l++)
                    {
                        int node_l = node_k_children[l];
                        node_adm_pairs[node * n_node + node_adm_cnt[node]] = node_l;
                        node_adm_cnt[node]++;
                    }
                }  // End of "if (n_child[node_k] == 0)"
            }  // End of k loop
        }  // End of j loop 
    }  // End of i loop

    *node_adm_cnt_ = node_adm_cnt;
    *nod_adm_pairs_ = node_adm_pairs;
}

// Estimated the number of proxy points for each layer,
// simplified from H2P_generate_proxy_point_nlayer()
void H2P_estimate_num_proxy_points(H2Pack_p h2pack, int **num_pp_)
{
    int pt_dim        = h2pack->pt_dim;
    int xpt_dim       = h2pack->xpt_dim;
    int krnl_dim      = h2pack->krnl_dim;
    int max_level     = h2pack->max_level;
    int n_leaf_node   = h2pack->n_leaf_node;
    int n_thread      = h2pack->n_thread;
    int *level_nodes  = h2pack->level_nodes;
    void *krnl_param  = h2pack->krnl_param;
    kernel_eval_fptr krnl_eval = h2pack->krnl_eval;
    int min_adm_level;
    DTYPE alpha_, reltol = h2pack->QR_stop_tol;
    if (h2pack->is_HSS)
    {
        min_adm_level = h2pack->HSS_min_adm_level;
        alpha_ = ALPHA_HSS;
    } else {
        min_adm_level = h2pack->min_adm_level;
        alpha_ = ALPHA_H2;
    }
    
    int X0_size = 2000, Y0_lsize = 4000, Y0_size = 6 * Y0_lsize;
    H2P_dense_mat_p X0_coord, Y0_coord, Y1_coord;
    H2P_dense_mat_p tmpA, tmpA1, QR_buff;
    H2P_int_vec_p   skel_idx, ID_buff, idx_buf2;
    H2P_dense_mat_init(&X0_coord,  pt_dim,  X0_size);
    H2P_dense_mat_init(&Y0_coord,  pt_dim,  Y0_size);
    H2P_dense_mat_init(&Y1_coord,  pt_dim,  Y0_size);
    H2P_dense_mat_init(&tmpA,      X0_size * krnl_dim, Y0_size * krnl_dim);
    H2P_dense_mat_init(&tmpA1,     X0_size * krnl_dim, Y0_size * krnl_dim);
    H2P_dense_mat_init(&QR_buff,   2 * Y0_size, 1);
    H2P_int_vec_init(&skel_idx, X0_size);
    H2P_int_vec_init(&ID_buff,  4 * Y0_size);
    H2P_int_vec_init(&idx_buf2, Y0_size);
    int *num_pp = (int *) malloc(sizeof(int) * (max_level + 1));
    memset(num_pp, 0, sizeof(int) * (max_level + 1));
    DTYPE maxL = h2pack->root_enbox[pt_dim];
    for (int i = min_adm_level; i <= max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int node = level_i_nodes[0];
        DTYPE *enbox = h2pack->enbox + node * pt_dim * 2;
        // Assuming all enboxes are cubic
        DTYPE L1 = enbox[pt_dim];
        DTYPE L2 = (1.0 + 2.0 * alpha_) * L1;
        DTYPE L3 = (1.0 + 8.0 * alpha_) * L1;

        srand(1924);
        srand48(1112);

        // 1. Generate initial candidate points in X and Y
        int n_layer = (int) DROUND((L3 - L2) / L1);
        int Y0_size = n_layer * Y0_lsize;
        DTYPE Y0_layer_width = (L3 - L2) / (DTYPE) n_layer;
        H2P_dense_mat_resize(X0_coord, pt_dim, X0_size);
        H2P_dense_mat_resize(Y0_coord, pt_dim, Y0_size);
        H2P_gen_coord_in_ring(X0_size, pt_dim, 0.0, L1, X0_coord->data, X0_coord->ld);
        for (int i = 0; i < n_layer; i++)
        {
            DTYPE layer_L0 = L2 + Y0_layer_width * (DTYPE) i;
            DTYPE layer_L1 = L2 + Y0_layer_width * (DTYPE) (i + 1);
            H2P_gen_coord_in_ring(Y0_lsize, pt_dim, layer_L0, layer_L1, Y0_coord->data + i * Y0_lsize, Y0_coord->ld);
        }

        // 2. Select skeleton points in domain X, use sparsity + randomize to reduce the ID cost
        // (1) Generate the kernel matrix
        H2P_dense_mat_resize(tmpA, X0_coord->ncol * krnl_dim, Y0_coord->ncol * krnl_dim);
        H2P_eval_kernel_matrix_OMP(
            krnl_eval, krnl_param, 
            X0_coord->data, X0_coord->ld, X0_coord->ncol,  
            Y0_coord->data, Y0_coord->ld, Y0_coord->ncol, 
            tmpA->data, tmpA->ld, n_thread
        );
        // (2) Generate sparse random matrix and multiply with the kernel matrix to get a reduced matrix
        H2P_int_vec_p   rndmat_idx = ID_buff;
        H2P_dense_mat_p rndmat_val = QR_buff;
        int max_nnz_col = 32;
        H2P_gen_rand_sparse_mat_trans(max_nnz_col, tmpA->ncol, tmpA->nrow, rndmat_val, rndmat_idx);
        H2P_dense_mat_resize(tmpA1, tmpA->nrow, tmpA->nrow);
        H2P_calc_sparse_mm_trans_OMP(
            tmpA->nrow, tmpA->nrow, tmpA->ncol, rndmat_val, rndmat_idx,
            tmpA->data, tmpA->ld, tmpA1->data, tmpA1->ld, n_thread
        );
        H2P_dense_mat_normalize_columns(tmpA1, QR_buff);
        // (3) Calculate ID approximation on the reduced matrix and select skeleton points in X
        if (krnl_dim == 1)
        {
            H2P_dense_mat_resize(QR_buff, tmpA1->nrow, 1);
        } else {
            int QR_buff_size = (2 * krnl_dim + 2) * tmpA1->ncol + (krnl_dim + 1) * tmpA1->nrow;
            H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
        }
        H2P_int_vec_set_capacity(ID_buff, 4 * tmpA1->nrow);
        DTYPE reltol_ = reltol * 1e-2;
        H2P_ID_compress(
            tmpA1, QR_REL_NRM, &reltol_, NULL, skel_idx, 
            n_thread, QR_buff->data, ID_buff->data, krnl_dim
        );
        H2P_dense_mat_select_columns(X0_coord, skel_idx);
        H2P_dense_mat_p Xp_coord = X0_coord;

        // 3. Create a random sparse sampling matrix S, re-index its column indices
        H2P_dense_mat_p spS_valbuf = QR_buff;   // QR_buff is not used now, use it as a buffer
        H2P_int_vec_p   spS_idxbuf = ID_buff;   // ID_buff is not used now, use it as a buffer
        int spS_k = Y0_coord->ncol * krnl_dim;
        int spS_n = Xp_coord->ncol * krnl_dim;
        H2P_gen_rand_sparse_mat_trans(max_nnz_col, spS_k, spS_n, spS_valbuf, spS_idxbuf);
        int *spS_rowptr = spS_idxbuf->data;
        int *spS_colidx = spS_rowptr + (spS_n + 1);
        H2P_int_vec_set_capacity(idx_buf2, spS_k * 3);
        int *selected_idx = idx_buf2->data;
        int *spS_col_flag = selected_idx + spS_k;
        int *spS_col_map  = spS_col_flag + spS_k;
        memset(spS_col_flag, 0, sizeof(int) * spS_k);
        for (int k = 0; k < spS_rowptr[spS_n]; k++) spS_col_flag[spS_colidx[k]] = 1;
        int uniq_col_idx = 0;
        for (int k = 0; k < spS_k; k++)
        {
            if (spS_col_flag[k] == 1)
            {
                spS_col_map[k] = uniq_col_idx;
                selected_idx[uniq_col_idx] = k;
                uniq_col_idx++;
            }
        }
        for (int k = 0; k < spS_rowptr[spS_n]; k++) spS_colidx[k] = spS_col_map[spS_colidx[k]];

        // 4. Gather Y0 coordinates corresponding to selected column indices
        //    and build the kernel matrix block
        idx_buf2->length = uniq_col_idx;
        H2P_dense_mat_resize(Y1_coord, xpt_dim, uniq_col_idx);
        H2P_gather_matrix_columns(
            Y0_coord->data, Y0_coord->ld, Y1_coord->data, Y1_coord->ld,
            xpt_dim, selected_idx, uniq_col_idx
        );
        int A1_nrow = Xp_coord->ncol * krnl_dim;
        int A1_ncol = Y1_coord->ncol * krnl_dim;
        H2P_dense_mat_resize(tmpA1, A1_nrow, A1_ncol);
        krnl_eval(
            Xp_coord->data, Xp_coord->ncol, Xp_coord->ld,
            Y1_coord->data, Y1_coord->ncol, Y1_coord->ld, 
            krnl_param, tmpA1->data, tmpA1->ld
        );

        // 5. Apply the sampling matrix to the kernel matrix block and normalize its columns
        H2P_dense_mat_p AS = Y1_coord;  // Y1_coord is no longer needed
        H2P_dense_mat_resize(AS, A1_nrow, spS_n);
        H2P_calc_sparse_mm_trans_OMP(
            A1_nrow, spS_n, A1_ncol, spS_valbuf, spS_idxbuf, 
            tmpA1->data, tmpA1->ld, AS->data, AS->ld, n_thread
        );
        H2P_dense_mat_normalize_columns(AS, QR_buff);

        // 6. Calculate the singular values of AS and get the rank
        H2P_dense_mat_p sv = tmpA1, superb = QR_buff;
        H2P_dense_mat_resize(sv, 1, AS->ncol);
        H2P_dense_mat_resize(superb, 1, AS->ncol);
        LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR, 'N', 'N', AS->nrow, AS->ncol, AS->data, AS->ld, 
            sv->data, NULL, 1, NULL, 1, superb->data
        );
        int cnt = 0;
        DTYPE tol = 0;
        for (int j = 0; j < AS->ncol; j++)
            if (sv->data[j] > tol) tol = sv->data[j];
        tol *= reltol;
        for (int j = 0; j < AS->ncol; j++)
            if (sv->data[j] >= tol) cnt++;
        num_pp[i] = cnt * 11 / 10;  // Give it some extra space
    }  // End of i loop

    H2P_dense_mat_destroy(&X0_coord);
    H2P_dense_mat_destroy(&Y0_coord);
    H2P_dense_mat_destroy(&Y1_coord);
    H2P_dense_mat_destroy(&tmpA);
    H2P_dense_mat_destroy(&tmpA1);
    H2P_dense_mat_destroy(&QR_buff);
    H2P_int_vec_destroy(&skel_idx);
    H2P_int_vec_destroy(&ID_buff);
    H2P_int_vec_destroy(&idx_buf2);
    *num_pp_ = num_pp;
}

void H2P_build_H2_UJ_random(H2Pack_p h2pack)
{
    int    xpt_dim        = h2pack->xpt_dim;
    int    krnl_dim       = h2pack->krnl_dim;
    int    n_node         = h2pack->n_node;
    int    max_child      = h2pack->max_child;
    int    n_leaf_node    = h2pack->n_leaf_node;
    int    n_point        = h2pack->n_point;
    int    n_thread       = h2pack->n_thread;
    int    stop_type      = h2pack->QR_stop_type;
    int    max_level      = h2pack->max_level;
    int    min_adm_level  = (h2pack->is_HSS) ? h2pack->HSS_min_adm_level : h2pack->min_adm_level;
    int    *children      = h2pack->children;
    int    *n_child       = h2pack->n_child;
    int    *level_nodes   = h2pack->level_nodes;
    int    *level_n_node  = h2pack->level_n_node;
    int    *node_level    = h2pack->node_level;
    int    *pt_cluster    = h2pack->pt_cluster;
    DTYPE  *coord         = h2pack->coord;
    size_t *mat_size      = h2pack->mat_size;
    void   *krnl_param    = h2pack->krnl_param;
    H2P_thread_buf_p *thread_buf = h2pack->tb;
    kernel_eval_fptr krnl_eval   = h2pack->krnl_eval;

    DTYPE QR_stop_tol = h2pack->QR_stop_tol * 1e-2;
    void *stop_param = NULL;
    if (stop_type == QR_RANK) 
        stop_param = &h2pack->QR_stop_rank;
    if ((stop_type == QR_REL_NRM) || (stop_type == QR_ABS_NRM))
        stop_param = &QR_stop_tol;
    
    // 1. Calculate all admissible pairs for each node
    int *node_adm_cnt = NULL, *node_adm_pairs = NULL;
    H2P_calc_adm_pairs(h2pack, &node_adm_cnt, &node_adm_pairs);

    // 2. Allocate U and J
    h2pack->n_UJ = n_node;
    h2pack->U       = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    h2pack->J       = (H2P_int_vec_p*)   malloc(sizeof(H2P_int_vec_p)   * n_node);
    h2pack->J_coord = (H2P_dense_mat_p*) malloc(sizeof(H2P_dense_mat_p) * n_node);
    ASSERT_PRINTF(h2pack->U       != NULL, "Failed to allocate %d U matrices\n", n_node);
    ASSERT_PRINTF(h2pack->J       != NULL, "Failed to allocate %d J matrices\n", n_node);
    ASSERT_PRINTF(h2pack->J_coord != NULL, "Failed to allocate %d J_coord auxiliary matrices\n", n_node);
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        h2pack->U[i]       = NULL;
        h2pack->J[i]       = NULL;
        h2pack->J_coord[i] = NULL;
    }
    H2P_dense_mat_p *U       = h2pack->U;
    H2P_int_vec_p   *J       = h2pack->J;
    H2P_dense_mat_p *J_coord = h2pack->J_coord;

    // 3. Initialize row indices associated with clusters for leaf nodes
    for (int i = 0; i < n_node; i++)
    {
        if (n_child[i] > 0) continue;
        int pt_s = pt_cluster[i * 2];
        int pt_e = pt_cluster[i * 2 + 1];
        int node_npt = pt_e - pt_s + 1;
        H2P_int_vec_init(&J[i], node_npt);
        for (int k = 0; k < node_npt; k++)
            J[i]->data[k] = pt_s + k;
        J[i]->length = node_npt;
        H2P_dense_mat_init(&J_coord[i], xpt_dim, node_npt);
        copy_matrix(
            sizeof(DTYPE), xpt_dim, node_npt, 
            coord + pt_s, n_point, J_coord[i]->data, node_npt, 0
        );
    }

    // 4. Calculate the proxy point size for each layer as the 
    //    upper bound of the rank of the kernel matrix block
    int *num_pp = NULL;
    double est_pp_st = get_wtime_sec();
    H2P_estimate_num_proxy_points(h2pack, &num_pp);
    double est_pp_et = get_wtime_sec();
    if (h2pack->print_dbginfo == 1)
    {
        printf("H2Pack estimate proxy point size time: %.3lf\n", est_pp_et - est_pp_st);
        for (int i = 0; i <= max_level; i++)
            printf("Level %d proxy point size: %d\n", i, num_pp[i]);
        printf("\n");
    }

    double *t_gather = (double*) malloc(sizeof(double) * n_thread);
    double *t_krnl   = (double*) malloc(sizeof(double) * n_thread);
    double *t_spmm   = (double*) malloc(sizeof(double) * n_thread);
    double *t_id     = (double*) malloc(sizeof(double) * n_thread);
    
    // 4. Construct U for nodes whose level is not smaller than min_adm_level.
    //    min_adm_level is the highest level that still has admissible blocks.
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2P_dense_mat_p A_block    = thread_buf[tid]->mat0;
        H2P_dense_mat_p skel_coord = thread_buf[tid]->mat1;
        H2P_dense_mat_p QR_buff    = thread_buf[tid]->mat2;
        H2P_dense_mat_p adm_coord;
        H2P_int_vec_p   sub_idx    = thread_buf[tid]->idx0;
        H2P_int_vec_p   ID_buff    = thread_buf[tid]->idx1;
        H2P_int_vec_p   idx_buf2;

        H2P_int_vec_init(&idx_buf2, 1024);
        H2P_dense_mat_init(&adm_coord, xpt_dim, 1024);

        double tid_gather = 0.0, tid_krnl = 0.0, tid_spmm = 0.0, tid_id = 0.0;
        double tid_s, tid_e;

        thread_buf[tid]->timer = -get_wtime_sec();
        for (int i = max_level; i >= min_adm_level; i--)
        {
            int *level_i_nodes = level_nodes + i * n_leaf_node;
            int level_i_n_node = level_n_node[i];

            // (1) Update row indices associated with clusters for current node
            tid_s = get_wtime_sec();
            #pragma omp barrier
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                if (n_child[node] > 0)
                {
                    // Non-leaf nodes, gather row indices from children nodes
                    int n_child_node = n_child[node];
                    int *child_nodes = children + node * max_child;
                    int J_child_size = 0;
                    for (int i_child = 0; i_child < n_child_node; i_child++)
                    {
                        int i_child_node = child_nodes[i_child];
                        J_child_size += J[i_child_node]->length;
                    }
                    H2P_int_vec_init(&J[node], J_child_size);
                    for (int i_child = 0; i_child < n_child_node; i_child++)
                    {
                        int i_child_node = child_nodes[i_child];
                        H2P_int_vec_concatenate(J[node], J[i_child_node]);
                    }
                }  // End of "if (n_child[node] > 0)"
            }  // End of j loop
            tid_e = get_wtime_sec();
            tid_gather += tid_e - tid_s;

            #pragma omp barrier
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node  = level_i_nodes[j];
                int level = node_level[node];

                // (2) Gather current node's skeleton points (== all children nodes' skeleton points)
                tid_s = get_wtime_sec();
                H2P_dense_mat_resize(skel_coord, xpt_dim, J[node]->length);
                if (n_child[node] == 0)
                {
                    copy_matrix(
                        sizeof(DTYPE), xpt_dim, J[node]->length, J_coord[node]->data, 
                        J_coord[node]->ld, skel_coord->data, skel_coord->ld, 0
                    );
                } else {
                    int n_child_node = n_child[node];
                    int *child_nodes = children + node * max_child;
                    int J_child_size = 0;
                    for (int i_child = 0; i_child < n_child_node; i_child++)
                    {
                        int i_child_node = child_nodes[i_child];
                        int src_ld = J_coord[i_child_node]->ncol;
                        int dst_ld = skel_coord->ncol;
                        DTYPE *src_mat = J_coord[i_child_node]->data;
                        DTYPE *dst_mat = skel_coord->data + J_child_size; 
                        copy_matrix(sizeof(DTYPE), xpt_dim, src_ld, src_mat, src_ld, dst_mat, dst_ld, 0);
                        J_child_size += J[i_child_node]->length;
                    }
                }  // End of "if (level == 0)"

                // (3) Gather all skeleton point indices from admissible pairs
                H2P_int_vec_p adm_J_set = sub_idx;  // sub_idx is not used now, use it as a buffer
                adm_J_set->length = 0;
                for (int k = 0; k < node_adm_cnt[node]; k++)
                {
                    int node_k = node_adm_pairs[node * n_node + k];
                    H2P_int_vec_concatenate(adm_J_set, J[node_k]);
                }
                int adm_J_set_size = adm_J_set->length;
                tid_e = get_wtime_sec();
                tid_gather += tid_e - tid_s;

                // (4) Create a random sparse sampling matrix S, re-index its column indices
                tid_s = get_wtime_sec();
                H2P_dense_mat_p spS_valbuf = QR_buff;  // QR_buff is not used now, use it as a buffer
                H2P_int_vec_p   spS_idxbuf = ID_buff;  // ID_buff is not used now, use it as a buffer
                int max_nnz_col = 32;
                int spS_k = adm_J_set_size;
                int spS_n = (skel_coord->ncol * krnl_dim < num_pp[level]) ? skel_coord->ncol * krnl_dim : num_pp[level];
                H2P_gen_rand_sparse_mat_trans(max_nnz_col, spS_k, spS_n, spS_valbuf, spS_idxbuf);
                int *spS_rowptr = spS_idxbuf->data;
                int *spS_colidx = spS_rowptr + (spS_n + 1);
                H2P_int_vec_set_capacity(idx_buf2, spS_k * 3);
                int *selected_idx = idx_buf2->data;
                int *spS_col_flag = selected_idx + spS_k;
                int *spS_col_map  = spS_col_flag + spS_k;
                memset(spS_col_flag, 0, sizeof(int) * spS_k);
                for (int k = 0; k < spS_rowptr[spS_n]; k++) spS_col_flag[spS_colidx[k]] = 1;
                int uniq_col_idx = 0;
                for (int k = 0; k < spS_k; k++)
                {
                    if (spS_col_flag[k] == 1)
                    {
                        spS_col_map[k] = uniq_col_idx;
                        selected_idx[uniq_col_idx] = k;
                        uniq_col_idx++;
                    }
                }
                for (int k = 0; k < spS_rowptr[spS_n]; k++) spS_colidx[k] = spS_col_map[spS_colidx[k]];
                tid_e = get_wtime_sec();
                tid_spmm += tid_e - tid_s;

                // (5) Map the selected column indices to the corresponding point indices 
                //     and gather corresponding coordinates. adm_J_set is no longer needed
                //     after this step, sub_idx is ready to be used as a buffer.
                tid_s = get_wtime_sec();
                for (int k = 0; k < uniq_col_idx; k++) selected_idx[k] = adm_J_set->data[selected_idx[k]];
                idx_buf2->length = uniq_col_idx;
                H2P_dense_mat_resize(adm_coord, xpt_dim, uniq_col_idx);
                H2P_gather_matrix_columns(
                    coord, n_point, adm_coord->data, uniq_col_idx,
                    xpt_dim, selected_idx, uniq_col_idx
                );
                tid_e = get_wtime_sec();
                tid_gather += tid_e - tid_s;

                // (6) Build the kernel matrix block and apply the sampling matrix.
                //     spS_valbuf and spS_idxbuf are no longer needed after this step,
                //     QR_buff and ID_buff are ready to be used as buffers.
                int A_blk_nrow = skel_coord->ncol * krnl_dim;
                int A_blk_ncol = adm_coord->ncol  * krnl_dim;
                tid_s = get_wtime_sec();
                H2P_dense_mat_resize(A_block, A_blk_nrow, A_blk_ncol);
                krnl_eval(
                    skel_coord->data, skel_coord->ncol, skel_coord->ld,
                    adm_coord->data,  adm_coord->ncol,  adm_coord->ld, 
                    krnl_param, A_block->data, A_block->ld
                );
                tid_e = get_wtime_sec();
                tid_krnl += tid_e - tid_s;
                tid_s = get_wtime_sec();
                H2P_dense_mat_p AS = adm_coord;  // adm_coord is no longer needed
                H2P_dense_mat_resize(AS, A_blk_nrow, spS_n);
                H2P_calc_sparse_mm_trans_OMP(
                    A_blk_nrow, spS_n, A_blk_ncol, spS_valbuf, spS_idxbuf, 
                    A_block->data, A_block->ld, AS->data, AS->ld, 1
                );
                H2P_dense_mat_normalize_columns(AS, QR_buff);
                tid_e = get_wtime_sec();
                tid_spmm += tid_e - tid_s;

                // (7) ID compress of AS
                // Note: AS is transposed in ID compress, be careful when calculating the buffer size
                tid_s = get_wtime_sec();
                if (krnl_dim == 1)
                {
                    H2P_dense_mat_resize(QR_buff, AS->nrow, 1);
                } else {
                    int QR_buff_size = (2 * krnl_dim + 2) * AS->ncol + (krnl_dim + 1) * AS->nrow;
                    H2P_dense_mat_resize(QR_buff, QR_buff_size, 1);
                }
                H2P_int_vec_set_capacity(ID_buff, 4 * AS->nrow);
                H2P_ID_compress(
                    AS, stop_type, stop_param, &U[node], sub_idx, 
                    1, QR_buff->data, ID_buff->data, krnl_dim
                );
                tid_e = get_wtime_sec();
                tid_id += tid_e - tid_s;
                
                // (8) Choose the skeleton points of this node
                tid_s = get_wtime_sec();
                for (int k = 0; k < sub_idx->length; k++)
                    J[node]->data[k] = J[node]->data[sub_idx->data[k]];
                J[node]->length = sub_idx->length;
                H2P_dense_mat_init(&J_coord[node], xpt_dim, sub_idx->length);
                H2P_gather_matrix_columns(
                    coord, n_point, J_coord[node]->data, J[node]->length, 
                    xpt_dim, J[node]->data, J[node]->length
                );
                tid_e = get_wtime_sec();
                tid_gather += tid_e - tid_s;
            }  // End of j loop
        }  // End of i loop
        thread_buf[tid]->timer += get_wtime_sec();

        H2P_int_vec_destroy(&idx_buf2);
        H2P_dense_mat_destroy(&adm_coord);
        t_gather[tid] = tid_gather;
        t_krnl[tid]   = tid_krnl;
        t_spmm[tid]   = tid_spmm;
        t_id[tid]     = tid_id;
    }  // End of "#pragma omp parallel num_thread(n_thread)"
    
    if (h2pack->print_dbginfo == 1)
    {
        printf("tid, gather, krnl, spmm, ID\n");
        for (int i = 0; i < n_thread; i++)
            printf("%2d, %.3lf, %.3lf, %.3lf, %.3lf\n", i, t_gather[i], t_krnl[i], t_spmm[i], t_id[i]);
    }

    if (h2pack->print_timers == 1)
    {
        double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
        for (int i = 0; i < n_thread; i++)
        {
            double thread_i_timer = thread_buf[i]->timer;
            avg_t += thread_i_timer;
            max_t = MAX(max_t, thread_i_timer);
            min_t = MIN(min_t, thread_i_timer);
        }
        avg_t /= (double) n_thread;
        INFO_PRINTF("Build U: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    }

    // 3. Initialize other not touched U J & add statistic info
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        if (U[i] == NULL)
        {
            H2P_dense_mat_init(&U[i], 1, 1);
            U[i]->nrow = 0;
            U[i]->ncol = 0;
            U[i]->ld   = 0;
        } else {
            mat_size[U_SIZE_IDX]      += U[i]->nrow * U[i]->ncol;
            mat_size[MV_FWD_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[MV_FWD_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
            mat_size[MV_BWD_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[MV_BWD_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
        }
        if (J[i] == NULL) H2P_int_vec_init(&J[i], 1);
        if (J_coord[i] == NULL)
        {
            H2P_dense_mat_init(&J_coord[i], 1, 1);
            J_coord[i]->nrow = 0;
            J_coord[i]->ncol = 0;
            J_coord[i]->ld   = 0;
        }
    }  // End of "for (int i = 0; i < h2pack->n_UJ; i++)"
    
    for (int i = 0; i < n_thread; i++)
        H2P_thread_buf_reset(thread_buf[i]);
    BLAS_SET_NUM_THREADS(n_thread);

    free(num_pp);
    free(node_adm_cnt);
    free(node_adm_pairs);
}

// These two functions are in H2Pack_build.c but not in H2Pack_utils.h
void H2P_build_B_AOT(H2Pack_p h2pack);
void H2P_build_D_AOT(H2Pack_p h2pack);

// Build H2 representation with a kernel function using randomized sampling
void H2P_build_random(
    H2Pack_p h2pack, const int BD_JIT, void *krnl_param, 
    kernel_eval_fptr krnl_eval, kernel_bimv_fptr krnl_bimv, const int krnl_bimv_flops
)
{
    double st, et;
    double *timers = h2pack->timers;
    
    if (krnl_eval == NULL)
    {
        ERROR_PRINTF("You need to provide a valid krnl_eval().\n");
        return;
    }

    h2pack->BD_JIT = BD_JIT;
    h2pack->krnl_param = krnl_param;
    h2pack->krnl_eval  = krnl_eval;
    h2pack->krnl_bimv  = krnl_bimv;
    h2pack->krnl_bimv_flops = krnl_bimv_flops;
    if (BD_JIT == 1 && krnl_bimv == NULL) 
        WARNING_PRINTF("krnl_eval() will be used in BD_JIT matvec. For better performance, consider using a krnl_bimv().\n");

    // 1. Build projection matrices and skeleton row sets
    st = get_wtime_sec();
    H2P_build_H2_UJ_random(h2pack);
    et = get_wtime_sec();
    timers[U_BUILD_TIMER_IDX] = et - st;

    // 2. Build generator matrices
    st = get_wtime_sec();
    H2P_generate_B_metadata(h2pack);
    if (BD_JIT == 0) H2P_build_B_AOT(h2pack);
    et = get_wtime_sec();
    timers[B_BUILD_TIMER_IDX] = et - st;
    
    // 3. Build dense blocks
    st = get_wtime_sec();
    H2P_generate_D_metadata(h2pack);
    if (BD_JIT == 0) H2P_build_D_AOT(h2pack);
    et = get_wtime_sec();
    timers[D_BUILD_TIMER_IDX] = et - st;

    // 4. Set up forward and backward permutation indices
    int n_point    = h2pack->n_point;
    int krnl_dim   = h2pack->krnl_dim;
    int *coord_idx = h2pack->coord_idx;
    int *fwd_pmt_idx = (int*) malloc(sizeof(int) * n_point * krnl_dim);
    int *bwd_pmt_idx = (int*) malloc(sizeof(int) * n_point * krnl_dim);
    for (int i = 0; i < n_point; i++)
    {
        for (int j = 0; j < krnl_dim; j++)
        {
            fwd_pmt_idx[i * krnl_dim + j] = coord_idx[i] * krnl_dim + j;
            bwd_pmt_idx[coord_idx[i] * krnl_dim + j] = i * krnl_dim + j;
        }
    }
    h2pack->fwd_pmt_idx = fwd_pmt_idx;
    h2pack->bwd_pmt_idx = bwd_pmt_idx;
}