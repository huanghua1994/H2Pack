#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "H2Pack.h"
#include "block_jacobi_precond.h"

// Construct a block_jacobi_precond from a H2Pack structure
void H2P_build_block_jacobi_precond(H2Pack_p h2pack, const DTYPE shift, block_jacobi_precond_p *precond_)
{
    block_jacobi_precond_p precond = (block_jacobi_precond_p) malloc(sizeof(block_jacobi_precond_s));
    assert(precond != NULL);

    int   n_point     = h2pack->n_point;
    int   n_block     = h2pack->n_leaf_node;
    int   n_thread    = h2pack->n_thread;
    int   krnl_dim    = h2pack->krnl_dim;
    int   *pt_cluster = h2pack->pt_cluster;
    int   *leaf_nodes = h2pack->height_nodes;
    int   *D_nrow     = h2pack->D_nrow;
    DTYPE *coord      = h2pack->coord;

    double st = get_wtime_sec();
    
    int    *blk_sizes   = (int*)    malloc(sizeof(int)    * n_block);
    int    *blk_displs  = (int*)    malloc(sizeof(int)    * (n_block + 1));
    size_t *blk_inv_ptr = (size_t*) malloc(sizeof(size_t) * n_block);
    assert(blk_sizes != NULL && blk_displs != NULL && blk_inv_ptr != NULL);
    size_t blk_total_size = 0;
    blk_displs[0] = 0;
    for (int i = 0; i < n_block; i++)
    {
        int node = leaf_nodes[i];
        blk_sizes[i] = D_nrow[i];
        blk_inv_ptr[i] = blk_total_size;
        blk_displs[i + 1] = blk_displs[i] + D_nrow[i];
        blk_total_size += D_nrow[i] * D_nrow[i];
    }
    DTYPE *blk_inv = (DTYPE*) malloc(sizeof(DTYPE) * blk_total_size);
    ASSERT_PRINTF(blk_inv != NULL, "Failed to allocate array of size %zu for block Jacobi preconditioner\n", blk_total_size);

    size_t total_msize = sizeof(int) * (2 * n_block + 1) + sizeof(size_t) * n_block + sizeof(DTYPE) * blk_total_size;

    int *all_ipiv = (int*) malloc(sizeof(int) * n_point * krnl_dim);
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_block; i++)
        {
            int   node      = leaf_nodes[i];
            int   pt_s      = pt_cluster[2 * node];
            int   pt_e      = pt_cluster[2 * node + 1];
            int   npt       = pt_e - pt_s + 1;
            int   blk_size  = blk_sizes[i];
            int   *ipiv     = all_ipiv + pt_s * krnl_dim;
            DTYPE *blk_node = blk_inv + blk_inv_ptr[i];
            if (blk_size == 0) continue;
            h2pack->krnl_eval(
                coord + pt_s, n_point, npt,
                coord + pt_s, n_point, npt,
                h2pack->krnl_param, blk_node, npt * krnl_dim
            );
            for (int j = 0; j < blk_size; j++)
                blk_node[j * blk_size + j] += shift;
            int info;
            info = LAPACK_GETRF(LAPACK_ROW_MAJOR, blk_size, blk_size, blk_node, blk_size, ipiv);
            ASSERT_PRINTF(info == 0, "Node %d: blk_size = %d, LAPACK_GETRF return %d\n", node, blk_size, info);
            info = LAPACK_GETRI(LAPACK_ROW_MAJOR, blk_size,           blk_node, blk_size, ipiv);
            ASSERT_PRINTF(info == 0, "Node %d: blk_size = %d, LAPACK_GETRI return %d\n", node, blk_size, info);
        }  // End of i loop
    }  // End of "#pragma omp parallel"
    free(all_ipiv);

    double et = get_wtime_sec();

    size_t pmt_idx_bytes = sizeof(int)   * h2pack->krnl_mat_size;
    size_t pmt_vec_bytes = sizeof(DTYPE) * h2pack->krnl_mat_size;
    int   *fwd_pmt = (int*)   malloc(pmt_idx_bytes);
    int   *bwd_pmt = (int*)   malloc(pmt_idx_bytes);
    DTYPE *pmt_b   = (DTYPE*) malloc(pmt_vec_bytes);
    DTYPE *pmt_x   = (DTYPE*) malloc(pmt_vec_bytes);
    ASSERT_PRINTF(
        fwd_pmt != NULL && bwd_pmt != NULL && pmt_b != NULL && pmt_x != NULL, 
        "Failed to allocate vector permutation arrays for FSAI preconditioner\n"
    );
    memcpy(fwd_pmt, h2pack->fwd_pmt_idx, pmt_idx_bytes);
    memcpy(bwd_pmt, h2pack->bwd_pmt_idx, pmt_idx_bytes);
    total_msize += 2 * (pmt_idx_bytes + pmt_vec_bytes);

    precond->mat_size    = h2pack->krnl_mat_size;
    precond->n_block     = n_block;
    precond->blk_sizes   = blk_sizes;
    precond->blk_displs  = blk_displs;
    precond->blk_inv     = blk_inv;
    precond->pmt_b       = pmt_b;
    precond->pmt_x       = pmt_x;
    precond->fwd_pmt     = fwd_pmt;
    precond->bwd_pmt     = bwd_pmt;
    precond->blk_inv_ptr = blk_inv_ptr;
    precond->t_build  = et - st;
    precond->t_apply  = 0.0;
    precond->n_apply  = 0;
    precond->mem_MB   = (double) total_msize / 1048576.0;
    *precond_ = precond;
}

// Apply block Jacobi preconditioner, x := M_{BJP}^{-1} * b
void block_jacobi_precond_apply(block_jacobi_precond_p precond, const DTYPE *b, DTYPE *x)
{
    if (precond == NULL) return;
    
    int    mat_size     = precond->mat_size;
    int    n_block      = precond->n_block;
    int    *blk_sizes   = precond->blk_sizes;
    int    *blk_displs  = precond->blk_displs;
    int    *fwd_pmt     = precond->fwd_pmt;
    int    *bwd_pmt     = precond->bwd_pmt;
    size_t *blk_inv_ptr = precond->blk_inv_ptr;
    DTYPE  *blk_inv     = precond->blk_inv;
    DTYPE  *pmt_b       = precond->pmt_b;
    DTYPE  *pmt_x       = precond->pmt_x;

    double st = get_wtime_sec();
    gather_vector_elements(sizeof(DTYPE), mat_size, fwd_pmt, b, pmt_b);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_block; i++)
    {
        int blk_size_i = blk_sizes[i];
        int blk_spos_i = blk_displs[i];
        const DTYPE *blk_inv_i = blk_inv + blk_inv_ptr[i];
        const DTYPE *b_blk = pmt_b + blk_spos_i;
        DTYPE *x_blk = pmt_x + blk_spos_i;
        CBLAS_GEMV(
            CblasRowMajor, CblasNoTrans, blk_size_i, blk_size_i, 
            1.0, blk_inv_i, blk_size_i, b_blk, 1, 0.0, x_blk, 1
        );
    }
    gather_vector_elements(sizeof(DTYPE), mat_size, bwd_pmt, pmt_x, x);
    double et = get_wtime_sec();
    precond->t_apply += et - st;
    precond->n_apply++;
}

// Destroy a block_jacobi_precond structure
void block_jacobi_precond_destroy(block_jacobi_precond_p *precond_)
{
    block_jacobi_precond_p precond = *precond_;
    if (precond == NULL) return;
    free(precond->blk_sizes);
    free(precond->blk_displs);
    free(precond->blk_inv);
    free(precond->pmt_b);
    free(precond->pmt_x);
    free(precond->fwd_pmt);
    free(precond->bwd_pmt);
    free(precond->blk_inv_ptr);
    free(precond);
    *precond_ = NULL;
}

// Print statistic info of a FSAI_precond structure
void block_jacobi_precond_print_stat(block_jacobi_precond_p precond)
{
    if (precond == NULL) return;
    printf(
        "Block Jacobi precond used memory = %.2lf MB, build time = %.3lf sec, apply avg time = %.3lf sec\n", 
        precond->mem_MB, precond->t_build, precond->t_apply / (double) precond->n_apply
    );
}
