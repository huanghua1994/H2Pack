#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "H2Pack_config.h"
#include "H2Pack_utils.h"
#include "H2Pack_typedef.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_ID_compress.h"

// Copy some columns from a matrix to another matrix
// Input parameters:
//   src_mat : Source matrix with required columns
//   src_ld  : Leading dimension of the source matrix, should >= max(col_idx(:))+1
//   dst_ld  : Leading dimension of the destination matrix, should >= ncol
//   nrow    : Number of rows in the source and destination matrices
//   col_idx : Indices of required columns
//   ncol    : Number of required columns
// Output parameter:
//   dst_mat : Destination matrix with required columns only
void H2P_copy_matrix_columns(
    DTYPE *src_mat, const int src_ld, DTYPE *dst_mat, const int dst_ld, 
    const int nrow, int *col_idx, const int ncol
)
{
    for (int irow = 0; irow < nrow; irow++)
    {
        DTYPE *src_row = src_mat + src_ld * irow;
        DTYPE *dst_row = dst_mat + dst_ld * irow;
        for (int icol = 0; icol < ncol; icol++)
            dst_row[icol] = src_row[col_idx[icol]];
    }
}

// Copy a block from ta matrix to another matrix
// Input parameters:
//   nrow    : Number of rows to be copied 
//   ncol    : Number of columns to be copied
//   src_mat : Source matrix 
//   src_ld  : Leading dimension of the source matrix
//   dst_ld  : Leading dimension of the destination matrix
// Output parameter:
//   dst_mat : Destination matrix
void H2P_copy_matrix_block(
    const int nrow, const int ncol, DTYPE *src_mat, const int src_ld, 
    DTYPE *dst_mat, const int dst_ld
)
{
    size_t row_msize = sizeof(DTYPE) * ncol;
    for (int irow = 0; irow < nrow; irow ++)
    {
        DTYPE *src_row = src_mat + irow * src_ld;
        DTYPE *dst_row = dst_mat + irow * dst_ld;
        memcpy(dst_row, src_row, row_msize);
    }
}


// Evaluate a kernel matrix
// Input parameters:
//   kernel  : Kernel matrix evaluation function
//   dim     : Dimension of point coordinate
//   coord   : Point coordinates
//   n_point : Total number of points 
//   idx_x   : Indices of points in the X point set
//   idx_y   : Indices of points in the Y point set
//   tmp_x   : Auxiliary matrix for storing X point coordinates
//   tmp_y   : Auxiliary matrix for storing Y point coordinates 
// Output parameter:
//   kernel_mat : Obtained kernel matrix, size idx_x->length * idx_y->length
void H2P_eval_kernel_matrix_index(
    kernel_func_ptr kernel, const int dim, DTYPE *coord, const int n_point, 
    H2P_int_vec_t idx_x, H2P_int_vec_t idx_y, H2P_dense_mat_t tmp_x, 
    H2P_dense_mat_t tmp_y, DTYPE *kernel_mat
)
{
    const int nrow = idx_x->length;
    const int ncol = idx_y->length;   
    H2P_dense_mat_resize(tmp_x, dim, nrow);
    H2P_dense_mat_resize(tmp_y, dim, ncol);
    H2P_copy_matrix_columns(coord, n_point, tmp_x->data, nrow, dim, idx_x->data, nrow);
    H2P_copy_matrix_columns(coord, n_point, tmp_y->data, ncol, dim, idx_y->data, ncol);
    kernel(
        tmp_x->data, nrow, nrow, 
        tmp_y->data, ncol, ncol, 
        dim, kernel_mat, ncol
    );
}

// Evaluate a kernel matrix
// Input parameters:
//   kernel  : Kernel matrix evaluation function
//   dim     : Dimension of point coordinate
//   x_coord : X point set coordinates, size nx-by-dim
//   y_coord : Y point set coordinates, size ny-by-dim
// Output parameter:
//   kernel_mat : Obtained kernel matrix, nx-by-ny
void H2P_eval_kernel_matrix_direct(
    kernel_func_ptr kernel, const int dim, H2P_dense_mat_t x_coord, 
    H2P_dense_mat_t y_coord, H2P_dense_mat_t kernel_mat
)
{
    const int nrow = x_coord->ncol;
    const int ncol = y_coord->ncol;
    H2P_dense_mat_resize(kernel_mat, nrow, ncol);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
        int rem = nrow % nt;
        int bs0 = nrow / nt;
        int bs1 = bs0 + 1;
        int x_blk_srow, x_blk_nrow;
        if (tid < rem)
        {
            x_blk_srow = tid * bs1;
            x_blk_nrow = bs1;
        } else {
            x_blk_srow = tid * bs0 + rem;
            x_blk_nrow = bs0;
        }
        DTYPE *kernel_mat_srow = kernel_mat->data + x_blk_srow * kernel_mat->ld;
        DTYPE *x_coord_spos = x_coord->data + x_blk_srow;
        
        kernel(
            x_coord_spos, x_coord->ncol, x_blk_nrow, 
            y_coord->data, ncol, ncol, 
            dim, kernel_mat_srow, kernel_mat->ld
        );
    }
}

// Evaluate a kernel matrix
// Input parameters:
//   kernel     : Kernel matrix evaluation function
//   dim        : Dimension of point coordinate
//   coord      : X point coordinates (not all used)
//   n_point    : Total number of points 
//   idx_x      : Indices of points in the x point set, length nx
//   box_center : X point box center coordinate, for shifting x points
//   pp_coord   : Proxy point set coordinates, size ny-by-dim
//   tmp_x      : Auxiliary matrix for storing x point coordinates
// Output parameter:
//   kernel_mat : Obtained kernel matrix, nx-by-ny
void H2P_eval_kernel_matrix_UJ_proxy(
    kernel_func_ptr kernel, const int dim, DTYPE *coord, const int n_point, 
    H2P_int_vec_t idx_x, DTYPE *box_center, H2P_dense_mat_t pp_coord, 
    H2P_dense_mat_t tmp_x, H2P_dense_mat_t kernel_mat
)
{
    const int nrow = idx_x->length;
    const int ncol = pp_coord->ncol;
    H2P_dense_mat_resize(kernel_mat, nrow, ncol);
    H2P_dense_mat_resize(tmp_x, dim, nrow);
    H2P_copy_matrix_columns(coord, n_point, tmp_x->data, nrow, dim, idx_x->data, nrow);
    for (int j = 0; j < dim; j++)
    {
        DTYPE *tmp_x_j = tmp_x->data + j * nrow;
        DTYPE box_center_j = box_center[j];
        #pragma omp simd
        for (int i = 0; i < nrow; i++) 
            tmp_x_j[i] -= box_center_j;
    }
    kernel(
        tmp_x->data, nrow, nrow,
        pp_coord->data, ncol, ncol, 
        dim, kernel_mat->data, kernel_mat->ld
    );
}

// Check if a coordinate is in box [-L/2, L/2]^dim
// Input parameters:
//   dim   : Dimension of coordinate
//   coord : Coordinate
//   L     : Box size
// Output parameter:
//   <return> : If the coordinate is in the box
int point_in_box(const int dim, DTYPE *coord, DTYPE L)
{
    int res = 1;
    DTYPE semi_L = L * 0.5;
    for (int i = 0; i < dim; i++)
    {
        DTYPE coord_i = coord[i];
        if ((coord_i < -semi_L) || (coord_i > semi_L))
        {
            res = 0;
            break;
        }
    }
    return res;
}

// Generate proxy points for constructing H2 projection and skeleton matrices
void H2P_generate_proxy_point(
    const int dim, const int max_level, const int start_level,
    DTYPE max_L, kernel_func_ptr kernel, H2P_dense_mat_t **pp_
)
{   
    // 1. Initialize proxy point arrays
    H2P_dense_mat_t *pp = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * (max_level + 1));
    assert(pp != NULL);
    for (int i = 0; i <= max_level; i++) pp[i] = NULL;
    
    // 2. Initialize temporary arrays. The numbers of Nx and Ny points are empirical values
    int Nx_size = 1500;
    int Ny_size = 10000;
    H2P_dense_mat_t tmpA, Nx_points, Ny_points, min_dist;
    H2P_int_vec_t skel_idx;
    H2P_dense_mat_init(&tmpA, Nx_size, Ny_size);
    H2P_dense_mat_init(&Nx_points, dim, Nx_size);
    H2P_dense_mat_init(&Ny_points, dim, Ny_size);
    H2P_dense_mat_init(&min_dist, Nx_size, 1);
    H2P_int_vec_init(&skel_idx, Nx_size);
    srand48(time(NULL));
    int nthreads = omp_get_max_threads();

    // 3. Construct proxy points on each level
    DTYPE pow_2_level = 0.5;
    H2P_dense_mat_t QR_buff;
    H2P_int_vec_t   ID_buff;
    H2P_dense_mat_init(&QR_buff, 2 * Ny_size, 1);
    H2P_int_vec_init(&ID_buff, 4 * Ny_size);
    for (int level = 0; level < start_level; level++) pow_2_level *= 2.0;
    for (int level = start_level; level <= max_level; level++)
    {
        // (1) Decide box sizes: Nx points are in box1, Ny points are in box3
        //     but not in box2 (points in box2 are inadmissible to Nx points)
        pow_2_level *= 2.0;
        DTYPE L1 = max_L / pow_2_level;
        DTYPE L2 = (1.0 + 2.0 * ALPHA_H2) * L1;
        DTYPE semi_L1   = L1 * 0.5;
        DTYPE semi_L3_0 = max_L - L1;
        DTYPE semi_L3_1 = (1.0 + 8.0 * ALPHA_H2) * L1;
        DTYPE semi_L3   = MIN(semi_L3_0, semi_L3_1);
        DTYPE L3 = 2.0 * semi_L3;
        
        // (2) Generate Nx and Ny points
        H2P_dense_mat_resize(Nx_points, dim, Nx_size);
        for (int i = 0; i < Nx_size * dim; i++)
        {
            DTYPE val = drand48();
            Nx_points->data[i] = L1 * val - semi_L1;
        }
        H2P_dense_mat_resize(Ny_points, dim, Ny_size);
        for (int i = 0; i < Ny_size; i++)
        {
            DTYPE *tmp_coord = tmpA->data;
            int flag = 1;
            while (flag == 1)
            {
                for (int j = 0; j < dim; j++)
                {
                    DTYPE val = drand48();
                    tmp_coord[j] = L3 * val - semi_L3;
                }
                flag = point_in_box(dim, tmp_coord, L2);
            }
            DTYPE *Ny_i = Ny_points->data + i;
            for (int j = 0; j < dim; j++)
                Ny_i[j * Ny_size] = tmp_coord[j];
        }
        

        // (3) Use ID to select skeleton points in Nx first, then use the
        //     skeleton Nx points to select skeleton Ny points
        DTYPE rel_tol = 1e-14;
        
        H2P_eval_kernel_matrix_direct(kernel, dim, Nx_points, Ny_points, tmpA);
        H2P_dense_mat_resize(QR_buff, 2 * tmpA->nrow, 1);
        H2P_int_vec_set_capacity(ID_buff, 4 * tmpA->nrow);
        H2P_ID_compress(
            tmpA, QR_REL_NRM, &rel_tol, NULL, skel_idx, 
            nthreads, QR_buff->data, ID_buff->data
        );
        H2P_dense_mat_select_columns(Nx_points, skel_idx);
        
        H2P_eval_kernel_matrix_direct(kernel, dim, Ny_points, Nx_points, tmpA);
        H2P_dense_mat_resize(QR_buff, 2 * tmpA->nrow, 1);
        H2P_int_vec_set_capacity(ID_buff, 4 * tmpA->nrow);
        H2P_ID_compress(
            tmpA, QR_REL_NRM, &rel_tol, NULL, skel_idx, 
            nthreads, QR_buff->data, ID_buff->data
        );
        H2P_dense_mat_select_columns(Ny_points, skel_idx);
        
        // (4) Make the skeleton Ny points dense and then use them as proxy points
        int ny = skel_idx->length;
        H2P_dense_mat_resize(min_dist, ny, 1);
        DTYPE *coord_i = tmpA->data;
        DTYPE *coord_j = tmpA->data + dim;
        for (int i = 0; i < ny; i++) min_dist->data[i] = 1e20;
        for (int i = 0; i < ny; i++)
        {
            for (int k = 0; k < dim; k++)
                coord_i[k] = Ny_points->data[i + k * Ny_points->ncol];
            
            for (int j = 0; j < i; j++)
            {
                DTYPE dist_ij = 0.0;
                for (int k = 0; k < dim; k++)
                {
                    DTYPE diff = coord_i[k] - Ny_points->data[j + k * Ny_points->ncol];
                    dist_ij += diff * diff;
                }
                dist_ij = DSQRT(dist_ij);
                min_dist->data[i] = MIN(min_dist->data[i], dist_ij);
                min_dist->data[j] = MIN(min_dist->data[j], dist_ij);
            }
        }
        const int Ny_size2 = 2 * ny;
        H2P_dense_mat_init(&pp[level], dim, Ny_size2);
        H2P_dense_mat_t pp_level = pp[level];
        // Also transpose the coordinate array for vectorizing kernel evaluation here
        for (int i = 0; i < ny; i++)
        {
            DTYPE *tmp_coord0 = tmpA->data;
            DTYPE *tmp_coord1 = tmpA->data + dim;
            DTYPE *Ny_point_i = Ny_points->data + i;
            for (int j = 0; j < dim; j++)
                tmp_coord0[j] = Ny_point_i[j * Ny_points->ncol];
            DTYPE radius_i_scale = min_dist->data[i] * 0.33;
            int flag = 1;
            while (flag == 1)
            {
                DTYPE radius_1 = 0.0;
                for (int j = 0; j < dim; j++)
                {
                    tmp_coord1[j] = drand48() - 0.5;
                    radius_1 += tmp_coord1[j] * tmp_coord1[j];
                }
                DTYPE inv_radius_1 = 1.0 / DSQRT(radius_1);
                for (int j = 0; j < dim; j++) 
                {
                    tmp_coord1[j] *= inv_radius_1;
                    tmp_coord1[j] *= radius_i_scale;
                    tmp_coord1[j] += tmp_coord0[j];
                }
                if ((point_in_box(dim, tmp_coord1, L2) == 0) &&
                    (point_in_box(dim, tmp_coord1, L3) == 1))
                    flag = 0;
            }
            DTYPE *coord_0 = pp_level->data + (2 * i);
            DTYPE *coord_1 = pp_level->data + (2 * i + 1);
            for (int j = 0; j < dim; j++)
            {
                coord_0[j * Ny_size2] = tmp_coord0[j];
                coord_1[j * Ny_size2] = tmp_coord1[j];
            }
        }
    }
    
    *pp_ = pp;
    H2P_int_vec_destroy(skel_idx);
    H2P_int_vec_destroy(ID_buff);
    H2P_dense_mat_destroy(QR_buff);
    H2P_dense_mat_destroy(tmpA);
    H2P_dense_mat_destroy(Nx_points);
    H2P_dense_mat_destroy(Ny_points);
    H2P_dense_mat_destroy(min_dist);
}

// Build H2 projection matrices using the direct approach
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameter:
//   h2pack : H2Pack structure with H2 projection matrices
void H2P_build_UJ_proxy(H2Pack_t h2pack)
{
    int   dim            = h2pack->dim;
    int   n_node         = h2pack->n_node;
    int   n_point        = h2pack->n_point;
    int   n_leaf_node    = h2pack->n_leaf_node;
    int   max_child      = h2pack->max_child;
    int   max_level      = h2pack->max_level;
    int   max_adm_height = h2pack->max_adm_height;
    int   min_adm_level  = h2pack->min_adm_level;
    int   stop_type      = h2pack->QR_stop_type;
    int   *children      = h2pack->children;
    int   *n_child       = h2pack->n_child;
    int   *height_n_node = h2pack->height_n_node;
    int   *height_nodes  = h2pack->height_nodes;
    int   *leaf_nodes    = h2pack->height_nodes;
    int   *node_level    = h2pack->node_level;
    int   *cluster       = h2pack->cluster;
    DTYPE *coord         = h2pack->coord;
    DTYPE *enbox         = h2pack->enbox;
    H2P_dense_mat_t *pp  = h2pack->pp;
    kernel_func_ptr kernel = h2pack->kernel;
    void  *stop_param;
    if (stop_type == QR_RANK) 
        stop_param = &h2pack->QR_stop_rank;
    if ((stop_type == QR_REL_NRM) || (stop_type == QR_ABS_NRM))
        stop_param = &h2pack->QR_stop_tol;
    
    // 1. Allocate U and J
    h2pack->n_UJ = n_node;
    h2pack->U       = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    h2pack->J       = (H2P_int_vec_t*)   malloc(sizeof(H2P_int_vec_t)   * n_node);
    h2pack->J_coord = (H2P_dense_mat_t*) malloc(sizeof(H2P_dense_mat_t) * n_node);
    assert(h2pack->U != NULL && h2pack->J != NULL && h2pack->J_coord != NULL);
    for (int i = 0; i < h2pack->n_UJ; i++)
    {
        h2pack->U[i]       = NULL;
        h2pack->J[i]       = NULL;
        h2pack->J_coord[i] = NULL;
    }
    H2P_dense_mat_t *U       = h2pack->U;
    H2P_int_vec_t   *J       = h2pack->J;
    H2P_dense_mat_t *J_coord = h2pack->J_coord;
    
    // 2. Hierarchical construction height by height. max_adm_height is the 
    //    highest position that still has admissible blocks, so we only need 
    //    to compress matrix blocks to that height since higher blocks 
    //    are inadmissible and cannot be compressed.
    for (int i = 0; i <= max_adm_height; i++)
    {
        int *height_i_nodes = height_nodes + i * n_leaf_node;
        int height_i_n_node = height_n_node[i];
        int nthreads = MIN(height_i_n_node, h2pack->n_thread);
        
        // (1) Update row indices associated with clusters at height i
        if (i == 0)
        {
            // Leaf nodes, use all points
            #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
            for (int j = 0; j < height_n_node[i]; j++)
            {
                int node = height_i_nodes[j];
                int s_index = cluster[node * 2];
                int e_index = cluster[node * 2 + 1];
                int node_npts = e_index - s_index + 1;
                H2P_int_vec_init(&J[node], node_npts);
                for (int k = 0; k < node_npts; k++)
                    J[node]->data[k] = s_index + k;
                J[node]->length = node_npts;

                H2P_dense_mat_init(&J_coord[node], dim, node_npts);
                H2P_copy_matrix_block(dim, node_npts, coord + s_index, n_point, J_coord[node]->data, node_npts);
            }
        } else {
            // Non-leaf nodes, gather row indices from children nodes
            #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
            for (int j = 0; j < height_n_node[i]; j++)
            {
                int node = height_i_nodes[j];
                int level = node_level[node];
                if (level < min_adm_level) continue;
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
            }
        }
        
        // (2) Compression at height i
        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            H2P_dense_mat_t A_block = h2pack->tb[tid]->mat0;
            H2P_dense_mat_t tmp_x   = h2pack->tb[tid]->mat1;
            H2P_dense_mat_t QR_buff = h2pack->tb[tid]->mat1;
            H2P_int_vec_t   sub_idx = h2pack->tb[tid]->idx0;
            H2P_int_vec_t   ID_buff = h2pack->tb[tid]->idx1;
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < height_n_node[i]; j++)
            {
                int node  = height_i_nodes[j];
                int level = node_level[node];
                if (level < min_adm_level) continue;
                
                DTYPE *node_box = enbox + node * 2 * dim;
                int nrow = J[node]->length;
                int ncol = pp[level]->ncol;
                H2P_dense_mat_resize(tmp_x, dim, J[node]->length);
                H2P_dense_mat_resize(A_block, nrow, ncol);

                // Gather coordinates of this node's points (== all children nodes' skeleton points)
                if (i == 0)
                {
                    tmp_x = J_coord[node];
                } else {
                    int n_child_node = n_child[node];
                    int *child_nodes = children + node * max_child;
                    int J_child_size = 0;
                    for (int i_child = 0; i_child < n_child_node; i_child++)
                    {
                        int   i_child_node = child_nodes[i_child];
                        DTYPE *src_mat     = J_coord[i_child_node]->data;
                        DTYPE *dst_mat     = tmp_x->data + J_child_size;
                        int   src_ld       = J_coord[i_child_node]->ncol;
                        int   dst_ld       = tmp_x->ncol;
                        H2P_copy_matrix_block(dim, src_ld, src_mat, src_ld, dst_mat, dst_ld);
                        J_child_size += J[i_child_node]->length;
                    }
                }
                
                // Shift points in this node so their center is at the original point
                for (int k = 0; k < dim; k++)
                {
                    DTYPE box_center_k = node_box[k] + 0.5 * node_box[dim + k];
                    DTYPE *tmp_x_row_k = tmp_x->data + k * nrow;
                    #pragma omp simd
                    for (int l = 0; l < nrow; l++)
                        tmp_x_row_k[l] -= box_center_k;
                }
                kernel(
                    tmp_x->data, nrow, nrow,
                    pp[level]->data, ncol, ncol, 
                    dim, A_block->data, A_block->ld
                );

                // ID compress
                H2P_dense_mat_resize(QR_buff, 2 * A_block->nrow, 1);
                H2P_int_vec_set_capacity(ID_buff, 4 * A_block->nrow);
                H2P_ID_compress(
                    A_block, stop_type, stop_param, &U[node], sub_idx, 
                    1, QR_buff->data, ID_buff->data
                );
                
                // Choose the skeleton points of this node
                for (int k = 0; k < U[node]->ncol; k++)
                    J[node]->data[k] = J[node]->data[sub_idx->data[k]];
                J[node]->length = U[node]->ncol;
                H2P_dense_mat_init(&J_coord[node], dim, U[node]->ncol);
                H2P_copy_matrix_columns(
                    coord, n_point, J_coord[node]->data, J[node]->length, 
                    dim, J[node]->data, J[node]->length
                );
            }
        }
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
            h2pack->mat_size[0] += U[i]->nrow * U[i]->ncol;
            h2pack->mat_size[3] += U[i]->nrow * U[i]->ncol;
            h2pack->mat_size[3] += U[i]->nrow + U[i]->ncol;
            h2pack->mat_size[5] += U[i]->nrow * U[i]->ncol;
            h2pack->mat_size[5] += U[i]->nrow + U[i]->ncol;
        }
        if (J[i] == NULL) H2P_int_vec_init(&J[i], 1);
        if (J_coord[i] == NULL)
        {
        	H2P_dense_mat_init(&J_coord[i], 1, 1);
            J_coord[i]->nrow = 0;
            J_coord[i]->ncol = 0;
            J_coord[i]->ld   = 0;
        }
    }
}

// Partition work units into multiple blocks s.t. each block has 
// approximately the same amount of work
// Input parameters:
//   n_work     : Number of work units
//   work_sizes : Work size of each work unit
//   total_size : Sum of work_sizes
//   n_block    : Number of blocks to be partitioned, the final result
//                may have fewer blocks
// Output parameter:
//   blk_displs : Indices of each block's first work unit
void H2P_partition_workload(
    const int n_work,  const int *work_sizes, const int total_size, 
    const int n_block, H2P_int_vec_t blk_displs
)
{
    int blk_size = total_size / n_block + 1;
    for (int i = 0; i < blk_displs->capacity; i++) blk_displs->data[i] = n_work;
    blk_displs->data[0] = 0;
    int curr_blk_size = 0, idx = 1;
    for (int i = 0; i < n_work; i++)
    {
        curr_blk_size += work_sizes[i];
        if (curr_blk_size >= blk_size)
        {
            blk_displs->data[idx] = i + 1;
            curr_blk_size = 0;
            idx++;
        }
    }
    if (curr_blk_size > 0)
    {
        blk_displs->data[idx] = n_work;
        idx++;
    }
    blk_displs->length = idx;
}

// Build H2 generator matrices
// Input parameter:
//   h2pack : H2Pack structure with H2 projection matrices
// Output parameter:
//   h2pack : H2Pack structure with H2 generator matrices
void H2P_build_B(H2Pack_t h2pack)
{
    int   dim          = h2pack->dim;
    int   n_node       = h2pack->n_node;
    int   n_point      = h2pack->n_point;
    int   n_thread     = h2pack->n_thread;
    int   n_r_adm_pair = h2pack->n_r_adm_pair;
    int   *r_adm_pairs = h2pack->r_adm_pairs;
    int   *node_level  = h2pack->node_level;
    int   *cluster     = h2pack->cluster;
    DTYPE *coord       = h2pack->coord;
    kernel_func_ptr kernel   = h2pack->kernel;
    H2P_int_vec_t   B_blk    = h2pack->B_blk;
    H2P_dense_mat_t *J_coord = h2pack->J_coord;

    double st, et;
    st = H2P_get_wtime_sec();

    // 1. Allocate B
    h2pack->n_B = n_r_adm_pair;
    h2pack->B_nrow = (int*) malloc(sizeof(int) * n_r_adm_pair);
    h2pack->B_ncol = (int*) malloc(sizeof(int) * n_r_adm_pair);
    h2pack->B_ptr  = (int*) malloc(sizeof(int) * (n_r_adm_pair + 1));
    assert(h2pack->B_nrow != NULL && h2pack->B_ncol != NULL && h2pack->B_ptr != NULL);
    int *B_nrow  = h2pack->B_nrow;
    int *B_ncol  = h2pack->B_ncol;
    int *B_ptr   = h2pack->B_ptr;
    
    // 2. Partition B matrices into multiple blocks s.t. each block has approximately
    //    the same workload (total size of B matrices in a block)
    B_ptr[0] = 0;
    int B_total_size = 0;
    int n_DTYPE_64B  = 64 / sizeof(DTYPE);
    H2P_int_vec_t *J = h2pack->J;
    h2pack->node_n_r_adm = (int*) malloc(sizeof(int) * n_node);
    assert(h2pack->node_n_r_adm != NULL);
    int *node_n_r_adm = h2pack->node_n_r_adm;
    memset(node_n_r_adm, 0, sizeof(int) * n_node);
    for (int i = 0; i < n_r_adm_pair; i++)
    {
        int node0  = r_adm_pairs[2 * i];
        int node1  = r_adm_pairs[2 * i + 1];
        int level0 = node_level[node0];
        int level1 = node_level[node1];
        node_n_r_adm[node0]++;
        node_n_r_adm[node1]++;
        int node0_npts, node1_npts;
        if (level0 == level1)
        {
            node0_npts = J[node0]->length;
            node1_npts = J[node1]->length;
        }
        if (level0 > level1)
        {
            int s_index1 = cluster[2 * node1];
            int e_index1 = cluster[2 * node1 + 1];
            node0_npts = J[node0]->length;
            node1_npts = e_index1 - s_index1 + 1;
        }
        if (level0 < level1)
        {
            int s_index0 = cluster[2 * node0];
            int e_index0 = cluster[2 * node0 + 1];
            node0_npts = e_index0 - s_index0 + 1;
            node1_npts = J[node1]->length;
        }
        int Bi_size = node0_npts * node1_npts;
        Bi_size = (Bi_size + n_DTYPE_64B - 1) / n_DTYPE_64B * n_DTYPE_64B;
        B_total_size += Bi_size;
        B_ptr[i + 1] = Bi_size;
    }
    H2P_partition_workload(n_r_adm_pair, B_ptr + 1, B_total_size, n_thread * 5, B_blk);
    for (int i = 1; i <= n_r_adm_pair; i++) B_ptr[i] += B_ptr[i - 1];


    // 3. Generate B matrices
    h2pack->B_data = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * B_total_size);
    assert(h2pack->B_data != NULL);
    DTYPE *B_data = h2pack->B_data;
    const int n_B_blk = B_blk->length;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic)
        for (int i_blk = 0; i_blk < n_B_blk; i_blk++)
        {
            int s_index = B_blk->data[i_blk];
            int e_index = B_blk->data[i_blk + 1];
            for (int i = s_index; i < e_index; i++)
            {
                int node0  = r_adm_pairs[2 * i];
                int node1  = r_adm_pairs[2 * i + 1];
                int level0 = node_level[node0];
                int level1 = node_level[node1];
                DTYPE *Bi  = B_data + B_ptr[i];

                // (1) Two nodes are of the same level, compress on both sides
                if (level0 == level1)
                {
                    int node0_npts = J[node0]->length;
                    int node1_npts = J[node1]->length;
                    B_nrow[i] = node0_npts;
                    B_ncol[i] = node1_npts;
                    kernel(
                    	J_coord[node0]->data, J_coord[node0]->ncol, J_coord[node0]->ncol,
                    	J_coord[node1]->data, J_coord[node1]->ncol, J_coord[node1]->ncol,
                    	dim, Bi, J_coord[node1]->ncol
                    );
                }
                
                // (2) node1 is a leaf node and its level is higher than node0's level, 
                //     only compress on node0's side
                if (level0 > level1)
                {
                    int s_index1 = cluster[2 * node1];
                    int e_index1 = cluster[2 * node1 + 1];
                    int node0_npts = J[node0]->length;
                    int node1_npts = e_index1 - s_index1 + 1;
                    B_nrow[i] = node0_npts;
                    B_ncol[i] = node1_npts;
                    kernel(
                    	J_coord[node0]->data, J_coord[node0]->ncol, J_coord[node0]->ncol,
                    	coord + s_index1, n_point, node1_npts, 
                    	dim, Bi, node1_npts
                    );
                }
                
                // (3) node0 is a leaf node and its level is higher than node1's level, 
                //     only compress on node1's side
                if (level0 < level1)
                {
                    int s_index0 = cluster[2 * node0];
                    int e_index0 = cluster[2 * node0 + 1];
                    int node0_npts = e_index0 - s_index0 + 1;
                    int node1_npts = J[node1]->length;
                    B_nrow[i] = node0_npts;
                    B_ncol[i] = node1_npts;
                    kernel(
                    	coord + s_index0, n_point, node0_npts, 
                    	J_coord[node1]->data, J_coord[node1]->ncol, J_coord[node1]->ncol,
                    	dim, Bi, J_coord[node1]->ncol
                    );
                }
            }
        }
    }
    
    // 4. Add statistic info
    h2pack->mat_size[1] = B_total_size;
    for (int i = 0; i < n_r_adm_pair; i++)
    {
        h2pack->mat_size[4] += 2 * (B_nrow[i] * B_ncol[i]);
        h2pack->mat_size[4] += 2 * (B_nrow[i] + B_ncol[i]);
    }
}

// Build dense blocks in the original matrices
// Input parameter:
//   h2pack : H2Pack structure with point partitioning info
// Output parameter:
//   h2pack : H2Pack structure with dense blocks
void H2P_build_D(H2Pack_t h2pack)
{
    int   dim            = h2pack->dim;
    int   n_thread       = h2pack->n_thread;
    int   n_point        = h2pack->n_point;
    int   n_leaf_node    = h2pack->n_leaf_node;
    int   n_r_inadm_pair = h2pack->n_r_inadm_pair;
    int   *leaf_nodes    = h2pack->height_nodes;
    int   *cluster       = h2pack->cluster;
    int   *r_inadm_pairs = h2pack->r_inadm_pairs;
    DTYPE *coord         = h2pack->coord;
    kernel_func_ptr kernel = h2pack->kernel;
    H2P_int_vec_t   D_blk0 = h2pack->D_blk0;
    H2P_int_vec_t   D_blk1 = h2pack->D_blk1;
    
    // 1. Allocate D
    h2pack->n_D = n_leaf_node + n_r_inadm_pair;
    h2pack->D_nrow = (int*) malloc(sizeof(int) * h2pack->n_D);
    h2pack->D_ncol = (int*) malloc(sizeof(int) * h2pack->n_D);
    h2pack->D_ptr  = (int*) malloc(sizeof(int) * (h2pack->n_D + 1));
    assert(h2pack->D_nrow != NULL && h2pack->D_ncol != NULL && h2pack->D_ptr != NULL);
    int *D_nrow  = h2pack->D_nrow;
    int *D_ncol  = h2pack->D_ncol;
    int *D_ptr   = h2pack->D_ptr;
    
    // 2. Partition D matrices into multiple blocks s.t. each block has approximately
    //    the same workload (total size of D matrices in a block)
    D_ptr[0] = 0;
    int D0_total_size = 0;
    int n_DTYPE_64B   = 64 / sizeof(DTYPE);
    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        int s_index = cluster[2 * node];
        int e_index = cluster[2 * node + 1];
        int node_npts = e_index - s_index + 1;
        int Di_size = node_npts * node_npts;
        Di_size = (Di_size + n_DTYPE_64B - 1) / n_DTYPE_64B * n_DTYPE_64B;
        D_ptr[i + 1] = Di_size;
        D0_total_size += Di_size;
    }
    H2P_partition_workload(n_leaf_node, D_ptr + 1, D0_total_size, n_thread * 5, D_blk0);
    int D1_total_size = 0;
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        int node0 = r_inadm_pairs[2 * i];
        int node1 = r_inadm_pairs[2 * i + 1];
        int s_index0 = cluster[2 * node0];
        int s_index1 = cluster[2 * node1];
        int e_index0 = cluster[2 * node0 + 1];
        int e_index1 = cluster[2 * node1 + 1];
        int node0_npts = e_index0 - s_index0 + 1;
        int node1_npts = e_index1 - s_index1 + 1;
        int Di_size = node0_npts * node1_npts;
        Di_size = (Di_size + n_DTYPE_64B - 1) / n_DTYPE_64B * n_DTYPE_64B;
        D_ptr[n_leaf_node + 1 + i] = Di_size;
        D1_total_size += Di_size;
    }
    H2P_partition_workload(n_r_inadm_pair, D_ptr + n_leaf_node + 1, D1_total_size, n_thread * 5, D_blk1);
    for (int i = 1; i <= n_leaf_node + n_r_inadm_pair; i++) D_ptr[i] += D_ptr[i - 1];
    
    h2pack->D_data = (DTYPE*) H2P_malloc_aligned(sizeof(DTYPE) * (D0_total_size + D1_total_size));
    assert(h2pack->D_data != NULL);
    DTYPE *D_data = h2pack->D_data;
    const int n_D0_blk = D_blk0->length;
    const int n_D1_blk = D_blk1->length;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
  
        // 3. Generate diagonal blocks (leaf node self interaction)
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk0 = 0; i_blk0 < n_D0_blk; i_blk0++)
        {
            int s_index = D_blk0->data[i_blk0];
            int e_index = D_blk0->data[i_blk0 + 1];
            for (int i = s_index; i < e_index; i++)
            {
                int node = leaf_nodes[i];
                int s_index = cluster[2 * node];
                int e_index = cluster[2 * node + 1];
                int node_npts = e_index - s_index + 1;
                DTYPE *Di = D_data + D_ptr[i];
                D_nrow[i] = node_npts;
                D_ncol[i] = node_npts;
                kernel(
                	coord + s_index, n_point, node_npts,
                	coord + s_index, n_point, node_npts,
                	dim, Di, node_npts
                );
            }
        }
        
        // 4. Generate off-diagonal blocks from inadmissible pairs
        #pragma omp for schedule(dynamic) 
        for (int i_blk1 = 0; i_blk1 < n_D1_blk; i_blk1++)
        {
            int s_index = D_blk1->data[i_blk1];
            int e_index = D_blk1->data[i_blk1 + 1];
            for (int i = s_index; i < e_index; i++)
            {
                int node0 = r_inadm_pairs[2 * i];
                int node1 = r_inadm_pairs[2 * i + 1];
                int s_index0 = cluster[2 * node0];
                int s_index1 = cluster[2 * node1];
                int e_index0 = cluster[2 * node0 + 1];
                int e_index1 = cluster[2 * node1 + 1];
                int node0_npts = e_index0 - s_index0 + 1;
                int node1_npts = e_index1 - s_index1 + 1;
                DTYPE *Di = D_data + D_ptr[i + n_leaf_node];
                D_nrow[i + n_leaf_node] = node0_npts;
                D_ncol[i + n_leaf_node] = node1_npts;
                kernel(
                	coord + s_index0, n_point, node0_npts,
                	coord + s_index1, n_point, node1_npts,
                	dim, Di, node1_npts
                );
            }
        }
    }
    
    // 5. Add statistic info
    h2pack->mat_size[2] = D0_total_size + D1_total_size;
    for (int i = 0; i < n_leaf_node; i++)
    {
        h2pack->mat_size[6] += D_nrow[i] * D_ncol[i];
        h2pack->mat_size[6] += D_nrow[i] + D_ncol[i];
    }
    for (int i = n_leaf_node; i < n_leaf_node + n_r_inadm_pair; i++)
    {
        h2pack->mat_size[6] += 2 * (D_nrow[i] * D_ncol[i]);
        h2pack->mat_size[6] += 2 * (D_nrow[i] + D_ncol[i]);
    }
}

// Build H2 representation with a kernel function
void H2P_build(H2Pack_t h2pack, kernel_func_ptr kernel, H2P_dense_mat_t *pp)
{
    double st, et;

    h2pack->kernel = kernel;
    h2pack->pp     = pp;

    // 1. Build projection matrices and skeleton row sets
    st = H2P_get_wtime_sec();
    H2P_build_UJ_proxy(h2pack);
    et = H2P_get_wtime_sec();
    h2pack->timers[1] = et - st;

    // 2. Build generator matrices
    st = H2P_get_wtime_sec();
    H2P_build_B(h2pack);
    et = H2P_get_wtime_sec();
    h2pack->timers[2] = et - st;
    
    // 3. Build dense blocks
    st = H2P_get_wtime_sec();
    H2P_build_D(h2pack);
    et = H2P_get_wtime_sec();
    h2pack->timers[3] = et - st;
}

