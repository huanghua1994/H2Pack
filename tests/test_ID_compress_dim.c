#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include <mkl.h>

#include "H2Pack_utils.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_ID_compress.h"

void RPY(
    int np0, const DTYPE *pos0, int np1, const DTYPE *pos1, 
    const DTYPE a, const DTYPE eta, DTYPE *A, const int ldA
)
{
    const DTYPE C   = 1.0 / (6.0 * M_PI * a * eta);
    const DTYPE aa  = a * a;
    const DTYPE a2  = 2.0 * a;
    const DTYPE aa2 = aa * 2.0;
    const DTYPE aa_2o3   = aa2 / 3.0;
    const DTYPE C_075    = C * 0.75;
    const DTYPE C_9o32oa = C * 9.0 / 32.0 / a;
    const DTYPE C_3o32oa = C * 3.0 / 32.0 / a;
    for (int i = 0; i < np0; i++)
    {
        DTYPE pos_i0 = pos0[i];
        DTYPE pos_i1 = pos0[i + np0];
        DTYPE pos_i2 = pos0[i + np0 * 2];
        for (int j = 0; j < np1; j++)
        {
            DTYPE r0 = pos_i0 - pos1[j];
            DTYPE r1 = pos_i1 - pos1[j + np1];
            DTYPE r2 = pos_i2 - pos1[j + np1 * 2];
            DTYPE s2 = r0 * r0 + r1 * r1 + r2 * r2;
            DTYPE s  = sqrt(s2);
            DTYPE inv_s = 1.0 / s;
            r0 *= inv_s;
            r1 *= inv_s;
            r2 *= inv_s;
            DTYPE t1, t2;
            if (s < a2)
            {
                t1 = C - C_9o32oa * s;
                t2 = C_3o32oa * s;
            } else {
                t1 = C_075 / s * (1 + aa_2o3 / s2);
                t2 = C_075 / s * (1 - aa2 / s2); 
            }
            int base = 3 * i * ldA + 3 * j;
            #define krnl(k, l) A[base + k * ldA + l]
            krnl(0, 0) = t2 * r0 * r0 + t1;
            krnl(0, 1) = t2 * r0 * r1;
            krnl(0, 2) = t2 * r0 * r2;
            krnl(1, 0) = t2 * r1 * r0;
            krnl(1, 1) = t2 * r1 * r1 + t1;
            krnl(1, 2) = t2 * r1 * r2;
            krnl(2, 0) = t2 * r2 * r0;
            krnl(2, 1) = t2 * r2 * r1;
            krnl(2, 2) = t2 * r2 * r2 + t1;
        }
    }
}

int main()
{
    int nrow, ncol, kdim = 3;
    printf("matrix size: ");
    scanf("%d%d", &nrow, &ncol);
    int A_nrow = nrow * kdim;
    int A_ncol = ncol * kdim;
    DTYPE tol_norm;
    printf("norm_rel_tol: ");
    scanf("%lf", &tol_norm);
    
    H2P_dense_mat_t A, A0, U;
    H2P_int_vec_t J;
    H2P_dense_mat_init(&A, A_nrow, A_ncol);
    H2P_dense_mat_init(&A0, A_nrow, A_ncol);
    H2P_int_vec_init(&J, A_nrow);
    
    DTYPE *coord0 = (DTYPE*) malloc(sizeof(DTYPE) * A_nrow);
    DTYPE *coord1 = (DTYPE*) malloc(sizeof(DTYPE) * A_ncol);
    assert(coord0 != NULL && coord1 != NULL);
    DTYPE *x0 = coord0, *x1 = coord1;
    DTYPE *y0 = coord0 + nrow, *y1 = coord1 + ncol;
    DTYPE *z0 = coord0 + nrow * 2, *z1 = coord1 + ncol * 2;
    for (int i = 0; i < nrow; i++) 
    {
        x0[i] = drand48();
        y0[i] = drand48();
        z0[i] = drand48();
    }
    for (int i = 0; i < ncol; i++) 
    {
        x1[i] = drand48() + 1.9;
        y1[i] = drand48() + 0.89;
        z1[i] = drand48() + 0.64;
    }
    
    RPY(nrow, coord0, ncol, coord1, 1.0, 1.0, A->data, A_ncol);
    memcpy(A0->data, A->data, sizeof(DTYPE) * A_nrow * A_ncol);
    DTYPE A0_fnorm = 0.0;
    for (int i = 0; i < A_nrow * A_ncol; i++)
        A0_fnorm += A->data[i] * A->data[i];
    
    int   *ID_buff = (int *)   malloc(sizeof(int)     * A->nrow * 4);
    DTYPE *QR_buff = (DTYPE *) malloc(sizeof(QR_buff) * A->nrow);
    H2P_ID_compress(
        A, QR_REL_NRM, &tol_norm, &U, J, 
        1, QR_buff, ID_buff, kdim
    );
    
    DTYPE *AJ = (DTYPE*) malloc(sizeof(DTYPE) * U->ncol * A_ncol);
    for (int i = 0; i < J->length; i++)
    {
        int i30 = i * 3 + 0;
        int i31 = i * 3 + 1;
        int i32 = i * 3 + 2;
        int j30 = J->data[i] * 3 + 0;
        int j31 = J->data[i] * 3 + 1;
        int j32 = J->data[i] * 3 + 2;
        memcpy(AJ + i30*A_ncol, A0->data + j30*A_ncol, sizeof(DTYPE) * A_ncol);
        memcpy(AJ + i31*A_ncol, A0->data + j31*A_ncol, sizeof(DTYPE) * A_ncol);
        memcpy(AJ + i32*A_ncol, A0->data + j32*A_ncol, sizeof(DTYPE) * A_ncol);
    }
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, A_nrow, A_ncol, U->ncol,
        1.0, U->data, U->ncol, AJ, A_ncol, -1.0, A0->data, A_ncol
    );
    DTYPE res_fnorm = 0.0;
    for (int i = 0; i < A_nrow * A_ncol; i++)
        res_fnorm += A0->data[i] * A0->data[i];
    res_fnorm = sqrt(res_fnorm);
    printf("U rank = %d (%d column blocks)\n", U->ncol, J->length);
    printf("||A - A_{H2}||_fro / ||A||_fro = %e\n", res_fnorm / A0_fnorm);
    
    free(QR_buff);
    free(ID_buff);
    free(coord0);
    free(coord1);
    H2P_int_vec_destroy(J);
    H2P_dense_mat_destroy(U);
    H2P_dense_mat_destroy(A);
    H2P_dense_mat_destroy(A0);
    return 0;
}