#include <algorithm>
#include <iostream>
const char* lu_desc = "Serial LU decomposition.";

/*
 * This routine performs a LU decomposition
 *  A = L * U
 * where A, L, and U are lda-by-lda matrices stored in row-major format.
 * On exit, A is overwritten by U.
 */
void lu_decomposition(int n, double* A, double* L, int rank, int num_procs) {
    for(int k = 0; k < n; k++){
        L[k * n + k] = 1;
        for(int i = k + 1; i < n; i++){
            L[i * n + k] = A[i * n + k] / A[k * n + k];
        }
        for(int j = k + 1; j < n; j++){
            for(int i = k + 1; i < n; i++){
                A[i * n + j] = A[i * n + j] - L[i * n + k] * A[k * n + j];
            }
            A[j * n + k] = 0;
        }
    }
}

int find_column_max(int n, double* A, int k){
    int p = k; int max_value = std::abs(A[k * n + k]);
    for(int i = k + 1; i < n; i++){
        if(std::abs(A[i * n + k]) > max_value){
            max_value = std::abs(A[i * n + k]);
            p = i;
        }
    }
    return p;
}

/*
 * This routine performs a pivoted LU decomposition
 *  P * A = L * U
 * where A, L, and U are n-by-n matrices stored in row-major format, P is stored in one dimensional array
 * On exit, A is overwritten by U.
 */
void pivoted_lu_decomposition(int n, double* A, double* L, int* P, int rank, int num_procs) {
    for(int k = 0; k < n; k++){
        // pivoting
        P[k] = find_column_max(n, A, k);
        if(P[k] != k){
            std::swap_ranges(&A[k * n], &A[(k + 1) * n], &A[P[k] * n]);
            std::swap_ranges(&L[k * n], &L[k * n + k], &L[P[k] * n]);
        }
        // Gaussian elimination
        L[k * n + k] = 1;
        for(int i = k + 1; i < n; i++){
            L[i * n + k] = A[i * n + k] / A[k * n + k];
        }
        for(int j = k + 1; j < n; j++){
            for(int i = k + 1; i < n; i++){
                A[i * n + j] = A[i * n + j] - L[i * n + k] * A[k * n + j];
            }
            A[j * n + k] = 0;
        }
    }
}

