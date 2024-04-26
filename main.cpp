#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <time.h>
#include <mpi.h>

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

void lu_decomposition(int n, double* A, double* L, double* U, int rank, int num_procs);

void printMatrix(double* A, int n, int m) {
	printf("\n");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            printf("%lf ", A[i * n + j]);
        }
        printf("\n");
    }
	printf("\n");
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv){
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: size of matrices" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set matrix initialization seed" << std::endl;
        return 0;
    }

    int n = 4;
    double* A = (double *) malloc(sizeof(double) * n * n);
    double* L = (double *) malloc(sizeof(double) * n * n);
    double* U = (double *) malloc(sizeof(double) * n * n);
    // A[0] = 10; A[1] = 5; A[2] = 9; A[3] = 7;
	for (int i = 1; i <= n * n; i++) {
		A[i-1] = i % 10;
	}

	int num_procs, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    lu_decomposition(n, A, L, U, rank, num_procs);
    printMatrix(A, n, n);
    printMatrix(L, n, n);
    printMatrix(U, n, n);
}

