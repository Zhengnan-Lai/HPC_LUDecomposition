#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <time.h>
#include <mpi.h>
#include <sstream>

void save(std::ofstream& fsave, double* A, int n, int m, char name){
    fsave << "[" << name << "]\n";
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            fsave << A[i * n + j] << " ";
        }
        fsave << "\n";
    }
    fsave << "\n";
}

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

void lu_decomposition(int n, double* A, double* L, int rank, int num_procs);

void print_matrix(double* A, int n, int m, char name) {
	printf("[%c]\n", name);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            printf("%lf ", A[i * n + j]);
        }
        printf("\n");
    }
	printf("\n");
}

void generate_matrix(double* A, int n, int seed){
    std::random_device rd;
    std::mt19937 gen(seed ? seed : rd());
    for (int i = 0; i < n * n; i++) {
		std::uniform_real_distribution<float> rand_real(-10.0, 10.0);
        A[i] = rand_real(gen);
	}
}

void check_correctness(double* A, double* L, double* U, int n){
    // std::cout << "in check correctness\n";
    for(int i = 0; i < n; i++) for(int j = 0 ; j < n; j++){
        for(int k = 0; k < n; k++){
            A[i * n + j] -= L[i * n + k] * U[k * n + j];
        }
        if (std::abs(A[i * n + j]) > 1e-3) {
            std::cout << "correctness issue\n";
            // exit(1);
        }
        // assert(std::abs(A[i * n + j]) < 1e-3);
    }
    // print_matrix(L, n, n, 'L');
    // print_matrix(U, n, n, 'U');
    // print_matrix(A, n, n, 'A');
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
        std::cout << "-c <int>: check correctness of the result" << std::endl;
        return 0;
    }
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);
    int n = find_int_arg(argc, argv, "-n", 50);
    int check_correct = find_int_arg(argc, argv, "-c", 0);
    int seed = find_int_arg(argc, argv, "-s", 0);

    int num_procs, rank;
    double* A = (double *) malloc(sizeof(double) * n * n);
    double* L = (double *) malloc(sizeof(double) * n * n);
    double* U = (double *) malloc(sizeof(double) * n * n);
    double* A_copy;
    if(rank == 0){
        generate_matrix(A, n, seed);
    }

    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Bcast(A, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank == 0 && check_correct){
        A_copy = (double *) malloc(sizeof(double) * n * n);
        memcpy(A_copy, A, sizeof(double) * n * n);
    }
    if(rank == 0){
        save(fsave, A, n, n, 'A');
        // print_matrix(A, n, n, 'A');
    }
    
    double total_compute_time = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    total_compute_time -= MPI_Wtime();
    lu_decomposition(n, A, L, rank, num_procs);
    // Synchronize again before obtaining final time
    MPI_Barrier(MPI_COMM_WORLD);
    total_compute_time += MPI_Wtime();

    // auto start_time = std::chrono::steady_clock::now();
    // lu_decomposition(n, A, L, rank, num_procs);
    // auto end_time = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end_time - start_time;
    // double seconds = diff.count();
    if(rank == 0){ 
        save(fsave, L, n, n, 'L');
        save(fsave, A, n, n, 'U');
    }
    if(rank == 0 && check_correct){
        check_correctness(A_copy, L, A, n);
        // print_matrix(L, n, n, 'L');
        // print_matrix(A, n, n, 'U');
        // print_matrix(A_copy, n, n, 'A');
    }
    if (rank == 0) {
        std::stringstream stats;
        stats << "Average computation time = " << total_compute_time << " seconds for " << n 
                  << " x " << n << " matrices.\n";
        std::cout << stats.str();
    }
    MPI_Finalize();
}

