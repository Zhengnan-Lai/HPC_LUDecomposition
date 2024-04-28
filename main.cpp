#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <time.h>
#include <mpi.h>

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

void printMatrix(double* A, int n, int m, char name) {
	printf("[%c]\n", name);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            printf("%lf ", A[i * n + j]);
        }
        printf("\n");
    }
	printf("\n");
}

void generateMatrix(double* A, int n, int seed){
    std::random_device rd;
    std::mt19937 gen(seed ? seed : rd());
    for (int i = 0; i < n * n; i++) {
		std::uniform_real_distribution<float> rand_real(-10.0, 10.0);
        A[i] = rand_real(gen);
	}
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
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);
    int n = find_int_arg(argc, argv, "-n", 50);
    int seed = find_int_arg(argc, argv, "-s", 0);

    int num_procs, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double* A = (double *) malloc(sizeof(double) * n * n);
    double* L = (double *) malloc(sizeof(double) * n * n);
    double* U = (double *) malloc(sizeof(double) * n * n);
    generateMatrix(A, n, seed);
    if(rank == 0){
        save(fsave, A, n, n, 'A');
    }
    auto start_time = std::chrono::steady_clock::now();
    lu_decomposition(n, A, L, rank, num_procs);
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    if(rank == 0){ 
        save(fsave, L, n, n, 'L');
        save(fsave, A, n, n, 'U');
    }
    
    if (rank == 0) {
        std::cout << "Average computation time = " << seconds << " seconds for " << n 
                  << " x " << n << " matrices.\n";
    }
    MPI_Finalize();
}

