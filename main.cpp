#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <time.h>

// =================
// Helper Functions
// =================

// I/O routines
// void save(std::ofstream& fsave, particle_t* parts, int num_parts, double size) {
//     static bool first = true;

//     if (first) {
//         fsave << num_parts << " " << size << "\n";
//         first = false;
//     }

//     for (int i = 0; i < num_parts; ++i) {
//         fsave << parts[i].x << " " << parts[i].y << "\n";
//     }

//     fsave << std::endl;
// }

// // Particle Initialization
// void init_particles(particle_t* parts, int num_parts, double size, int part_seed) {
//     std::random_device rd;
//     std::mt19937 gen(part_seed ? part_seed : rd());

//     int sx = (int)ceil(sqrt((double)num_parts));
//     int sy = (num_parts + sx - 1) / sx;

//     std::vector<int> shuffle(num_parts);
//     for (int i = 0; i < shuffle.size(); ++i) {
//         shuffle[i] = i;
//     }

//     for (int i = 0; i < num_parts; ++i) {
//         // Make sure particles are not spatially sorted
//         std::uniform_int_distribution<int> rand_int(0, num_parts - i - 1);
//         int j = rand_int(gen);
//         int k = shuffle[j];
//         shuffle[j] = shuffle[num_parts - i - 1];

//         // Distribute particles evenly to ensure proper spacing
//         parts[i].x = size * (1. + (k % sx)) / (1 + sx);
//         parts[i].y = size * (1. + (k / sx)) / (1 + sy);

//         // Assign random velocities within a bound
//         std::uniform_real_distribution<float> rand_real(-1.0, 1.0);
//         parts[i].vx = rand_real(gen);
//         parts[i].vy = rand_real(gen);
//     }
// }

// // Command Line Option Processing
// int find_arg_idx(int argc, char** argv, const char* option) {
//     for (int i = 1; i < argc; ++i) {
//         if (strcmp(argv[i], option) == 0) {
//             return i;
//         }
//     }
//     return -1;
// }

// int find_int_arg(int argc, char** argv, const char* option, int default_value) {
//     int iplace = find_arg_idx(argc, argv, option);

//     if (iplace >= 0 && iplace < argc - 1) {
//         return std::stoi(argv[iplace + 1]);
//     }

//     return default_value;
// }

// char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
//     int iplace = find_arg_idx(argc, argv, option);

//     if (iplace >= 0 && iplace < argc - 1) {
//         return argv[iplace + 1];
//     }

//     return default_value;
// }

// // ==============
// // Main Function
// // ==============

// int main(int argc, char** argv) {
//     // Parse Args
//     if (find_arg_idx(argc, argv, "-h") >= 0) {
//         std::cout << "Options:" << std::endl;
//         std::cout << "-h: see this help" << std::endl;
//         std::cout << "-n <int>: size of matrices" << std::endl;
//         std::cout << "-o <filename>: set the output file name" << std::endl;
//         std::cout << "-s <int>: set matrix initialization seed" << std::endl;
//         return 0;
//     }

//     // Open Output File
//     char* savename = find_string_option(argc, argv, "-o", nullptr);
//     std::ofstream fsave(savename);

//     // Initialize Particles
//     int num_parts = find_int_arg(argc, argv, "-n", 1000);
//     int part_seed = find_int_arg(argc, argv, "-s", 0);
//     double size = sqrt(density * num_parts);

//     particle_t* parts = new particle_t[num_parts];

//     init_particles(parts, num_parts, size, part_seed);

//     // Algorithm
//     auto start_time = std::chrono::steady_clock::now();

//     init_simulation(parts, num_parts, size);
    
//     // time_t start = clock();
// #ifdef _OPENMP
// #pragma omp parallel default(shared)
// #endif
//     {
//         for (int step = 0; step < nsteps; ++step) {
//             simulate_one_step(parts, num_parts, size);

//             // Save state if necessary
// #ifdef _OPENMP
// #pragma omp master
// #endif
//             if (fsave.good() && (step % savefreq) == 0) {
//                 save(fsave, parts, num_parts, size);
//             }
//         }
//     }
//     // time_t end = clock();
//     auto end_time = std::chrono::steady_clock::now();

//     std::chrono::duration<double> diff = end_time - start_time;
//     double seconds = diff.count();
//     // std::cout << "Computation Time = " <<  ((double) (end - start) / CLOCKS_PER_SEC) << " seconds\n";
//     // Finalize
//     std::cout << "Simulation Time = " << seconds << " seconds for " << num_parts << " particles.\n";
//     fsave.close();
//     delete[] parts;
// }

void lu_decomposition(int n, double* A, double* L, double* U);

int main(int argc, char** argv){
    int n = 4;
    double* A = (double *) malloc(sizeof(double) * n * n);
    double* L = (double *) malloc(sizeof(double) * n * n);
    double* U = (double *) malloc(sizeof(double) * n * n);
    for(int i = 1; i <= n * n; i++) A[i-1] = (i*i) % 10;
    lu_decomposition(n, A, L, U);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%lf ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%lf ", L[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%lf ", U[i * n + j]);
        }
        printf("\n");
    }
}