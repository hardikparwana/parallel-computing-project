#include <iostream>
// #include <double.h>
#include <assert.h>
#include <chrono>
// #include <cmath>
// #include <random>

using namespace std;

#define SIZE 2
#define NUM 1500
#define THREADS 100

__global__ void cuda_mean( double* in_vectors, int n, int N, double *out_vector ){
	/*
	N: size of single vector
	n: number of vectors
	*/

    __shared__ double temp_sum[THREADS * SIZE];// 1024 by default //sum done by each thread in shared memory //
	__shared__ double temp_sum2[THREADS * SIZE];
    // printf("%d, %d, %f %f \n", threadIdx.x, threadIdx.x + n, in_vectors[threadIdx.x], in_vectors[threadIdx.x + n]);
	// each thread does computation for all the elements

	for (int j=0; j<N; j++){
		temp_sum[threadIdx.x + THREADS*j] = 0;
		temp_sum2[threadIdx.x + THREADS*j] = 0;
	}						

	int num_elements_per_thread = (n + THREADS - 1) / THREADS;

	// first collapse to 1024 elements only
	for (int j=0; j<N; j++){
		for (int i=0; i<num_elements_per_thread; i++){
				if ( (threadIdx.x + i*THREADS) < n ){
					temp_sum[ j*THREADS + threadIdx.x ] = temp_sum[ j*THREADS + threadIdx.x ] + in_vectors[ threadIdx.x + i*THREADS + j*n  ]; // threadIdx.x + i*1024 is vector number and we need j-th element of it
				}
		}
	}// this will ensure all min(n,1024 active threads)
    // printf("id: %d, %f, %f ", threadIdx.x, temp_sum[threadIdx.x], temp_sum[threadIdx.x+1024]);
	// now we have max 1024 vectors only. but last few columns empty possibly
	__syncthreads(); // do this to have left 1024 max elements that now need to be summed

	int elements_per_thread = 5; // no reason for particularly choosing 5 but we will be now summing in groups of 5. 
	// this reduces the size of array by 5 every time and the procedure is repetated until less than 5 threads remain

	int active_threads = 0;
	if (n>=THREADS)
		active_threads = THREADS;
	else
		active_threads = n;
	int new_active_threads = 1 + (active_threads-1) / elements_per_thread; // active threads after summed in groups of 5
	
	// temp_sum2[threadIdx.x] = 0;
	__syncthreads();

	while (active_threads > 1){//elements_per_thread){
		int startIdx = threadIdx.x * elements_per_thread; // start vector number. now take 5 consecutive vectors

		if (startIdx < active_threads){

			for (int j=0; j<N; j++){
				for (int i=0; i<elements_per_thread; i++){
					int targetIdx = startIdx + i;
					if ( targetIdx < active_threads ){
						temp_sum2[j*THREADS + startIdx] = temp_sum2[j*THREADS + startIdx] + temp_sum[j*THREADS + targetIdx];
					}
				}
			}			
		}

		// copy results from temp_sum2 to temp_sum to repeat the procedure in next iteration
		__syncthreads();
		if (startIdx < active_threads){
			for (int j=0; j<N; j++){
					temp_sum[j*THREADS + threadIdx.x] = temp_sum2[j*THREADS + startIdx];
			}
		}

		__syncthreads();
		for (int j=0;j<N;j++){
			temp_sum2[threadIdx.x + j*THREADS] = 0;
		}
		active_threads = new_active_threads;
		new_active_threads = 1 + (active_threads-1) / elements_per_thread;
		__syncthreads();

	}

	// now serial sum the remaining elements
	for (int j=0; j<N; j++){
		out_vector[j] = temp_sum[j*THREADS];
	}	

}

__global__ void cuda_covariance( double* in_vectors, int n, int N, double* mu, double *out_matrix ){

		/*
		N: size of single vector
		n: number of vectors
		*/  // assume less than 1024 vectors.. gross simplification
        __shared__ double in_matrix[THREADS * SIZE * SIZE];
		__shared__ double temp_sum[THREADS * SIZE * SIZE];// 1024 by default //sum done by each thread in shared memory //
		__shared__ double temp_sum2[THREADS * SIZE * SIZE];

		// each thread does computation for all the elements
		for (int j=0; j<N*N; j++){
				in_matrix[ threadIdx.x + THREADS * j  ] = 0;
				temp_sum[ threadIdx.x + THREADS * j] = 0;		
				temp_sum2[ threadIdx.x + THREADS * j] = 0;	
		}
        
		int num_elements_per_thread = (n + THREADS - 1) / THREADS;
        // if (threadIdx.x == 0)
        //     printf("ele per thread: %d \n", num_elements_per_thread);
	
		// subtract mu from particles first
		__syncthreads();
			for (int i=0; i<num_elements_per_thread; i++){			
                for (int j=0; j<N; j++){
				if ((threadIdx.x + i*THREADS) < n){  // vector number = round robbin fashion. (not consecutive vectors)
                // printf("id: %d, element no: %d \n", threadIdx.x, j*n + threadIdx.x);
					in_vectors[ j*n + threadIdx.x + i*THREADS ] = in_vectors[ j*n + threadIdx.x + i*THREADS ] - mu[j]/n;
                    // if (threadIdx.x == 1000)
                    //     printf("id: %d, element no: %d, element: %f \n", threadIdx.x, j*n + threadIdx.x + i*1024, in_vectors[ j*n + threadIdx.x + i*1024 ]);
				}
			}
		}
		__syncthreads();
		//
		// compute N^2 matrices for each vector. each thread takes num_elements_per_thread
		int vector_start_number = threadIdx.x * num_elements_per_thread; // start vector number

        // make matrix elements and collapse to 1024 elements at the same time
		for (int j=0;j<N;j++){
			for (int k=0;k<N;k++){

                for (int i=0; i<num_elements_per_thread; i++){ // vector number
                    if ((threadIdx.x + i*THREADS) < n){                
                        // j,k element of N^ matrix to be computed here					
                            // in_matrix[ (threadIdx.x + i*1024)*N*N + j*N + k ] = in_matrix[ (threadIdx.x + i*1024)*N*N + j*N + k ] + in_vectors[ threadIdx.x + i*1024 + j*n ] * in_vectors[ threadIdx.x + i*1024 + k*n ];
                            int element_number = j*N + k;
                            int Idx_mod = threadIdx.x + i*THREADS;
                            if (Idx_mod >= THREADS)
                                Idx_mod = Idx_mod % THREADS;
                            // in_matrix[ (threadIdx.x + i*1024) + element_number * 1024 ] = in_matrix[ (threadIdx.x + i*1024) + element_number * 1024 ] + in_vectors[ threadIdx.x + i*1024 + j*n ] * in_vectors[ threadIdx.x + i*1024 + k*n ];
                            temp_sum[ (Idx_mod) + element_number * THREADS ] = temp_sum[ (Idx_mod) + element_number * THREADS ] + in_vectors[ threadIdx.x + i*THREADS + j*n ] * in_vectors[ threadIdx.x + i*THREADS + k*n ];
                            // 4 + 4 : 8 elements
                            // if (threadIdx.x == 0)
                            //     printf(" id: %d, element number: %d , element: %f, vector j:%d, vector k: %d \n", threadIdx.x, (threadIdx.x + i*1024) + element_number * 1024, in_matrix[ (threadIdx.x + i*1024) + element_number * 1024 ], threadIdx.x + i*1024 + j*n, threadIdx.x + i*1024 + k*n);
                        } // each matrix takes N*N elements
                    }				
			}
		}// this will ensure all min(n,1024 active threads)
		__syncthreads(); // do this to have left 1024 max elements that now need to be summed

		//collapse first to 1024 matrices now (elements), each vector N*N elements
        // int bound  = 1024;
        // if (n <= 1024)
        //     bound = n;
		// int startIdx = threadIdx.x;
		// for (int j=0; j<(N*N); j++){
		// 		for (int i=0; i<num_elements_per_thread; i++){
		// 			if ((threadIdx.x + i*1024) < bound){ //(not consecutive vectors)
		// 				temp_sum[ threadIdx.x + j*1024 ] = temp_sum[ threadIdx.x + j*1024 ] + in_matrix[ threadIdx.x + i*1024 + j*1024 ];
        //                 if (threadIdx.x==0)
        //                     printf("collapsed id: %d, element number: %d, element: %f, matrix col: %d \n", threadIdx.x, j*1024 + threadIdx.x, temp_sum[j*1024 + threadIdx.x], threadIdx.x + i*1024 + j*1024);
		// 			}
		// 	}
		// }// this will ensure all min(n,1024 active threads)

		// __syncthreads(); // do this to have left 1024 max elements that now need to be summed

		// now add all matrices. treat them same as vectors with N^2 elements

		int elements_per_thread = 5; // no reason for particularly choosing 5 but we will be now summing in groups of 5. 
		// this reduces the size of array by 5 every time and the procedure is repetated until less than 5 threads remain

		int active_threads = 0;
		if (n>=THREADS)
			active_threads = THREADS;
		else
			active_threads = n;
		int new_active_threads = 1 + (active_threads-1) / elements_per_thread; // active threads after summed in groups of 5
		
		temp_sum2[threadIdx.x] = 0;
		__syncthreads();

		while (active_threads > 1){//elements_per_thread){
		int startIdx = threadIdx.x * elements_per_thread; // start vector number. now take 5 consecutive vectors
		if (startIdx < active_threads){
			for (int j=0; j<N*N; j++){
				for (int i=0; i<elements_per_thread; i++){
					int targetIdx = startIdx + i;
					if ( targetIdx < active_threads ){
						temp_sum2[j*THREADS + startIdx] = temp_sum2[j*THREADS + startIdx] + temp_sum[j*THREADS + targetIdx];
					}
				}
			}			
		}

		// copy results from temp_sum2 to temp_sum to repeat the procedure in next iteration
		__syncthreads();
		if (startIdx < active_threads){
			for (int j=0; j<N*N; j++){
					temp_sum[j*THREADS + threadIdx.x] = temp_sum2[j*THREADS + startIdx];
                    // printf("copied id: %d, element number: %d, element: %f \n", threadIdx.x, j*1024 + threadIdx.x, temp_sum[j*1024 + threadIdx.x]);
			}
		}

		__syncthreads();
		for (int j=0;j<N*N;j++){
			temp_sum2[threadIdx.x + j*THREADS] = 0;
		}
		active_threads = new_active_threads;
		new_active_threads = 1 + (active_threads-1) / elements_per_thread;
		__syncthreads();

	}

	// now serial sum the remaining elements
	for (int j=0; j<(N*N); j++){
		out_matrix[j] = temp_sum[j*THREADS];
	}	

}

int main(){

 
    int n = NUM;
    int N = SIZE;
    
    double init_mean = 0;
    double init_std = 1.0;
    // cout << "hello" << endl;
    // initialize vectors
    int size = N*n * sizeof(double);
    double *vectors = (double *)malloc(size);
    double *d_vectors;
    cout << "SIZE:  " << N*n << endl;
    if (cudaMalloc(&d_vectors, size) != cudaSuccess){
		cout << "Could not allocate d_A" << endl;
	}
    
    double *d_mean;
    double *d_cov;
    // cout << "hello1" << endl;
    // std::random_device rd{};
    // std::mt19937 gen{rd()};
    // std::normal_distribution<> d{init_mean,init_std};
    for (int i=0; i<n; i++){
        for (int j=0; j<N; j++){
            vectors[i + j*n] = i + sin(j*i);
            // if (i<500){
            //     vectors[i + j*n] = j*i;//d(gen);
            // }
            // else{
            //     vectors[i + j*n] = j+1;//d(gen);
            // }
            
        }
    }
    // vectors[0] = 3; vectors[1] = 6; vectors[2] = 4;
    // vectors[3] = 7; vectors[4] = 12; vectors[5] = -9;
    // for (int i=0; i<N*n; i++){
    //     cout << "vectors " << i << "\t" << vectors[i] << endl;
    // }
    // cout << "hello2" << endl;
    // X^T A X + B
   
    ///////////////// Serial Computation ////////////////////////////

    // Mean
    double mean[N];
    double mean_out[N];
    double cov[N*N];
    double cov_out[N*N];

    for (int i=0; i<N; i++){
        mean[i] = 0;
        for (int j=0; j<N; j++){
            cov[i*N+j] = 0;
        }
    }
    for (int i=0; i<n; i++){
        for (int j=0; j<N; j++){
            // cout << "element number: " << (i+n*j) << endl;
            mean[j] = mean[j] + vectors[ i + n*j ];
        }
    }
    cout << "hello3" << "\t" << mean[0]/n << "\t" << mean[1]/n << "\t" << mean[2]/n << "\t" << mean[3]/n << endl;
    
    for (int i=0; i<n; i++){
        for (int j=0; j<N; j++){
            for (int k=0; k<N; k++){
                cov[j*N + k] = cov[j*N + k] + (vectors[ i + n*j ] - mean[j]/n ) * ( vectors[ i + n*k ] - mean[k]/n );
            }
        }
    }
    cout << "hello4" << "\t" << cov[0]/(n-1) << "\t" << cov[1]/(n-1) << "\t" << cov[2]/(n-1) << "\t" << cov[3]/(n-1) << endl;
    //////////////// Parallel Computation ////////////////////////
    int num_threads = n;
    int num_blocks = 1;
    int max_threads_per_block = THREADS;
    // int num_blocks = ( num_threads + max_threads_per_block - 1 ) / max_threads_per_block;

    // Initialize data to device
    // cout << "hello y0" << endl;
    // int size = N*n;
	
    // cout << "hello y1" << endl;
    if (cudaMalloc(&d_mean, N*sizeof(double)) != cudaSuccess){
		cout << "Could not allocate d_mean" << endl;
	}
    // cout << "hello y2" << endl;
    if (cudaMalloc(&d_cov, N*N*sizeof(double)) != cudaSuccess){
		cout << "Could not allocate d_cov" << endl;
	}
    // cout << "hello y" << endl;
    if (cudaMemcpy( d_vectors, vectors, N*n*sizeof(double), cudaMemcpyHostToDevice ) !=cudaSuccess){
		cout << "Could not copy vectors to d_vectors" << endl;
	}
	cout << "CUDA kernel call for matrix operations: num blocks: " << num_blocks << " threads " << max_threads_per_block << std::endl;	

    cuda_mean<<< num_blocks, THREADS >>>( d_vectors, n, N, d_mean );
    cuda_covariance<<< num_blocks, THREADS >>>( d_vectors, n, N, d_mean, d_cov );

    cout << "hello5" << endl;
    // copy output to input matrix
    cudaMemcpy(mean_out, d_mean, N*sizeof(double), cudaMemcpyDeviceToHost  );	
    cudaMemcpy(cov_out, d_cov, N*N*sizeof(double), cudaMemcpyDeviceToHost  );	

    cout << "results mean: " << mean_out[0]/n << "\t" << mean_out[1]/n << endl;
    cout << "results cov: " << cov_out[0]/(n-1) << "\t" << cov_out[1]/(n-1) << "\t" << cov_out[2]/(n-1) << "\t" << cov_out[3]/(n-1) << endl;

    return 0;
    


}