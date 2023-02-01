#include <iostream>
#include <assert.h>
#include <chrono>

using namespace std;

#define SIZE 2
#define NUM 1000
#define THREADS 100

__global__ void cuda_mean( double* in_vectors, int n, int N, double *out_vector ){
	/*
	N: size of single vector
	n: number of vectors
	*/

    __shared__ double temp_sum[THREADS * SIZE];// 1024 by default //sum done by each thread in shared memory //
	__shared__ double temp_sum2[THREADS * SIZE];
   
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
	}
	__syncthreads(); // do this to have left 1024 max elements that now need to be summed

	int elements_per_thread = 5; // no reason for particularly choosing 5 but we will be now summing in groups of 5. 
	// this reduces the size of array by 5 every time and the procedure is repetated until less than 5 threads remain

	int active_threads = 0;
	if (n>=THREADS)
		active_threads = THREADS;
	else
		active_threads = n;
	int new_active_threads = 1 + (active_threads-1) / elements_per_thread; // active threads after summed in groups of 5
	
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
       
		// subtract mu from particles first
		__syncthreads();
			for (int i=0; i<num_elements_per_thread; i++){			
                for (int j=0; j<N; j++){
				if ((threadIdx.x + i*THREADS) < n){  // vector number = round robbin fashion. (not consecutive vectors)
					in_vectors[ j*n + threadIdx.x + i*THREADS ] = in_vectors[ j*n + threadIdx.x + i*THREADS ] - mu[j]/n;
				}
			}
		}
		__syncthreads();
		
		// compute N^2 matrices for each vector. each thread takes num_elements_per_thread
		int vector_start_number = threadIdx.x * num_elements_per_thread; // start vector number

        // make matrix elements and collapse to 1024 elements at the same time
		for (int j=0;j<N;j++){
			for (int k=0;k<N;k++){

                for (int i=0; i<num_elements_per_thread; i++){ // vector number
                    if ((threadIdx.x + i*THREADS) < n){                
                            int element_number = j*N + k;
                            int Idx_mod = threadIdx.x + i*THREADS;
                            if (Idx_mod >= THREADS)
                                Idx_mod = Idx_mod % THREADS;
                            temp_sum[ (Idx_mod) + element_number * THREADS ] = temp_sum[ (Idx_mod) + element_number * THREADS ] + in_vectors[ threadIdx.x + i*THREADS + j*n ] * in_vectors[ threadIdx.x + i*THREADS + k*n ];
                        } // each matrix takes N*N elements
                    }				
			}
		}// this will ensure all min(n,1024 active threads)
		__syncthreads(); // do this to have left 1024 max elements that now need to be summed

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

	cout << "NUM: " << NUM << " SIZE: " << SIZE << " threads: " << THREADS << endl;
    
    double init_mean = 0;
    double init_std = 1.0;
   
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

    for (int i=0; i<n; i++){
        for (int j=0; j<N; j++){
            vectors[i + j*n] = i + sin(j*i);
            
        }
    }

    ///////////////// Serial Computation ////////////////////////////

    // Mean
    double mean[N];
    double mean_out[N];
    double cov[N*N];
    double cov_out[N*N];

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i=0; i<N; i++){
        mean[i] = 0;
        for (int j=0; j<N; j++){
            cov[i*N+j] = 0;
        }
    }
    for (int i=0; i<n; i++){
        for (int j=0; j<N; j++){
            mean[j] = mean[j] + vectors[ i + n*j ];
        }
    }
    
    for (int i=0; i<n; i++){
        for (int j=0; j<N; j++){
            for (int k=0; k<N; k++){
                cov[j*N + k] = cov[j*N + k] + (vectors[ i + n*j ] - mean[j]/n ) * ( vectors[ i + n*k ] - mean[k]/n );
            }
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Serial Time(milliseconds) elapsed to perform mean, cov: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0 << std::endl;
    cout << "Serial mean (first 2 elements)" << "\t" << mean[0]/n << "\t" << mean[1]/n  << endl;
    cout << "Serial cov (first 2 elements)" << "\t" << cov[0]/(n-1) << "\t" << cov[1]/(n-1)  << endl;
    //////////////// Parallel Computation ////////////////////////
    int num_threads = n;
    int num_blocks = 1;
    int max_threads_per_block = THREADS;

 
    if (cudaMalloc(&d_mean, N*sizeof(double)) != cudaSuccess){
		cout << "Could not allocate d_mean" << endl;
	}   
    if (cudaMalloc(&d_cov, N*N*sizeof(double)) != cudaSuccess){
		cout << "Could not allocate d_cov" << endl;
	}   
    if (cudaMemcpy( d_vectors, vectors, N*n*sizeof(double), cudaMemcpyHostToDevice ) !=cudaSuccess){
		cout << "Could not copy vectors to d_vectors" << endl;
	}
	cout << "CUDA kernel call for matrix operations: num blocks: " << num_blocks << " threads " << max_threads_per_block << std::endl;	

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

    cuda_mean<<< num_blocks, THREADS >>>( d_vectors, n, N, d_mean );
    cuda_covariance<<< num_blocks, THREADS >>>( d_vectors, n, N, d_mean, d_cov );

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
    float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time(milliseconds) elapsed for CUDA functions: " << milliseconds << endl;

    cudaMemcpy(mean_out, d_mean, N*sizeof(double), cudaMemcpyDeviceToHost  );	
    cudaMemcpy(cov_out, d_cov, N*N*sizeof(double), cudaMemcpyDeviceToHost  );	

    cout << "results mean: " << mean_out[0]/n << "\t" << mean_out[1]/n << endl;
    cout << "results cov: " << cov_out[0]/(n-1) << "\t" << cov_out[1]/(n-1) << endl;

    return 0;
    


}