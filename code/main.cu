#include <iostream>
#include <float.h>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <random>

using namespace std;

#define SIZE 2

__global__ void cuda_mean( float* in_vectors, int n, int N, float *out_vector ){
	/*
	N: size of single vector
	n: number of vectors
	*/

    // __shared__ float temp_sum[1024 * SIZE];// 1024 by default //sum done by each thread in shared memory //
	// __shared__ float temp_sum2[1024 * SIZE];
    printf("%d, %d, %f %f \n", threadIdx.x, threadIdx.x + n, in_vectors[threadIdx.x], in_vectors[threadIdx.x + n]);
	// each thread does computation for all the elements
/*
	for (int j=0; j<N; j++){
		temp_sum[threadIdx.x + 1024*j] = 0;
		temp_sum2[threadIdx.x + 1024*j] = 0;
	}						// temp_sum[ j*1024 + threadIdx.x ] = temp_sum[ j*1024 + threadIdx.x ] + in_particles[ threadIdx.x + i*1024 + j*n  ]; 
						//temp_sum[ j*1024 + threadIdx.x ]
	int num_elements_per_thread = (n + 1024 - 1) / 1024;

	// first collapse to 1024 elements only
	for (int j=0; j<N; j++){
		for (int i=0; i<num_elements_per_thread; i++){
				if ( (threadIdx.x + i*1024) < n ){
					temp_sum[ j*1024 + threadIdx.x ] = temp_sum[ j*1024 + threadIdx.x ] + in_vectors[ threadIdx.x + i*1024 + j*n  ]; // threadIdx.x + i*1024 is vector number and we need j-th element of it
				}
		}
	}// this will ensure all min(n,1024 active threads)

	// now we have max 1024 vectors only. but last few columns empty possibly
	__syncthreads(); // do this to have left 1024 max elements that now need to be summed

	int elements_per_thread = 5; // no reason for particularly choosing 5 but we will be now summing in groups of 5. 
	// this reduces the size of array by 5 every time and the procedure is repetated until less than 5 threads remain

	int active_threads = 0;
	if (n>=1024)
		active_threads = 1024;
	else
		active_threads = n;
	int new_active_threads = 1 + (active_threads-1) / elements_per_thread; // active threads after summed in groups of 5
	
	temp_sum2[threadIdx.x] = 0;
	__syncthreads();

	while (active_threads > 1){//elements_per_thread){
		int startIdx = threadIdx.x * elements_per_thread; // start vector number. now take 5 consecutive vectors
		if (startIdx < active_threads){
			for (int j=0; j<N; j++){
				for (int i=0; i<elements_per_thread; i++){
					int targetIdx = startIdx + i;
					if ( targetIdx < active_threads ){
						temp_sum2[j*1024 + startIdx] = temp_sum2[j*1024 + startIdx] + temp_sum[j*1024 + targetIdx];
					}
				}
			}			
		}

		// copy results from temp_sum2 to temp_sum to repeat the procedure in next iteration
		__syncthreads();
		if (startIdx < active_threads){
			for (int j=0; j<N; j++){
					temp_sum[j*1024 + threadIdx.x] = temp_sum2[j*1024 + startIdx];
			}
		}

		__syncthreads();
		for (int j=0;j<N;j++){
			temp_sum2[threadIdx.x + j*1024] = 0;
		}
		active_threads = new_active_threads;
		new_active_threads = 1 + (active_threads-1) / elements_per_thread;
		__syncthreads();

	}

	// now serial sum the remaining elements
	for (int j=0; j<N; j++){
		out_vector[j] = temp_sum[j];
	}	
*/
}

__global__ void cuda_covariance( float* in_vectors, int n, int N, float* mu, float *out_matrix ){

		/*
		N: size of single vector
		n: number of vectors
		*/  // assume less than 1024 vectors.. gross simplification
        __shared__ float in_matrix[1024 * SIZE * SIZE];
		__shared__ float temp_sum[1024 * SIZE * SIZE];// 1024 by default //sum done by each thread in shared memory //
		__shared__ float temp_sum2[1024 * SIZE * SIZE];

		// each thread does computation for all the elements
		for (int j=0; j<N*N; j++){
				in_matrix[ threadIdx.x + 1024 * j  ] = 0;
				temp_sum[ threadIdx.x + 1024 * j] = 0;		
				temp_sum2[ threadIdx.x + 1024 * j] = 0;	
		}
        
		int num_elements_per_thread = (n + 1024 - 1) / 1024;
	
		// subtract mu from particles first
		for (int j=0; j<N; j++){
			for (int i=0; i<num_elements_per_thread; i++){			
				if ((threadIdx.x + i*1024) < n){  // vector number = round robbin fashion. (not consecutive vectors)
					in_vectors[ j*n + threadIdx.x + i*2024 ] = in_vectors[ j*n + threadIdx.x + i*1024 ] - mu[j];
				}
			}
		}
		__syncthreads();
		//

		// XX^: N^2 elements
		// total: nN^2, now sum all them up 
		// compute N^2 matrices for each vector. each thread takes num_elements_per_thread
		//N^2 * n
		int vector_start_number = threadIdx.x * num_elements_per_thread; // start vector number

		for (int j=0;j<N;j++){
			for (int k=0;k<N;k++){

			for (int i=0; i<num_elements_per_thread; i++){ // vector number

				if ((threadIdx.x + i*1024) < n){
			
					// j,k element of N^ matrix to be computed here					
						in_matrix[ threadIdx.x + i*1024 + j*N + k ] = in_matrix[ threadIdx.x + i*1024 + j*N + k ] + in_vectors[ threadIdx.x + i*1024 + j*n ] * in_vectors[ threadIdx.x + i*1024 + k*n ];
					} // each matrix takes N*N elements
				}				
			}
		}// this will ensure all min(n,1024 active threads)
		__syncthreads(); // do this to have left 1024 max elements that now need to be summed

		//collapse first to 1024 matrices now
		int startIdx = threadIdx.x;
		for (int j=0; j<(N*N); j++){
				for (int i=0; i<num_elements_per_thread; i++){
					if ((threadIdx.x + i*1024) < n){ //(not consecutive vectors)
						temp_sum[ threadIdx.x + j*1024 ] = temp_sum[ threadIdx.x + j*1024 ] + in_matrix[ threadIdx.x + i*1024 + j*n ];
					}
			}
		}// this will ensure all min(n,1024 active threads)

		__syncthreads(); // do this to have left 1024 max elements that now need to be summed

		// now add all matrices. treat them same as vectors with N^2 elements

		int elements_per_thread = 5; // no reason for particularly choosing 5 but we will be now summing in groups of 5. 
		// this reduces the size of array by 5 every time and the procedure is repetated until less than 5 threads remain

		int active_threads = 0;
		if (n>=1024)
			active_threads = 1024;
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
						temp_sum2[j*1024 + startIdx] = temp_sum2[j*1024 + startIdx] + temp_sum[j*1024 + targetIdx];
					}
				}
			}			
		}

		// copy results from temp_sum2 to temp_sum to repeat the procedure in next iteration
		__syncthreads();
		if (startIdx < active_threads){
			for (int j=0; j<N*N; j++){
					temp_sum[j*1024 + threadIdx.x] = temp_sum2[j*1024 + startIdx];
			}
		}

		__syncthreads();
		for (int j=0;j<N*N;j++){
			temp_sum2[threadIdx.x + j*1024] = 0;
		}
		active_threads = new_active_threads;
		new_active_threads = 1 + (active_threads-1) / elements_per_thread;
		__syncthreads();

	}

	// now serial sum the remaining elements
	for (int j=0; j<(N*N); j++){
		out_matrix[j] = temp_sum[j];
	}	

}

int main(){

    int n = 5;
    int N = 2;
    
    float init_mean = 0;
    float init_std = 1.0;
    cout << "hello" << endl;
    // initialize vectors
    int size = N*n;
    float *vectors = (float *)malloc(size);
    float *d_vectors;
    cout << "SIZE:  " << N*n << endl;
    if (cudaMalloc(&d_vectors, N*n) != cudaSuccess){
		cout << "Could not allocate d_A" << endl;
	}
    float *d_mean;
    float *d_cov;
    cout << "hello1" << endl;
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{init_mean,init_std};
    for (int i=0; i<n; i++){
        for (int j=0; j<N; j++){
            vectors[i + j*n] = 1.0;//d(gen);
        }
    }
    for (int i=0; i<N*n; i++){
        cout << "vectors " << i << "\t" << vectors[i] << endl;
    }
    cout << "hello2" << endl;
    // X^T A X + B
   
    ///////////////// Serial Computation ////////////////////////////

    // Mean
    float mean[N];
    float mean_out[N];
    float cov[N*N];
    float cov_out[N*N];

    for (int i=0; i<N; i++){
        mean[i] = 0;
        for (int j=0; j<N; j++){
            cov[i*n+j] = 0;
        }
    }
    for (int i=0; i<n; i++){
        for (int j=0; j<N; j++){
            cout << "element number: " << (i+n*j) << endl;
            mean[j] = mean[j] + vectors[ i + n*j ];
        }
    }
    cout << "hello3" << "\t" << mean[0] << "\t" << mean[1] << endl;
    
    for (int i=0; i<n; i++){
        for (int j=0; j<N; j++){
            for (int k=0; k<N; k++){
                cov[j*N + k] = cov[j*N + k] + vectors[ i + n*j ] * vectors[ i + n*k ];
            }
        }
    }
    cout << "hello4" << endl;
    //////////////// Parallel Computation ////////////////////////
    int num_threads = n;
    int num_blocks = 1;
    int max_threads_per_block = n;//1024;
    // int num_blocks = ( num_threads + max_threads_per_block - 1 ) / max_threads_per_block;

    // Initialize data to device
    cout << "hello y0" << endl;
    // int size = N*n;
	
    cout << "hello y1" << endl;
    if (cudaMalloc(&d_mean, N) != cudaSuccess){
		cout << "Could not allocate d_A" << endl;
	}
    cout << "hello y2" << endl;
    if (cudaMalloc(&d_cov, N*N) != cudaSuccess){
		cout << "Could not allocate d_A" << endl;
	}
    cout << "hello y" << endl;
    if (cudaMemcpy( d_vectors, vectors, N*n, cudaMemcpyHostToDevice ) !=cudaSuccess){
		cout << "Could not copy A to d_A" << endl;
	}
    // for (int i=0; i<(N*n);i++){
    //     cout << "cuda " << i << "\t" << d_vectors[i] << endl;
    // }
	cout << "CUDA kernel call for matrix operations: num blocks: " << num_blocks << " threads " << num_threads << std::endl;	

    cuda_mean<<< num_blocks, max_threads_per_block >>>(d_vectors, n, N, d_mean);
    // cuda_covariance<<< num_blocks, max_threads_per_block >>>(d_vectors, n, N, d_mean, d_cov);

    cout << "hello5" << endl;
    // copy output to input matrix
    cudaMemcpy(mean_out, d_mean, N, cudaMemcpyDeviceToHost  );	
    cudaMemcpy(cov_out, d_cov, N*N, cudaMemcpyDeviceToHost  );	

    cout << "results mean: " << mean_out[0] << "\t" << mean_out[1] << endl;

    // cout << "hello6" << endl;


}