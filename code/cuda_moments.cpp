#include "include/moments.hpp"


__device__ mean( double* in_vectors, int n, int N, double *out_vector ){
	/*
	N: size of single vector
	n: number of vectors
	*/

	extern __shared__ float temp_sum[1024 * N];// 1024 by default //sum done by each thread in shared memory //
	extern __shared__ float temp_sum2[1024 * N];

	// each thread does computation for all the elements

	for (int j=0; j<N; j++){
		temp_sum[threadIdx.x + 1024*j] = 0;
		temp_sum2[threadIdx.x + 1024*j] = 0;
	}						// temp_sum[ j*1024 + threadIdx.x ] = temp_sum[ j*1024 + threadIdx.x ] + in_particles[ threadIdx.x + i*1024 + j*n  ]; 
						//temp_sum[ j*1024 + threadIdx.x ]
	int num_elements_per_thread = (n + 1024 - 1) / 1024;

	int myIdx = threadIdx.x;
	// first collapse to 1024 elements only
	for (int j=0; j<N; j++){
		for (int i=0; i<num_elements_per_thread; i++){
				if ( (threadIdx.x + i*1024) < n ){
					temp_sum[ j*1024 + threadIdx.x ] = temp_sum[ j*1024 + threadIdx.x ] + in_particles[ threadIdx.x + i*1024 + j*n  ]; // threadIdx.x + i*1024 is vector number and we need j-th element of it
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
		out[j] = temp_sum[j];
	}	

}

void covariance( double* in_vectors, int n, int N, double* mu, double *out_vector ){

		/*
		N: size of single vector
		n: number of vectors
		*/  // assume less than 1024 vectors.. gross simplification
		extern __shared__ float in_matrix[1024 * N * N];
		extern __shared__ float temp_sum[1024 * N * N];// 1024 by default //sum done by each thread in shared memory //
		extern __shared__ float temp_sum2[1024 * N * N];

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
		int vector_start_number = threadIdx.x * num_elements_per_thread // start vector number

		for (int j=0;j<N;j++){
			for (int k=0;k<N;k++){

			for (int i=0; i<num_elements_per_thread; i++){ // vector number

				if ((threadIdx.x + i*1024) < n){
			
					// j,k element of N^ matrix to be computed here					
						in_matrix[ threadIdx.x + i*1024 + j*N + k ] = in_matrix[ threadIdx.x + i*1024 + j*N + k ] + in[ threadIdx.x + i*1024 + j*n ] * in[ threadIdx.x + i*1024 + k*n ];
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
	for (int j=0; j<N; j++){
		out[j] = temp_sum[j];
	}	

}
