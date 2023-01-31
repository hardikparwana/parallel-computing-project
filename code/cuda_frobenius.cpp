#include "include/cuda_frobenius.hpp"


__device__ float frobenius_norm( float *in, int n, float* out ){


	__shared__ float temp_sum[1024]; //sum done by each thread in shared memory
	__shared__ float temp_sum2[1024];

	temp_sum[threadIdx.x] = 0;
	int num_elements_per_thread = (n + 1024 - 1) / 1024;

	int startIdx = threadIdx.x;
	for (int i=0; i<num_elements_per_thread; i++){
		if ((startIdx + i*1024) < n){
			temp_sum[ threadIdx.x ] = temp_sum[ threadIdx.x ] + in[ startIdx + i*1024 ] * in[ startIdx + i*1024 ];
		}
	}// this will ensure all min(n,1024 active threads)

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
		startIdx = threadIdx.x * elements_per_thread;
		if (startIdx < active_threads){
			for (int i=0; i<elements_per_thread; i++){
				int targetIdx = startIdx + i;
				if ( targetIdx < active_threads ){
					temp_sum2[startIdx] = temp_sum2[startIdx] + temp_sum[targetIdx];
				}
			}
		}

		// copy results from temp_sum2 to temp_sum to repeat the procedure in next iteration
		__syncthreads();
		if (startIdx < active_threads){
			temp_sum[threadIdx.x] = temp_sum2[startIdx];
		}

		__syncthreads();
		temp_sum2[threadIdx.x] = 0;
		active_threads = new_active_threads;
		new_active_threads = 1 + (active_threads-1) / elements_per_thread;
		__syncthreads();

	}

	// now serial sum the remaining elements and do root
	out[0] = sqrt(temp_sum[0]);


}

// __device__ double cuda_frobenius2( double** matrix ){ // flattened 2D array

// 	double norm = 0.0;
// 	for (int i=0; i<DIM; i++){
// 		for (int j=0; j<DIM; j++){
// 				norm += matrix[i][j] * matrix[i][j];		
// 		}
// 	}
// 	norm = sqrt(norm);
//     return norm;
// }

// __device__ double cuda_frobenius3( double*** matrix ){
// 	double norm = 0.0;
// 	for (int i=0; i<DIM; i++){
// 		for (int j=0; j<DIM; j++){
// 			for (int k=0; k<DIM; k++){
// 				norm += matrix[i][j][k] * matrix[i][j][k];		
// 			}
// 		}
// 	}
// 	norm = sqrt(norm);
//     return norm;
// }

// __device__ double cuda_frobenius4( double**** matrix ){
// 	double norm; 0.0;
// 	for (int i=0; i<DIM; i++){
// 		for (int j=0; j<DIM; j++){
// 			for (int k=0; k<DIM; k++){
//                 for (int l=0; l<DIM;  l++){
//                     norm += matrix[i][j][k][l] * matrix[i][j][k][l];	
//                 }					
// 			}
// 		}
// 	}
// 	norm = sqrt(norm);
//     return norm;
// }

// if flattened, then only the total size matters for frobenius norm