#include "include/tensor_product.hpp"

void tensor_product2( double* v, double tensor[][DIM] ){
	for (int i=0; i<DIM; i++){
		for (int j=0; j<DIM; j++){
			tensor[i][j] = v[i] * v[j];
		}
	}
}

void tensor_product3( double* v, double*** tensor ){ // int k
	// k: is only the dimension of the matrix: what matters is 1 to n only
	for (int h=0; h<DIM; h++){
		for (int i=0; i<DIM; i++){
			for (int j=0; j<DIM; j++){
				tensor[h][i][j] = v[h] * v[i] * v[j];
			}
		}
	}
	
}

void tensor_product4( double* v, double**** tensor ){
	for (int g=0; g<DIM; g++){
		for (int h=0; h<DIM; h++){
			for (int i=0; i<DIM; i++){
				for (int j=0; j<DIM; j++){
					tensor[g][h][i][j] =  v[g] * v[h] * v[i] * v[j];
				}
			}
		}
	}
}