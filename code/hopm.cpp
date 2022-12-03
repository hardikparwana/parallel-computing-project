
#include "include/hopm.hpp"

void HOPM4( double**** tensor, double* v1, double& lambda ){
	return;
}

void HOPM3( double*** tensor, double* v1, double& lambda ){
	// reshape into d x d^{k-1} and compute left eigenvector
	int k = 3;
	double v0[DIM];
	double vectors[k][DIM];
	for (int i=0; i<k;k++){
		for (int j=0; j<DIM; j++){
			vectors[i][j] = v0[j];
		}
	}
	lambda = 100000;// Infinity
	double lambda_prev = 0;
	double epsilon = 0.3;

	while ( abs(lambda-lambda_prev)>epsilon ){
	//	for (int l=1; l<k; l++){
	//		v_s^l = T[][][]
	//	}

		for (int h=0; h<DIM; h++){
			for (int i=0; i<DIM; i++){
				for (int j=0; j<DIM; j++){
					for (int l=0; l<k; l++){
						for (int s=0; s<DIM; s++){		
							if (l==0){
								vectors[l][s] = vectors[l][s] + tensor[s][i][j] * vectors[l+1][l+1] * vectors[l+2][l+2];
							}
							else if (l==1){
								vectors[l][s] = vectors[l][s] + tensor[h][s][j] * vectors[l-1][l-1] * vectors[l+1][l+1];
							}
							else if (l==2){
								vectors[l][s] = vectors[l][s] + tensor[h][i][s] * vectors[l-2][l-2] * vectors[l-1][l-1];
							}
						}
					}
				}
			}
		}

		lambda_prev = lambda;
		lambda = 0;
		for (int h=0; h<DIM; h++){
			for (int i=0; i<DIM; i++){
				for (int j=0; j<DIM; j++){
					lambda = lambda + tensor[h][i][j] * vectors[0][0];
				}
			}
		}
	}

	for (int i=0;i<DIM;i++){
		v1[i] = vectors[0][i];	
	}

	return;
}