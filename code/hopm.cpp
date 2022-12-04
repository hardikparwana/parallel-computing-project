#include "include/hopm.hpp"
#include "include/unfolding.hpp"
#include "include/svd.hpp"

void HOPM3( double*** tensor, double* v1, double& lambda ){
	// reshape into d x d^{k-1} and compute left eigenvector
	int k = 3;

	double **unfolded_matrix;
	unfolded_matrix = new double* [DIM];
	for (int i=0; i<DIM; i++){
		unfolded_matrix[i] = new double[DIM * DIM];
	}
	unfolding3(tensor, unfolded_matrix);

	double* v0;
	v0 = new double[DIM];
	svd( unfolded_matrix, v0 );

	double vectors[k][DIM];
	for (int i=0; i<k;k++){
		for (int j=0; j<DIM; j++){
			vectors[i][j] = v0[j];
		}
	}
	lambda = 100000;// Infinity
	double lambda_prev = 0;
	double epsilon = 0.3;
	int iter = 0;

	double vectors_temp[k][DIM];
	for (int i=0; i<k; i++){
		for (int j=0; j<DIM; j++){
			vectors_temp[i][j] = 0;;
		}
	}

	int l;
	while ( ( abs(lambda-lambda_prev)>epsilon ) && (iter<100) ){

		for (int s=0; s<DIM; s++){

			l = 0;
			for (int i=0; i<DIM; i++){
				for (int j=0; j<DIM; j++){
						vectors_temp[l][s] = vectors[l][s] + tensor[s][i][j] * vectors[1][i] * vectors[2][j];
					}
			}

			l = 1;
			for (int h=0; h<DIM; h++){
				for (int j=0; j<DIM; j++){
						vectors_temp[l][s] = vectors[l][s] + tensor[h][s][j] * vectors[0][h] * vectors[2][j];
					}
			}

			l = 2;
			for (int h=0; h<DIM; h++){
				for (int i=0; i<DIM; i++){
						vectors_temp[l][s] = vectors[l][s] + tensor[h][i][s] * vectors[0][h] * vectors[1][i];
					}
			}
		}

		lambda_prev = lambda;
		lambda = 0;
		for (int h=0; h<DIM; h++){
			for (int i=0; i<DIM; i++){
				for (int j=0; j<DIM; j++){
					// compute new lambda
					lambda = lambda + tensor[h][i][j] * vectors_temp[0][h] * vectors_temp[0][i] * vectors_temp[0][j];
				}
			}
		}

		// copy vectors+temp tpo vectors
		for (int i=0; i<DIM; i++){
			vectors[0][i] = vectors[0][i];
			vectors[1][i] = vectors[1][i];
			vectors[2][i] = vectors[2][i];
		}

		// increment the counter
		iter += 1;
	}

	for (int i=0;i<DIM;i++){
		v1[i] = vectors[0][i];	
	}

	return;
}

void HOPM4( double**** tensor, double* v1, double& lambda ){
	// reshape into d x d^{k-1} and compute left eigenvector
	int k = 4;

	double **unfolded_matrix;
	unfolded_matrix = new double* [DIM];
	for (int i=0; i<DIM; i++){
		unfolded_matrix[i] = new double[DIM * DIM * DIM];
	}
	unfolding4(tensor, unfolded_matrix);

	double* v0;
	v0 = new double[DIM];
	svd( unfolded_matrix, v0 );

	double vectors[k][DIM];
	for (int i=0; i<k;k++){
		for (int j=0; j<DIM; j++){
			vectors[i][j] = v0[j];
		}
	}
	lambda = 100000;// Infinity
	double lambda_prev = 0;
	double epsilon = 0.3;
	int iter = 0;

	double vectors_temp[k][DIM];
	for (int i=0; i<k; i++){
		for (int j=0; j<DIM; j++){
			vectors_temp[i][j] = 0;;
		}
	}

	int l;
	while ( ( abs(lambda-lambda_prev)>epsilon ) && (iter<100) ){

		for (int s=0; s<DIM; s++){

			l = 0;
			for (int h=0; h<DIM; h++){
				for (int i=0; i<DIM; i++){
					for (int j=0; j<DIM; j++){
							vectors_temp[l][s] = vectors[l][s] + tensor[s][h][i][j] * vectors[1][h] * vectors[2][i] * vectors[3][j];
						}
				}
			}

			l = 1;
			for (int g=0; g<DIM; g++){
				for (int i=0; i<DIM; i++){
					for (int j=0; j<DIM; j++){
							vectors_temp[l][s] = vectors[l][s] + tensor[g][s][i][j] * vectors[0][g] * vectors[2][i] * vectors[3][j];
						}
				}
			}

			l = 2;
			for (int g=0; g<DIM; g++){
				for (int h=0; h<DIM; h++){
					for (int j=0; j<DIM; j++){
							vectors_temp[l][s] = vectors[l][s] + tensor[g][h][s][j] * vectors[0][g] * vectors[1][h] * vectors[3][j];
						}
				}
			}

			l = 3;
			for (int g=0; g<DIM; g++){
				for (int h=0; h<DIM; h++){
					for (int i=0; i<DIM; i++){
							vectors_temp[l][s] = vectors[l][s] + tensor[g][h][i][s] * vectors[0][g] * vectors[1][h] * vectors[2][i];
						}
				}
			}
		}

		lambda_prev = lambda;
		lambda = 0;
		for (int g=0; g<DIM; g++){
			for (int h=0; h<DIM; h++){
				for (int i=0; i<DIM; i++){
					for (int j=0; j<DIM; j++){
						// compute new lambda
						lambda = lambda + tensor[g][h][i][j] * vectors_temp[0][g] * vectors_temp[0][h] * vectors_temp[0][i] * vectors_temp[0][j];
					}
				}
			}
		}

		// copy vectors+temp tpo vectors
		for (int i=0; i<DIM; i++){
			vectors[0][i] = vectors[0][i];
			vectors[1][i] = vectors[1][i];
			vectors[2][i] = vectors[2][i];
			vectors[3][i] = vectors[3][i];
		}

		// increment the counter
		iter += 1;
	}

	for (int i=0;i<DIM;i++){
		v1[i] = vectors[0][i];	
	}

	return;
}