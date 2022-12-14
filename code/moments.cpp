#include "include/moments.hpp"

void mean( double** particles, double *weights, int N, double *mu ){
	for (int i=0; i<DIM; i++){
		mu[i] = 0;
	}

	for (int i=0; i<N; i++){
		double* particle = particles[i];
		double weight = weights[i];
		for (int j=0; j<DIM; j++){
			mu[j] = mu[j] + weight * particle[j];
		}
	}
	return;
}

void covariance( double** particles, double *weights, int N, double* mu, double **cov ){
        for (int i=0; i<DIM; i++){
		for (int j=0; j<DIM; j++){
                	cov[i][j] = 0;
		}
        }

        for (int k=0; k<N; k++){
                double* particle = particles[k];
                double weight = weights[k];
                for (int i=0; i<DIM; i++){
			for (int j=0; j<DIM; j++){
                        	cov[i][j] = cov[i][j] + weight * (particle[i] - mu[i] ) * (  particle[j] - mu[j]);
			}
                }
        }
        return;
}

void skewness( double** particles, double* weights, int N, double* mu, double*** skew ){
	bool initialzied = false;

	for (int i=0; i<DIM; i++){
		for (int j=0; j<DIM; j++){
			for (int l=0;l<DIM;l++){
				skew[i][j][l] = 0;
			}
		}
	}

	for (int k=0; k<N; k++){
		double* particle = particles[k];
		double weight = weights[k];
		for (int i=0; i<DIM; i++){
			for (int j=0; j<DIM;j++){
				for (int l=0; l<DIM; l++){
					skew[i][j][l] = skew[i][j][l] + weight * ( particle[i]-mu[i] ) * ( particle[j]-mu[j] ) * ( particle[l]-mu[l] );
				}
			}
		}
	}
}

void kurtosis( double** particles, double *weights, int N, double* mu, double**** kurt  ){

	bool initialzied = false;

	for (int i=0; i<DIM; i++){
		for (int j=0; j<DIM; j++){
			for (int l=0;l<DIM;l++){
					for (int m=0; m<DIM; m++){
						kurt[i][j][l][m] = 0;
					}
			}
		}
	}

	for (int k=0; k<N; k++){
		double* particle = particles[k];
		double weight = weights[k];
		for (int i=0; i<DIM; i++){
			for (int j=0; j<DIM;j++){
				for (int l=0; l<DIM; l++){
					for (int m=0; m<DIM; m++){
						kurt[i][j][l][m] = kurt[i][j][l][m] + weight * ( particle[i]-mu[i] ) * ( particle[j]-mu[j] ) * ( particle[l]-mu[l] ) * ( particle[m]-mu[m] );
					}
				}
			}
		}
	}


}
