#include <iostream>
#include <random>
#include <cmath>

#include <cuda.h>
#include <curand_kernel.h>

#define DIM 3

using namespace std;

typedef struct prediction
{
    double pred[2];
}pred;

pred gaussian_process(double* state){

	// design control input?
	pred out;
	out.pred[0] = 6;
	out.pred[1] = 4;
	return out;
	
}

void step_random(double *current_state, int dim, double dt, double *next_state){

    std::random_device rd{};
    std::mt19937 gen{rd()};

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    for (int i=0; i<dim; i++)
    {
	pred gp = gaussian_process( current_state );
	double mean = gp.pred[0];
	double std = gp.pred[1];
	std::normal_distribution<> d{mean,std};

	double state_dot = d(gen);
	next_state[i] = current_state[i] + state_dot * dt;
    }

}

void generate_new( double* particle, double temp_particles[][DIM], double* temp_weights, int N){
	for (int k=0; k<N; k++){
		for (int i=0; i<DIM; i++){
			temp_particles[k][i] = particle[i];
			temp_weights[k] = 1/3.0;
		}
	}
}

void expand( double** particles, double* weights, int N, int n, double** particles_new, double* weights_new ){

	int count  = 0;
	int num_new = 3;
	double temp_particles[num_new][DIM];
	double temp_weights[num_new];
	for (int k=0; k<N; k++){
		double* particle = particles[k];
		double weight = weights[k];

		// assuming 3 new partciles
		generate_new(particle, temp_particles, temp_weights, num_new);

		for (int z=0; z<num_new; z++){
			for (int i=0; i<n; i++){
				particles_new[count][i] = temp_particles[z][i];
				weights_new[count] = weight * temp_weights[z];
			}
		}

	}

	return;
}

void mean( double** particles, double *weights, int N, int n, double *mu ){
	for (int i=0; i<n; i++){
		mu[i] = 0;
	}

	for (int i=0; i<N; i++){
		double* particle = particles[i];
		double weight = weights[i];
		for (int j=0; j<n; j++){
			mu[j] = mu[j] + weight * particle[j];
		}
	}
	return;
}

void covariance( double** particles, double *weights, int N, int n, double* mu, double **cov ){
        for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
                	cov[i][j] = 0;
		}
        }

        for (int k=0; k<N; k++){
                double* particle = particles[k];
                double weight = weights[k];
                for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
                        	cov[i][j] = cov[i][j] + weight * (particle[i] - mu[i] ) * (  particle[j] - mu[j]);
			}
                }
        }
        return;
}

void skewness( double** particles, double* weights, int N, int n, double* mu, double*** skew ){
	bool initialzied = false;

	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			for (int l=0;l<n;l++){
				skew[i][j][l] = 0;
			}
		}
	}

	for (int k=0; k<N; k++){
		double* particle = particles[k];
		double weight = weights[k];
		for (int i=0; i<n; i++){
			for (int j=0; j<n;j++){
				for (int l=0; l<n; l++){
					skew[i][j][l] = skew[i][j][l] + weight * ( particle[i]-mu[i] ) * ( particle[j]-mu[j] ) * ( particle[l]-mu[l] );
				}
			}
		}
	}
}

float kurtosis( double** particles, double *weights, int N, int n, double* mu, double**** kurt  ){

	bool initialzied = false;

	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			for (int l=0;l<n;l++){
					for (int m=0; m<n; m++){
						kurt[i][j][l][m] = 0;
					}
			}
		}
	}

	for (int k=0; k<N; k++){
		double* particle = particles[k];
		double weight = weights[k];
		for (int i=0; i<n; i++){
			for (int j=0; j<n;j++){
				for (int l=0; l<n; l++){
					for (int m=0; m<n; m++){
						kurt[i][j][l][m] = kurt[i][j][l][m] + weight * ( particle[i]-mu[i] ) * ( particle[j]-mu[j] ) * ( particle[l]-mu[l] ) * ( particle[m]-mu[m] );
					}
				}
			}
		}
	}


}

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

// access 3D array[n][o][p] for i,j,k: array[  i*(o*p) + j*(p) + k   ]
// access 4D array[m][n][o][p] for i,j,k,l: array[ i*(n*o*p) + j*(o*p) + k*p + l ]: easy now

double frobenius3( double***matrix ){
	double norm;
	for (int i=0; i<DIM; i++){
		for (int j=0; j<DIM; j++){
			for (int k=0; k<DIM; k++){
				norm += matrix[i][j][k] * matrix[i][j][k];		
			}
		}
	}
	norm = sqrt(norm);
}

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
	

void rank1_decomposition3(double*** tensor, double tolerance, double* vector){
	double norm = frobenius3(tensor);
	double v[DIM];
	double lambda;
	int k = 3;

	double s[100];
	int i = 0;
	while (norm > tolerance){
		HOPM3( tensor, v, lambda );
		s[i] = lambda/abs(lambda);

		double vi = pow(abs(lambda), 1.0/k);
		//T = T - s[i] * vectors[i]
		

		i = i + 1;
	}
}

void rank1_decomposition4(double**** tensor, double tolerance, double* vector){
	double norm = frobenius4(tensor);
	double v[DIM];
	double lambda;
	int k = 3;

	double s[100];
	int i = 0;
	while (norm > tolerance){
		HOPM4( tensor, v, lambda );
		s[i] = lambda/abs(lambda);

		double vi = pow(abs(lambda), 1.0/k);
		//T = T - s[i] * vectors[i]
		

		i = i + 1;
	}
}

void generate_sigma_points(double mu, double** cov, double*** skew, double**** kurtosis){

	int N = d + J + L;
	double tolerance = 0.5;

	double* v;
       	rank1_decomposition3( skew, v, tolerance );

	double* u;
	rank1_decomposition4( kurtosis, u, tolerance );

	// compute Cbar
	double Ctilde[DIM][DIM];
	double ui[L][DIM][DIM]
	for (int i=0; i<L; i++){
		double ui[DIM][DIM];
		tensor_product2( u[i], ui[i] );
	}
	for (int i=0; i<DIM; i++){
		for (int j=0; j<DIM; j++){
			Ctilde[i][j] = 0;
			for (int k=0; k<L; k++){
				Ctilde[i][j] = Ctilde[i][j] + s[k] * ui[k][i][j]
			}
		}
	}

	double lambda_max = max_eigenvalue( Ctilde );
	double lambda_min = min_eigenvalue( Cov );
	double delta = sqrt( lambda_max / lambda_min );
	

	double Chat[DIM][DIM] = C - 1.0/delta/delta * Cbar;

	double Cbar[DIM][DIM][DIM][DIM];
	double Chati[DIM][DIM][DIM][DIM][DIM];
	for (int i=0; i<DIM; i++){
		double Chatroot = symmetric_sqrt(Chat);
		transpose(Chatroot, DIM); // now each column is a row
		tensor_product4( Chatroot[i], Chati[i]  );
	}

	for (int i=0; i<DIM; i++){
		for (int j=0; j<DIM; j++){
			for (int k=0; k<DIM; k++){
				for (int l=0; l<DIM; l++){
					bar[i][j][k][k] = 0;
					for ( int m=0; m<DIM; m++ ){
						bar[i][j][k][l] = bar[i][j][k][l] + Chati[m][i][j][k][l];
					}
				}
			}
		}
	}

	double beta = 1/2.0 * sqrt( tolerance/2/frobenius2(C) );
	double gamma = 1.0;

	double Lhat = 0;
	for (int i=0; i<L; i++){
		Lhat += s[i];
	}

	double mu_tilde[DIM];
	for (int i=0; i<DIM; i++){
		mu_tilde[i] = 0;
		for ( int j=0; j<J; j++ ){
			mu_tilde[i] = mu_tilde[i] + vi[j][i];
		}
	}

	double mu_hat[DIM];
	for (int i=0; i<DIM; i++){
		mu_hat[i] = ( 1-d/beta/beta-Lhat/pow(delta,4) ) * mu[i] - 1.0/gamma/gamma*mu_tilde;
	}

	double mu3[DIM][DIM][DIM];
	tensor_product3( mu, mu3 );
	double alpha = 1/2.0 * sqrt( tolerance/2/frobenius3(mu3)  );

	// finally make new points
	
	// first 3 particles
	for (int i=0; i<DIM; i++){
		particles[0][i] = mu[i];
		particles[1][i] = mu[i] + alpha * mu_hat[i];
		particles[2][i] = mu[i] - alpha * mu_hat[i];

		for (int j=3; j<d+3; j++){
			particles[j][i] = mu[i] + beta * Chatroot[j][i];
		}
		for (int j=d+3; j<2*d+3; j++){
			particles[j][i] = mu[i] - beta * Chatroot[j-d][i];
		}

		for (int j=2*d+3; j<2*d+3+J; j++){
			particles[j][i] = mu[i] + gamma * v[j-2*d][i];
		}
		for (int j=2*d+3+J; j<2*d+3+2*J; j++){
			particles[j][i] = mu[i] - gamma * v[j-2*d-J][i];
		}

		for (int j=2*d+3+2*J; j< 2*d+3+2*J+L; j++){
			particles[j][i] = mu[i] + delta * u[i-2*d-2*J][i];
		}
		for (int j=2*d+3+2*J+L; j<N; j++){
			particles[j][i] =  = mu[i] - delta * u[i-2*d-2*J-L];
		}
	}	

	return;		
}

int main(){

	int N = 2; // number of particles
	int n = 3;
	double dt = 0.01;

	double state_init[n]={1,2,3};
	double states[N][n];
	double states_next[N][n];


	// initialize particles
	for (int i=0; i<N; i++){
		for (int j=0; j<n; j++){
			states[i][j] = i+j; //state_init[j];
		}
	}
	
	for (int i=0; i<N; i++){
		//std::cout << states[i][0] << states[i][1] << states[i][2] << std::endl;
		step_random( states[i], n, dt, states_next[i] );
		//cout << "next state " << states_next[i][0] << "\t" << states_next[i][1] << "\t" << states_next[i][2] << endl; 
	}

	return 0;
}
