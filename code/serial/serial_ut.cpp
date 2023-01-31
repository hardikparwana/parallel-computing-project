/*
	Compile using: nvcc frobenius.cpp moments.cpp tensor_product.cpp unfolding.cpp svd.cpp rank_decomposition.cpp hopm.cpp serial_ut.cpp
*/
// free up memory everywhere

#include "include/params.hpp"
// #include <cuda.h>
// #include <curand_kernel.h>

#include "include/frobenius.hpp"
#include "include/tensor_product.hpp"
#include "include/moments.hpp"
#include "include/unfolding.hpp"
#include "include/rank_decomposition.hpp"

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

// access 3D array[n][o][p] for i,j,k: array[  i*(o*p) + j*(p) + k   ]
// access 4D array[m][n][o][p] for i,j,k,l: array[ i*(n*o*p) + j*(o*p) + k*p + l ]: easy now
/*
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

*/
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
