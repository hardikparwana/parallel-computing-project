#include <iostream>
#include <random>

using namespace std;

typedef struct prediction
{
    double pred[2];
}pred;

pred gaussian_process(double* state){

	// design control input?
	pred out;
	std::cout << "GP received state " << state[0] << state[1] << state[2] << endl;
	out.pred[0] = 6;
	out.pred[1] = 4;
	return out;
	
}

void step_random(double *current_state, int dim, double dt, double *next_state){

    std::cout << "received state " << current_state[0] << current_state[1] << current_state[2] << std::endl;
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
		std::cout << states[i][0] << states[i][1] << states[i][2] << std::endl;
		step_random( states[i], n, dt, states_next[i] );
	}

	return 0;
}
