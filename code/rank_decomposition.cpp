#include "include/params.hpp"
#include "include/rank_decomposition.hpp"
#include "include/frobenius.hpp"
#include "include/hopm.hpp"

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