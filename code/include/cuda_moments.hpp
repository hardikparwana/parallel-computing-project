#include "params.hpp"

void mean( double** particles, double *weights, int N, double *mu );
void covariance( double** particles, double *weights, int N, double* mu, double **cov );
void skewness( double** particles, double* weights, int N, double* mu, double*** skew );
void kurtosis( double** particles, double *weights, int N, double* mu, double**** kurt  );
