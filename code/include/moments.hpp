#include "params.hpp"

void mean( double** particles, double *weights, int N, int n, double *mu );
void covariance( double** particles, double *weights, int N, int n, double* mu, double **cov );
void skewness( double** particles, double* weights, int N, int n, double* mu, double*** skew );
void kurtosis( double** particles, double *weights, int N, int n, double* mu, double**** kurt  );
