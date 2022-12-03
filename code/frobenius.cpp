#include "include/frobenius.hpp"

double frobenius2( double** matrix ){
	double norm = 0.0;
	for (int i=0; i<DIM; i++){
		for (int j=0; j<DIM; j++){
				norm += matrix[i][j] * matrix[i][j];		
		}
	}
	norm = sqrt(norm);
    return norm;
}

double frobenius3( double*** matrix ){
	double norm = 0.0;
	for (int i=0; i<DIM; i++){
		for (int j=0; j<DIM; j++){
			for (int k=0; k<DIM; k++){
				norm += matrix[i][j][k] * matrix[i][j][k];		
			}
		}
	}
	norm = sqrt(norm);
    return norm;
}

double frobenius4( double**** matrix ){
	double norm; 0.0;
	for (int i=0; i<DIM; i++){
		for (int j=0; j<DIM; j++){
			for (int k=0; k<DIM; k++){
                for (int l=0; l<DIM;  l++){
                    norm += matrix[i][j][k][l] * matrix[i][j][k][l];	
                }					
			}
		}
	}
	norm = sqrt(norm);
    return norm;
}