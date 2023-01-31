#include "include/params.hpp"
#include "include/rank_decomposition.hpp"
#include "include/frobenius.hpp"
#include "include/hopm.hpp"
#include "include/tensor_product.hpp"

void rank1_decomposition3(double*** tensor, double tolerance, double** vtilde, double* s, double& iter_final){ // vtilde[100][DIM], s[100], iter

	double norm = frobenius3(tensor);
	double* v; v = new double[DIM];
	double lambda;

	int iter = 0;

	double*** product_temp;
	product_temp = new double** [DIM];
	for (int i=0; i<DIM; i++){
		product_temp[i] = new double* [DIM];
		for (int j=0; j<DIM; j++){
			product_temp[i][j] = new double[DIM];
		}
	}

	while ( ( (norm > tolerance) || (iter==0) ) && (iter<100) ){
		HOPM3( tensor, v, lambda );
		s[iter] = lambda/abs(lambda);

		double lambda_1_k = pow(abs(lambda), 1.0/3);
		for (int i=0; i<DIM; i++){
			vtilde[iter][i] = lambda_1_k * v[i];
		}	
		tensor_product3( vtilde[iter], product_temp );

		// update tensor
		norm = 0;
		for (int i=0; i<DIM; i++){
			for (int j=0; j<DIM; j++){
				for (int k=0; k<DIM; k++){
					tensor[i][j][k] = tensor[i][j][k] - s[iter] * product_temp[i][j][k];
					norm = norm + tensor[i][j][k] * tensor[i][j][k];
				}
			}
		}
		norm = sqrt(norm);

		iter = iter + 1;
	}

	iter_final = iter;

}

void rank1_decomposition4(double**** tensor, double tolerance, double** vtilde, double* s, double& iter_final){

	double norm = frobenius4(tensor);
	double* v; v = new double[DIM];
	double lambda;

	int iter = 0;

	double**** product_temp;
	product_temp = new double*** [DIM];
	for (int i=0; i<DIM; i++){
		product_temp[i] = new double** [DIM];
		for (int j=0; j<DIM; j++){
			product_temp[i][j] = new double* [DIM];
			for (int k=0; k<DIM; k++){
				product_temp[i][j][k] = new double[DIM];
			}
		}
	}

	while ( ( (norm > tolerance) || (iter==0) ) && (iter<100) ){
		HOPM4( tensor, v, lambda );
		s[iter] = lambda/abs(lambda);

		double lambda_1_k = pow(abs(lambda), 1.0/3);
		for (int i=0; i<DIM; i++){
			vtilde[iter][i] = lambda_1_k * v[i];
		}	
		tensor_product4( vtilde[iter], product_temp );

		// update tensor
		norm = 0;
		for (int i=0; i<DIM; i++){
			for (int j=0; j<DIM; j++){
				for (int k=0; k<DIM; k++){
					for (int l=0; l<DIM; l++){
						tensor[i][j][k][l] = tensor[i][j][k][l] - s[iter] * product_temp[i][j][k][l];
						norm = norm + tensor[i][j][k][l] * tensor[i][j][k][l];
					}
				}
			}
		}
		norm = sqrt(norm);

		iter = iter + 1;
	}

	iter_final = iter;

}