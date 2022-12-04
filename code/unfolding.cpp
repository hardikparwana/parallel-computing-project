#include "include/unfolding.hpp"

void unfolding3(double*** tensor, double** matrix){
    int j_offset = 0;
    for (int k=0; k<DIM; k++){
        // fill up one by one
        for (int i=0; i<DIM; i++){
            for (int j=0; j<DIM; j++){
                matrix[i][j+j_offset] = tensor[k][i][j];
            }
        }
        j_offset += DIM;
    }
}

void unfolding4(double**** tensor, double** matrix){
    int j_offset = 0;
    for (int k=0; k<DIM; k++){
        // fill up one by one
        for (int h=0; h<DIM; h++){
            for (int i=0; i<DIM; i++){
                for (int j=0; j<DIM; j++){
                    matrix[i][j+j_offset] = tensor[k][h][i][j];
                }
            }
            j_offset += DIM;
        }
        j_offset += DIM;
    }
}