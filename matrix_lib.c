#include "matrix_lib.h"

float random_float(float low, float high) {
    return low + ((float) rand() / (float) RAND_MAX) * (high - low);
}

Matrix matrix_create(int rows, int cols){
    assert(rows > 0);
    assert(cols > 0);
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float*) malloc(sizeof(float) * rows * cols);
    assert(m.data != NULL);
    return m;
}

void matrix_init(Matrix m, float low, float high){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            matrix_pos(m, i, j) = random_float(low, high);
        }
    }
}

void MATRIX_PRINT(Matrix m, const char *name){
    printf("%s = [\n", name);
    for (int i = 0; i < m.rows; i++){
        printf("%*s", (int)strlen(name)+4, "");
        for (int j = 0; j < m.cols; j++){
            printf(" %f ", matrix_pos(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void matrix_add(Matrix m0, Matrix m1){
    assert(m0.cols == m1.cols);
    assert(m0.rows == m1.rows);
    for(int i = 0; i < m0.rows; i++){
        for(int j = 0; j < m0.cols; j++){
            matrix_pos(m0, i, j) += matrix_pos(m1, i, j);
        }
    }
}

void matrix_mult(Matrix res, Matrix m0, Matrix m1){
    assert(m0.cols == m1.rows);
    assert(m0.rows == res.rows);
    assert(m1.cols == res.cols);
    for(int i = 0; i < res.rows; i++){
        for(int j = 0; j < res.cols; j++){
            float result = 0;
            for(int k = 0; k < m0.cols; k++){
                result += matrix_pos(m0, i, k) * matrix_pos(m1, k, j);
            }
            matrix_pos(res, i, j) = result;
        }
    }
}

void matrix_scale(Matrix m0, int scale){
    for(int i = 0; i < m0.rows; i++){
        for(int j = 0; j < m0.cols; j++){
            matrix_pos(m0, i, j) *= scale;
        }
    }
}
