#ifndef MATRIX_LIB_H
#define MATRIX_LIB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

typedef struct {
    int rows;
    int cols;
    float *data;
} Matrix;

#define matrix_print(m) MATRIX_PRINT((m), (#m))
#define matrix_pos(m, i, j) (m).data[(i)*((m).cols)+(j)]


float random_float(float low, float high);
Matrix matrix_create(int rows, int cols);
void matrix_init(Matrix m, float low, float high);
void MATRIX_PRINT(Matrix m, const char *name);
void matrix_add(Matrix m0, Matrix m1);
void matrix_mult(Matrix res, Matrix m0, Matrix m1);
void matrix_scale(Matrix m0, int scale);

#endif // MATRIX_LIB_H
