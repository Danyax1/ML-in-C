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
    int stride;
    float *data;
} Matrix;

#define matrix_print(m) MATRIX_PRINT((m), (#m))
#define matrix_pos(m, i, j) (m).data[(i)*((m).stride)+(j)]


float random_float(float low, float high);
Matrix matrix_create(int rows, int cols);
void matrix_rand(Matrix m, float low, float high);
void matrix_fill(Matrix m, float fill);
Matrix matrix_row(Matrix m, int row);
Matrix matrix_column(Matrix m, int col);
void MATRIX_PRINT(Matrix m, const char *name);
void matrix_add(Matrix m0, Matrix m1);
void matrix_mult(Matrix res, Matrix m0, Matrix m1);
void matrix_scale(Matrix m0, int scale);
void matrix_rearrange(Matrix m, int rows, int cols);

#endif // MATRIX_LIB_H
