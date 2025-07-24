#ifndef MATRIX_LIB_H
#define MATRIX_LIB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

float random_float(float low, float high);

typedef struct {
    int rows;
    int cols;
    int stride;
    float *data;
} Matrix;

#define matrix_print(m) MATRIX_PRINT((m), (#m))
#define matrix_pos(m, i, j) (m).data[(i)*((m).stride)+(j)]

Matrix matrix_create(int rows, int cols);
void matrix_free(Matrix m);

void matrix_rand(Matrix m, float low, float high);
void matrix_fill(Matrix m, float fill);
void matrix_id(Matrix m);

Matrix matrix_row(Matrix m, int row);
Matrix matrix_column(Matrix m, int col);
void matrix_swap_row(Matrix m, int r1, int r2);
void matrix_swap_col(Matrix m, int c1, int c2);

void martix_copy(Matrix dest, Matrix src);
void MATRIX_PRINT(Matrix m, const char *name);

void matrix_add(Matrix m0, Matrix m1);
void matrix_mult(Matrix res, Matrix m0, Matrix m1);
void matrix_scale(Matrix m0, int scale);

void matrix_rearrange(Matrix m, int rows, int cols);

#endif // MATRIX_LIB_H
