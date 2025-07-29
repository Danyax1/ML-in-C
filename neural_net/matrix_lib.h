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

#define mt_print(m) MATRIX_PRINT((m), (#m))
#define mt_pos(m, i, j) (m).data[(i)*((m).stride)+(j)]

Matrix mt_create(int rows, int cols);
void mt_free(Matrix *m);

void mt_rand(Matrix m, float low, float high);
void mt_fill(Matrix m, float fill);
void mt_set(Matrix *m, float* matr, int rows, int cols);
void mt_id(Matrix m);

Matrix mt_row(Matrix m, int row);
Matrix mt_column(Matrix m, int col);
void mt_swap_row(Matrix m, int r1, int r2);
void mt_swap_col(Matrix m, int c1, int c2);

void mt_copy(Matrix dest, Matrix src);
void MATRIX_PRINT(Matrix m, const char *name);

void mt_add(Matrix m0, Matrix m1);
void mt_sub(Matrix m0, Matrix m1);
void mt_mult(Matrix res, Matrix m0, Matrix m1);
void mt_scale(Matrix m0, float scale);
float mt_det(Matrix m);

void mt_rearrange(Matrix *m, int rows, int cols);
void split_dataset(float* data, int input_size, int output_size, int n_samples, Matrix input, Matrix output);
#endif // MATRIX_LIB_H
