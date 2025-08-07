#ifndef MATRIX_LIB_H
#define MATRIX_LIB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

float random_int(const int low, const int high);
float random_float(const float low, const float high);

typedef struct {
    int rows;
    int cols;
    int stride;
    float *data;
} Matrix;

#define mt_print(m) MATRIX_PRINT((m), (#m))
#define mt_pos(m, i, j) (m).data[(i)*((m).stride)+(j)]

Matrix mt_create(const int rows, const int cols);
void mt_free(Matrix *m);

void mt_rand(const Matrix m, const float low, const float high);
void mt_fill(const Matrix m, const float fill);
void mt_set(Matrix *m, float* matr, const int rows, const int cols);
void mt_id(const Matrix m);

Matrix mt_row(const Matrix m, const int row);
Matrix mt_column(const Matrix m, const int col);
void mt_swap_row(const Matrix m, const int r1, const int r2);
void mt_swap_col(const Matrix m, const int c1, const int c2);

void mt_copy(const Matrix dest, const Matrix src);
void MATRIX_PRINT(const Matrix m, const char *name);

void mt_add(const Matrix m0, const Matrix m1);
void mt_sub(const Matrix m0, const Matrix m1);
void mt_mult(const Matrix res, const Matrix m0, const Matrix m1);
void mt_scale(const Matrix m0, const float scale);
float mt_det(const Matrix m);

void mt_rearrange(Matrix *m, const int rows, const int cols);
void mt_randomize_rows(const Matrix m, const Matrix m1);
void split_dataset(const float* data, const int input_size, const int output_size, const int n_samples, const Matrix input, const Matrix output);

void mt_save(const Matrix m, const char *filepath);
void mt_load(const Matrix m, const char *filepath);
#endif // MATRIX_LIB_H
