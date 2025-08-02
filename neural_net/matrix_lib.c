#include "matrix_lib.h"

float random_float(const float low, const float high) {
    return low + ((float) rand() / (float) RAND_MAX) * (high - low);
}

Matrix mt_create(const int rows, const int cols){
    assert(rows > 0);
    assert(cols > 0);
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = (float*) calloc(rows * cols, sizeof(float) );
    assert(m.data != NULL);
    return m;
}

void mt_free(Matrix *m) {
    if (m->data) {
        free(m->data);
        m->data = NULL;  // prevent double free
    }
}


void mt_rand(const Matrix m, const float low, const float high){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            mt_pos(m, i, j) = random_float(low, high);
        }
    }
}
void mt_fill(const Matrix m, const float fill){
        for(int i = 0; i < m.rows; i++){
            for(int j = 0; j < m.cols; j++){
                mt_pos(m, i, j) = fill;
            }
    }
};
void mt_set(Matrix *m, float* matr, const int rows, const int cols) {
    assert(m->rows == rows);
    assert(m->cols == cols);
    m->data = matr;
}

void mt_id(const Matrix m){
    assert(m.cols == m.rows);
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            if (i == j){
                mt_pos(m, i, j) = 1;
            } else {
                mt_pos(m, i, j) = 0;
            }
        }
    }
};

Matrix mt_row(const Matrix m, const int row){
    assert(row < m.rows);
    assert(row >= 0);
    Matrix res;
    res.rows = 1;
    res.cols = m.cols;
    res.stride = m.stride;
    res.data = &(m.data[(row) * m.cols]);
    return res;
};
Matrix mt_column(const Matrix m, const int col){
    assert(col < m.cols);
    assert(col >= 0);
    Matrix res;
    res.rows = m.rows;
    res.cols = 1;
    res.stride = m.stride;
    res.data = &(m.data[(col)]);
    return res;

};
void mt_swap_row(const Matrix m, const int r1, const int r2) {
    assert(r1 >= 0 && r1 < m.rows);
    assert(r2 >= 0 && r2 < m.rows);
    if (r1 == r2) return;

    for (int i = 0; i < m.cols; i++) {
        float tmp = mt_pos(m, r1, i);
        mt_pos(m, r1, i) = mt_pos(m, r2, i);
        mt_pos(m, r2, i) = tmp;
    }
}


void mt_swap_col(const Matrix m, const int c1, const int c2){
    assert(c1 >= 0 && c1 < m.cols);
    assert(c2 >= 0 && c2 < m.cols);
    if (c1 == c2) return;

    for (int i = 0; i < m.rows; i++) {
        const float tmp = mt_pos(m, i, c1);
        mt_pos(m, i, c1) = mt_pos(m, i, c2);
        mt_pos(m, i, c2) = tmp;
    }
};

void mt_copy(const Matrix dest, const Matrix src){
    assert(dest.rows == src.rows);
    assert(dest.cols == src.cols);
    for(int i = 0; i < dest.rows; i++){
        for(int j = 0; j < dest.cols; j++){
            mt_pos(dest, i, j) = mt_pos(src, i, j);
        }
    }
};

void MATRIX_PRINT(const Matrix m, const char *name){
    printf("%s = [\n", name);
    for (int i = 0; i < m.rows; i++){
        printf("%*s", (int)strlen(name)+4, "");
        for (int j = 0; j < m.cols; j++){
            printf(" %9f ", mt_pos(m, i, j));
        }
        printf("\n");
    }
    printf("%*s", (int)strlen(name)+3, "");
    printf("]\n");
}

void mt_add(const Matrix m0, const Matrix m1){
    assert(m0.cols == m1.cols);
    assert(m0.rows == m1.rows);
    for(int i = 0; i < m0.rows; i++){
        for(int j = 0; j < m0.cols; j++){
            mt_pos(m0, i, j) += mt_pos(m1, i, j);
        }
    }
}

void mt_sub(const Matrix m0, const Matrix m1){
    assert(m0.cols == m1.cols);
    assert(m0.rows == m1.rows);
    for(int i = 0; i < m0.rows; i++){
        for(int j = 0; j < m0.cols; j++){
            mt_pos(m0, i, j) = mt_pos(m1, i, j);
        }
    }
}

void mt_mult(const Matrix res, const Matrix m0, const Matrix m1){
    assert(m0.cols == m1.rows);
    assert(m0.rows == res.rows);
    assert(m1.cols == res.cols);
    for(int i = 0; i < res.rows; i++){
        for(int j = 0; j < res.cols; j++){
            float result = 0;
            for(int k = 0; k < m0.cols; k++){
                result += mt_pos(m0, i, k) * mt_pos(m1, k, j);
            }
            mt_pos(res, i, j) = result;
        }
    }
}

void mt_scale(const Matrix m0, const float scale){
    for(int i = 0; i < m0.rows; i++){
        for(int j = 0; j < m0.cols; j++){
            mt_pos(m0, i, j) *= scale;
        }
    }
}

float mt_det(const Matrix m){
    assert(m.cols == m.rows);
    const int n = m.cols;
    int sign = 1;
    Matrix U = mt_create(n, n);
    mt_copy(U, m);

    for(int i = 0; i < n; i++){
        
        int pivot  = i;
        while(mt_pos(U, i, i) == 0 && pivot < n){
            if (mt_pos(U, pivot, i)){
                mt_swap_row(U, i, pivot);
                break;
            }
            pivot++;
        }
        if (pivot != i){
            sign *= -1;
        }
        if (!mt_pos(U, i, i)){
            mt_free(&U);
            return 0;
        }
        for(int j = i+1; j < n; j++){
            const float coef = mt_pos(U, j, i)/mt_pos(U, i, i);
            for(int k = i; k < n; k++){
                mt_pos(U ,j, k) -= coef * mt_pos(U, i, k);
            }
        }
    }
    float det = 1.0f;
    for(int i = 0; i < n; i++){
        det *= mt_pos(U, i ,i);
    }
    mt_free(&U);
    return det * (float)sign;
};

void mt_rearrange(Matrix *m, const int rows, const int cols) {
    assert(m->cols * m->rows == rows * cols);
    assert(rows > 0);

    m->rows = rows;
    m->cols = cols;
}

void split_dataset(const float* data, const int input_size, const int output_size, const int n_samples, const Matrix input, const Matrix output) {
    assert(input.rows == output.rows);

    const int total = input_size + output_size;
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < input_size; j++) {
            mt_pos(input, i, j) = data[i * total + j];
        }
        for (int j = 0; j < output_size; j++) {
            mt_pos(output, i, j) = data[i * total + input_size + j];
        }
    }
}
