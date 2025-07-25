#include "matrix_lib.h"

float random_float(float low, float high) {
    return low + ((float) rand() / (float) RAND_MAX) * (high - low);
}

Matrix mt_create(int rows, int cols){
    assert(rows > 0);
    assert(cols > 0);
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = (float*) malloc(sizeof(float) * rows * cols);
    assert(m.data != NULL);
    return m;
}

void mt_free(Matrix *m) {
    if (m->data) {
        free(m->data);
        m->data = NULL;  // prevent double free
    }
}


void mt_rand(Matrix m, float low, float high){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            mt_pos(m, i, j) = random_float(low, high);
        }
    }
}
void mt_fill(Matrix m, float fill){
        for(int i = 0; i < m.rows; i++){
            for(int j = 0; j < m.cols; j++){
                mt_pos(m, i, j) = fill;
            }
    }
};
void mt_set(Matrix *m, float* matr, int rows, int cols) {
    assert(m->rows == rows);
    assert(m->cols == cols);
    m->data = matr;
}

void mt_id(Matrix m){
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

Matrix mt_row(Matrix m, int row){
    assert(row < m.rows);
    assert(row >= 0);
    Matrix res;
    res.rows = 1;
    res.cols = m.cols;
    res.data = &(m.data[(row) * m.cols]);
    return res;
};
Matrix mt_column(Matrix m, int col){
    assert(col < m.cols);
    assert(col >= 0);
    Matrix res;
    res.rows = m.rows;
    res.cols = 1;
    res.stride = m.stride;
    res.data = &(m.data[(col)]);
    return res;

};
void mt_swap_row(Matrix m, int r1, int r2) {
    assert(r1 >= 0 && r1 < m.rows);
    assert(r2 >= 0 && r2 < m.rows);
    if (r1 == r2) return;

    for (int i = 0; i < m.cols; i++) {
        float tmp = mt_pos(m, r1, i);
        mt_pos(m, r1, i) = mt_pos(m, r2, i);
        mt_pos(m, r2, i) = tmp;
    }
}


void mt_swap_col(Matrix m, int c1, int c2){
    assert(c1 >= 0 && c1 < m.cols);
    assert(c2 >= 0 && c2 < m.cols);
    if (c1 == c2) return;

    for (int i = 0; i < m.rows; i++) {
        float tmp = mt_pos(m, i, c1);
        mt_pos(m, i, c1) = mt_pos(m, i, c2);
        mt_pos(m, i, c2) = tmp;
    }
};

void mt_copy(Matrix dest, Matrix src){
    assert(dest.rows == src.rows);
    assert(dest.cols == src.cols);
    for(int i = 0; i < dest.rows; i++){
        for(int j = 0; j < dest.cols; j++){
            mt_pos(dest, i, j) = mt_pos(src, i, j);
        }
    }
};

void MATRIX_PRINT(Matrix m, const char *name){
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

void mt_add(Matrix m0, Matrix m1){
    assert(m0.cols == m1.cols);
    assert(m0.rows == m1.rows);
    for(int i = 0; i < m0.rows; i++){
        for(int j = 0; j < m0.cols; j++){
            mt_pos(m0, i, j) += mt_pos(m1, i, j);
        }
    }
}

void mt_mult(Matrix res, Matrix m0, Matrix m1){
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

void mt_scale(Matrix m0, int scale){
    for(int i = 0; i < m0.rows; i++){
        for(int j = 0; j < m0.cols; j++){
            mt_pos(m0, i, j) *= scale;
        }
    }
}

float mt_det(Matrix m){
    assert(m.cols == m.rows);
    int n = m.cols;
    int sign = 1;
    Matrix U = mt_create(n, n);
    mt_copy(U, m);

    for(int i = 0; i < n; i++){
        
        int pivot  = i;
        while(mt_pos(U, i, i) == 0 && pivot < n){
            if (mt_pos(U, pivot, i)){
                // printf("SWAPPING i: %d, pivot: %d\n", i, pivot);
                mt_swap_row(U, i, pivot);
                break;
            }
            pivot++;
        }
        if (pivot != i){
            sign *= -1;
        }
        if (!mt_pos(U, i, i)){
            // printf("1 col is all 0");
            return 0;
        }
        // mt_print(U);
        for(int j = i+1; j < n; j++){
            float coef = mt_pos(U, j, i)/mt_pos(U, i, i);
            // printf("i: %d, j: %d, coef: %f\n", i, j, coef);
            for(int k = i; k < n; k++){
                // printf("U[j][k] = %d", );
                mt_pos(U ,j, k) -= coef * mt_pos(U, i, k);
            }
        }
    }
    float det = 1.0f;
    for(int i = 0; i < n; i++){
        det *= mt_pos(U, i ,i);
    }
    return det * sign;
};



void mt_rearrange(Matrix *m, int rows, int cols) {
    assert(m->cols * m->rows == rows * cols);
    assert(rows > 0);

    m->rows = rows;
    m->cols = cols;
}
