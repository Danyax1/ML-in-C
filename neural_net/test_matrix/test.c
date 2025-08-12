#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "..\matrix_lib.h"

#define EPSILON 1e-5f

void test_create_and_fill() {
    Matrix m = mt_create(3, 3);
    mt_fill(m, 5.0f);
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            assert(fabs(mt_pos(m, i, j) - 5.0f) < EPSILON);
    mt_free(&m);
    printf("test_create_and_fill passed\n");
}

void test_copy_and_add() {
    Matrix a = mt_create(2, 2);
    Matrix b = mt_create(2, 2);
    mt_fill(a, 1.0f);
    mt_fill(b, 2.0f);
    mt_add(a, b); // a += b
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(fabs(mt_pos(a, i, j) - 3.0f) < EPSILON);
    mt_free(&a);
    mt_free(&b);
    printf("test_copy_and_add passed\n");
}

void test_identity_and_det() {
    Matrix id = mt_create(3, 3);
    mt_id(id);
    float det = mt_det(id);
    assert(fabs(det - 1.0f) < EPSILON);
    mt_free(&id);
    printf("test_identity_and_det passed\n");
}

void test_multiply() {
    Matrix a = mt_create(2, 3);
    Matrix b = mt_create(3, 2);
    Matrix result = mt_create(2, 2);

    mt_fill(a, 1.0f); // Every element = 1
    mt_fill(b, 2.0f); // Every element = 2
    mt_mult(result, a, b); // result = a * b

    for (int i = 0; i < result.rows; ++i)
        for (int j = 0; j < result.cols; ++j)
            assert(fabs(mt_pos(result, i, j) - 6.0f) < EPSILON);

    mt_free(&a);
    mt_free(&b);
    mt_free(&result);
    printf("test_multiply passed\n");
}

void test_save_and_load() {
    const char *filepath = "test_matrix.txt";
    Matrix a = mt_create(2, 2);
    mt_fill(a, 7.0f);
    mt_save(a, filepath);

    Matrix b = mt_create(2, 2);
    mt_load(b, filepath);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(fabs(mt_pos(b, i, j) - 7.0f) < EPSILON);

    mt_free(&a);
    mt_free(&b);
    printf("test_save_and_load passed\n");
}

void test_swap_rows_cols() {
    Matrix m = mt_create(2, 2);
    mt_pos(m, 0, 0) = 1;
    mt_pos(m, 0, 1) = 2;
    mt_pos(m, 1, 0) = 3;
    mt_pos(m, 1, 1) = 4;

    mt_swap_row(m, 0, 1);
    assert(mt_pos(m, 0, 0) == 3 && mt_pos(m, 1, 0) == 1);

    mt_swap_col(m, 0, 1);
    assert(mt_pos(m, 0, 0) == 4 && mt_pos(m, 0, 1) == 3);

    mt_free(&m);
    printf("test_swap_rows_cols passed\n");
}

void test_row_column_views() {
    Matrix m = mt_create(3, 3);
    mt_fill(m, 0.0f);
    mt_pos(m, 1, 1) = 42;

    Matrix row = mt_row(m, 1);
    assert(row.rows == 1 && row.cols == 3);
    assert(mt_pos(row, 0, 1) == 42);

    Matrix col = mt_column(m, 1);
    assert(col.rows == 3 && col.cols == 1);
    assert(mt_pos(col, 1, 0) == 42);

    mt_free(&m);
    printf("test_row_column_views passed\n");
}

int main() {
    test_create_and_fill();
    test_copy_and_add();
    test_identity_and_det();
    test_multiply();
    test_save_and_load();
    test_swap_rows_cols();
    test_row_column_views();

    printf("All matrix tests passed!\n");
    return 0;
}
