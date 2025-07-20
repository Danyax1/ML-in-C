#include <time.h>
#include "matrix_lib.h"


int main(){

    srand(time(0));

    Matrix m0 = matrix_create(3, 2);
    matrix_init(m0, 0, 1);
    matrix_print(m0);

    Matrix m1 = matrix_create(2, 4);
    matrix_init(m1, 0, 1);
    matrix_print(m1);

    Matrix result = matrix_create(3, 4);

    matrix_mult(result, m0, m1);

    matrix_print(result);

    return 0;
}