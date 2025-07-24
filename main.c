#include <time.h>
#include "matrix_lib.h"
#include "neural_network.h"


int main(){

    srand(time(0));

    // Matrix m1  = matrix_create(6, 5);
    // Matrix m2  = matrix_create(5, 6);
    // Matrix res = matrix_create(6, 6);

    // matrix_fill(m1, 0.88);
    // matrix_rand(m2, 0, 2);

    // matrix_print(m2);
    // matrix_swap_row(m2, 4, 2);
    // matrix_print(m2);
    Matrix m = matrix_create(3, 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            matrix_pos(m, i, j) = i * 10 + j;

    matrix_print(m);
    matrix_swap_col(m, 0, 2);
    matrix_print(m);
        // matrix_mult(res, m1, m2);

    // matrix_print(res);
    // printf("----------------------------\n");
    // matrix_activate(res, sigmoid);
    // matrix_print(res);

    // Matrix id = matrix_create(6, 6);;
    // matrix_id(res);
    // martix_copy(id, res);
    // matrix_print(id);


    // matrix_free(id);
    // matrix_free(res);

    // int arch[] = {3, 3, 3};
    // int arch_len = sizeof(arch)/sizeof(arch[0]);
    // int l_count = arch_len - 1;
    

    // N_Net nn = create_n_net(l_count, arch_len, arch);
    // rand_n_net(nn, 0, 10);
    // n_net_print(nn);


    return 0;
}