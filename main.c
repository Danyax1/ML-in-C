#include <time.h>
#include "matrix_lib.h"
#include "neural_network.h"


int main(){

    srand(time(0));

    Matrix m1  = matrix_create(6, 5);
    Matrix m2  = matrix_create(5, 6);
    Matrix res = matrix_create(6, 6);

    matrix_fill(m1, 0.88);
    matrix_rand(m2, -2, 2);

    matrix_mult(res, m1, m2);

    matrix_print(res);
    printf("----------------------------\n");
    matrix_activate(res, sigmoid);
    matrix_print(res);

    Matrix id = matrix_create(6, 6);;
    matrix_id(res);
    martix_copy(id, res);
    matrix_print(id);


    matrix_free(id);
    matrix_free(res);

    // int arch[] = {3, 3, 3};
    // int arch_len = sizeof(arch)/sizeof(arch[0]);
    // int l_count = arch_len - 1;
    

    // N_Net nn = create_n_net(l_count, arch_len, arch);
    // rand_n_net(nn, 0, 10);
    // n_net_print(nn);


    return 0;
}