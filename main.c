#include <time.h>
#include "matrix_lib.h"
#include "neural_network.h"


int main(){

    srand(time(0));

    Matrix m1  = mt_create(6, 5);
    Matrix m2  = mt_create(5, 6);
    Matrix res = mt_create(6, 6);

    mt_fill(m1, 0.88);
    mt_rand(m2, -2, 2);

    mt_print(m2);
    mt_swap_row(m2, 4, 2);
    mt_print(m2);
    Matrix m = mt_create(3, 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            mt_pos(m, i, j) = i * 10 + j;

    mt_print(m);
    mt_swap_col(m, 0, 2);
    mt_print(m);
    mt_mult(res, m1, m2);

    mt_print(res);
    printf("----------------------------\n");
    mt_activate(res, sigmoid);
    mt_print(res);

    Matrix id = mt_create(6, 6);;
    mt_id(res);
    martix_copy(id, res);
    mt_print(id);


    mt_free(id);
    mt_free(res);

    int arch[] = {3, 3, 3};
    int arch_len = sizeof(arch)/sizeof(arch[0]);
    int l_count = arch_len - 1;
    

    N_Net nn = create_n_net(l_count, arch_len, arch);
    rand_n_net(nn, 0, 10);
    n_net_print(nn);


    return 0;
}