#include <time.h>
#include "matrix_lib.h"
#include "neural_network.h"


int main(){

    srand(time(0));

    // Matrix m1  = mt_create(6, 5);
    // Matrix m2  = mt_create(5, 6);
    // Matrix res = mt_create(6, 6);

    // mt_fill(m1, 1);
    // mt_fill(m2, -2);

    // // mt_print(m2);
    // // mt_swap_row(m2, 4, 2);
    // // mt_print(m2);
    // // Matrix m = mt_create(3, 3);
    // // for (int i = 0; i < 3; i++)
    // //     for (int j = 0; j < 3; j++)
    // //         mt_pos(m, i, j) = i * 10 + j;

    // // mt_print(m);
    // // mt_swap_col(m, 0, 2);
    // // mt_print(m);
    // mt_mult(res, m1, m2);

    // mt_print(res);

    // Matrix m_det = mt_create(2, 2);
    // mt_pos(m_det, 0, 0) = 0;
    // mt_pos(m_det, 0, 1) = 1;
    // mt_pos(m_det, 1, 0) = 4;
    // mt_pos(m_det, 1, 1) = 6;

    // mt_print(m_det);

    // printf("Determinant of res: %f\n", mt_det(m_det));
    // printf("----------------------------\n");
    // mt_activate(res, sigmoid);
    // mt_print(res);

    // Matrix id = mt_create(6, 6);;
    // mt_id(res);
    // mt_copy(id, res);
    // mt_print(id);


    // mt_free(&id);
    // mt_free(&res);
    float model[][2]={
        {1, 1},
        {0, 0},
        {0, 1},
        {1, 0},
    };
    float res[]={
        1, 0, 0, 1
    };
    Matrix expect = mt_create(1, 4);
    mt_set(&expect, (float*)res, 1, 4);
    mt_print(expect);
    Matrix samples = mt_create(4, 2);
    mt_set(&samples, (float*)model, 4, 2);
    mt_print(samples);

    


    int arch[] = {2, 2, 4};
    int arch_len = sizeof(arch)/sizeof(arch[0]);
    int l_count = arch_len - 1;
    

    N_Net nn = create_n_net(l_count, arch_len, arch);
    rand_n_net(nn, -3, 3);
    set_n_net_input(nn, mt_row(samples, 1));
    forward_n_net(nn);
    print_n_net(nn);

    float mse = loss_n_net(nn, expect);
    printf("mse = %f\n", mse);


    return 0;
}