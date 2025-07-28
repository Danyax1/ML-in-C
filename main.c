#include <time.h>
#include "matrix_lib.h"
#include "neural_network.h"

#define N_SAMPLES 4
#define INPUT_SIZE 2
#define OUTPUT_SIZE 1

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

    float dataset[N_SAMPLES][INPUT_SIZE + OUTPUT_SIZE] = {
        {0, 0, 1},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0},
    };

    Matrix inputs = mt_create(N_SAMPLES, INPUT_SIZE);
    Matrix outputs = mt_create(N_SAMPLES, OUTPUT_SIZE);

    split_dataset(&dataset[0][0], INPUT_SIZE, OUTPUT_SIZE, N_SAMPLES, inputs, outputs);
    //  mt_print(inputs);
    //  mt_print(outputs);
    


    int arch[] = {2, 2, 1};
    int arch_len = sizeof(arch)/sizeof(arch[0]);
    int l_count = arch_len - 1;
    

    N_Net nn = create_n_net(l_count, arch_len, arch);
    rand_n_net(nn, -3, 3);
    N_Net grad = create_n_net(l_count, arch_len, arch);

    for(int iter = 0; iter < 1000; iter++){

        float loss = 0;

        for(int i = 0; i < N_SAMPLES; i++){
            set_n_net_input(nn, mt_row(inputs, i));
            forward_n_net(nn);
            // print_n_net(nn);

            float rate = 1;
            

            float mse = loss_n_net(nn, mt_row(outputs, 1));
            loss += mse;
            backprop_n_net(nn, grad, inputs, outputs);
            learn_n_net(nn, grad, rate);
            // print_n_net(grad);
            forward_n_net(nn);
            // print_n_net(nn);
        }

        printf("%-6d loss: %f\n", iter, loss/N_SAMPLES);
    }
    printf("------------------------------\n");
    for(int i = 0; i < inputs.rows; i++){
        printf("%d @ %d = %f\n", (int)dataset[i][0],(int)dataset[i][1], mt_pos(output_n_net(nn), 0, i));
    }

    return 0;
}