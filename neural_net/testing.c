#include <time.h>
#include "matrix_lib.h"
#include "neural_network.h"

#define N_SAMPLES 16
#define INPUT_SIZE 4
#define OUTPUT_SIZE 1

int main(){

    srand(time(0));

    Matrix m1  = mt_create(6, 5);
    Matrix m2  = mt_create(5, 6);
    Matrix res = mt_create(6, 6);
    
    mt_fill(m1, 1);
    mt_fill(m2, -2);
    
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
    
    Matrix m_det = mt_create(2, 2);
    mt_pos(m_det, 0, 0) = 0;
    mt_pos(m_det, 0, 1) = 1;
    mt_pos(m_det, 1, 0) = 4;
    mt_pos(m_det, 1, 1) = 6;
    
    mt_print(m_det);
    
    printf("Determinant of res: %f\n", mt_det(m_det));
    printf("----------------------------\n");
    mt_activate(res, sigmoid);
    mt_print(res);
    
    Matrix id = mt_create(6, 6);;
    mt_id(res);
    mt_copy(id, res);
    mt_print(id);
    
    mt_free(&m);
    mt_free(&m1);
    mt_free(&m2);
    mt_free(&m_det);
    mt_free(&id);
    mt_free(&res);

    float dataset[N_SAMPLES][INPUT_SIZE + OUTPUT_SIZE] = {
      // A1 A0 B1 B0|Sum
        {0, 0, 0, 0, 0},
        {0, 0, 0, 1, 1},
        {0, 0, 1, 0, 2},
        {0, 0, 1, 1, 3},
        {0, 1, 0, 0, 1},
        {0, 1, 0, 1, 2},
        {0, 1, 1, 0, 3},
        {0, 1, 1, 1, 4},
        {1, 0, 0, 0, 2},
        {1, 0, 0, 1, 3},
        {1, 0, 1, 0, 4},
        {1, 0, 1, 1, 5},
        {1, 1, 0, 0, 3},
        {1, 1, 0, 1, 4},
        {1, 1, 1, 0, 5},
        {1, 1, 1, 1, 6},
    };
    

    Matrix inputs = mt_create(N_SAMPLES, INPUT_SIZE);
    Matrix outputs = mt_create(N_SAMPLES, OUTPUT_SIZE);


    split_dataset(&dataset[0][0], INPUT_SIZE, OUTPUT_SIZE, N_SAMPLES, inputs, outputs);

    for (int i = 0; i < outputs.rows; i++) {
        mt_pos(outputs, i, 0) /= 6.0f;
    }
    //  mt_print(inputs);
    //  mt_print(outputs);

    int arch[] = {INPUT_SIZE, 2*INPUT_SIZE, INPUT_SIZE,  1};
    int arch_len = sizeof(arch)/sizeof(arch[0]);
    int l_count = arch_len - 1;
    

    N_Net nn = create_n_net(l_count, arch_len, arch);
    rand_n_net(nn, -3, 3);
    N_Net grad = create_n_net(l_count, arch_len, arch);
    const float rate = -(1);

    for(int iter = 0; iter < 50; iter++){

        float loss = 0;

        for(int i = 0; i < N_SAMPLES; i++){
            set_n_net_input(nn, mt_row(inputs, i));
            forward_n_net(nn);
            float mse = loss_n_net(nn, mt_row(outputs, i));
            loss += mse;
        }
        printf("%-6d loss: %f\n", iter, loss/N_SAMPLES);
        backprop_n_net(nn, grad, inputs, outputs);
        learn_n_net(nn, grad, rate);
    }
    printf("------------------------------\n");
    
    for (int i = 0; i < inputs.rows; i++) {
        set_n_net_input(nn, mt_row(inputs, i));
        forward_n_net(nn);

        float out = mt_pos(output_n_net(nn), 0, 0);
        printf("%d%d + %d%d = %f\n", (int)dataset[i][0], (int)dataset[i][1], (int)dataset[i][2],(int)dataset[i][3], out*6.0f);
    }
    free_n_net(&nn);
    free_n_net(&grad);
    mt_free(&inputs);
    mt_free(&outputs);

    return 0;
}