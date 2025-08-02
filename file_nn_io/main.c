#include <time.h>
#include "..\neural_net\matrix_lib.h"
#include "..\neural_net\neural_network.h"

#define N_SAMPLES 16
#define INPUT_SIZE 4
#define OUTPUT_SIZE 1

int main(){

    srand(time(0));

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
    float rate = (1);

    train_n_net(nn, grad, inputs, outputs, N_SAMPLES, rate, 10000, true);

    printf("------------------------------\n");
    
    for (int i = 0; i < inputs.rows; i++) {
        set_n_net_input(nn, mt_row(inputs, i));
        forward_n_net(nn);

        float out = mt_pos(output_n_net(nn), 0, 0);
        printf("%d%d + %d%d = %f\n", (int)dataset[i][0], (int)dataset[i][1], (int)dataset[i][2],(int)dataset[i][3], out*6.0f);
    }
    printf("------------------------------\n");
    return 0;
}