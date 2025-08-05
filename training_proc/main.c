#include <time.h>
#include "..\neural_net\matrix_lib.h"
#include "..\neural_net\neural_network.h"

#define GRID 14

int main(void){
    srand(time(0));

    Matrix hand_digit = mt_create(GRID, GRID);
    mt_load(hand_digit, "..\\python_interface\\digit_config.txt");
    mt_rearrange(&hand_digit, 1, GRID*GRID);

    Matrix goal_digit = mt_create(1, 1);
    mt_pos(goal_digit, 0, 0) = 6.0f/9.0f;

    // mt_print(hand_digit);
    // mt_print(goal_digit);


    int arch[] = {GRID*GRID, GRID, GRID, 1};
    int arch_len = sizeof(arch)/sizeof(arch[0]);
    int l_count = arch_len - 1;
    

    N_Net nn = create_n_net(l_count, arch_len, arch);
    rand_n_net(nn, -2.0f, 2.0f);
    N_Net grad = create_n_net(l_count, arch_len, arch);
    float rate = (1);

    
    for(int iter = 0; iter < 10; iter++){
        float loss = 0;
        set_n_net_input(nn, hand_digit);
        forward_n_net(nn);
        float mse = loss_n_net(nn, goal_digit);
        loss += mse;
        
        printf("%-6d loss: %f\n", iter, loss/1);
        backprop_n_net(nn, grad, hand_digit, goal_digit);
        learn_n_net(nn, grad, rate);
    }


    // print_n_net(nn);
    set_n_net_input(nn, hand_digit);
    forward_n_net(nn);
    float out = mt_pos(output_n_net(nn), 0, 0);

    printf("RESULT: %f", out);



    return 0;
}