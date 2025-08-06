#include <time.h>
#include "..\neural_net\matrix_lib.h"
#include "..\neural_net\neural_network.h"

#define GRID 14
#define SAMPLES 10000
#define OUTPUT_SIZE 1


void load_digit_dataset(Matrix data, const char* filepath) {
    FILE* file = fopen(filepath, "r");
    assert(file);

    int rows = data.rows;
    int cols = data.cols;

    for (int i = 0; i < rows; i++) {
        int r, c, s;
        fscanf(file, "%d %d %d\n", &r, &c, &s);
        assert(r * c == cols);

        for (int j = 0; j < cols; j++) {
            fscanf(file, "%f", &mt_pos(data, i, j));
        }
    }

    fclose(file);
}

void load_digit_labels(Matrix labels, const char* filepath) {
    FILE* file = fopen(filepath, "r");
    assert(file);

    for (int i = 0; i < labels.rows; i++) {
        int digit;
        fscanf(file, "%d", &digit);
        mt_pos(labels, i, 0) = digit / 9.0f;  // normalize
    }

    fclose(file);
}

int main(void){
    srand(time(0));

    Matrix X = mt_create(SAMPLES, GRID*GRID);
    Matrix Y = mt_create(SAMPLES, OUTPUT_SIZE);


    int arch[] = {GRID*GRID, GRID, GRID, 1};
    int arch_len = sizeof(arch)/sizeof(arch[0]);
    int l_count = arch_len - 1;
    

    N_Net nn = create_n_net(l_count, arch_len, arch);
    rand_n_net(nn, -2.0f, 2.0f);
    N_Net grad = create_n_net(l_count, arch_len, arch);
    float rate = (1);

    
    for (int epoch = 0; epoch < 50; epoch++) {
        float loss = 0;
        for (int i = 0; i < SAMPLES; i++) {
            set_n_net_input(nn, mt_row(X, i));
            forward_n_net(nn);
            loss += loss_n_net(nn, mt_row(Y, i));
        }
        printf("Epoch %3d Loss: %8f\n", epoch, loss / SAMPLES);
        backprop_n_net(nn, grad, X, Y);
        learn_n_net(nn, grad, rate);
    }

    save_n_net(nn, "nn_config.txt");

    return 0;
}