#include <time.h>
#include "..\neural_net\matrix_lib.h"
#include "..\neural_net\neural_network.h"

#define GRID 14
#define SAMPLES 49000
#define OUTPUT_SIZE 1


void load_digit_dataset(Matrix data, Matrix check, const char* filepath) {
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
    for (int i = 0; i < check.rows; i++) {
        int r, c, s;
        fscanf(file, "%d %d %d\n", &r, &c, &s);
        assert(r * c == check.cols);

        for (int j = 0; j < check.cols; j++) {
            fscanf(file, "%f", &mt_pos(check, i, j));
        }
    }

    fclose(file);
}

void load_digit_labels(Matrix labels, Matrix check, const char* filepath) {
    FILE* file = fopen(filepath, "r");
    assert(file);

    for (int i = 0; i < labels.rows; i++) {
        int digit;
        fscanf(file, "%d", &digit);
        mt_pos(labels, i, 0) = digit / 9.0f;  // normalize
    }
    for (int i = 0; i < check.rows; i++) {
        int digit;
        fscanf(file, "%d", &digit);
        mt_pos(check, i, 0) = digit / 9.0f;  // normalize
    }

    fclose(file);
}

int main(void){
    srand(time(0));

    Matrix X = mt_create(SAMPLES, GRID*GRID);
    Matrix Y = mt_create(SAMPLES, OUTPUT_SIZE);

    Matrix X_check = mt_create(50000 - SAMPLES, GRID*GRID);
    Matrix Y_check = mt_create(50000 - SAMPLES, OUTPUT_SIZE);

    load_digit_dataset(X, X_check, "../data/digits.txt");
    load_digit_labels(Y, Y_check, "../data/labels.txt");

    // mt_print(mt_row(X, SAMPLES-1));
    // mt_print(mt_row(Y, SAMPLES-1));

    // return 0;

    int arch[] = {GRID*GRID, GRID, GRID, 1};
    int arch_len = sizeof(arch)/sizeof(arch[0]);
    int l_count = arch_len - 1;
    

    N_Net nn = create_n_net(l_count, arch_len, arch);
    rand_n_net(nn, -2.0f, 2.0f);
    N_Net grad = create_n_net(l_count, arch_len, arch);
    float rate = (1);

    
    for (int epoch = 0; epoch < 2000; epoch++) {
        backprop_n_net(nn, grad, X, Y);
        learn_n_net(nn, grad, rate);
        mt_randomize_rows(X, Y);
    }
    
    printf("---------------------------------\n");
    
    float loss = 0;
    for (int i = 0; i < SAMPLES; i++) {
        set_n_net_input(nn, mt_row(X, i));
        forward_n_net(nn);
        loss += loss_n_net(nn, mt_row(Y, i));
    }
    printf("Loss: %8f\n", loss / SAMPLES);

    int correct = 0;
    for(int i = 0; i < 1000; i++){
        set_n_net_input(nn, mt_row(X_check, i));
        forward_n_net(nn);
        int out = (int)(mt_pos(output_n_net(nn), 0, 0)*9 + 0.5);
        int expect = (int)(mt_pos(Y_check, i, 0)*9 + 0.5);
        // printf("Out: %d, Expect: %d\n", out, expect);

        if(out == expect) correct++;
    }
    printf("Correct: %f %%", (float)correct/10.0f);

    save_n_net(nn, "nn_config.txt");

    return 0;
}