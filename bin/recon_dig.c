#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "..\neural_net\matrix_lib.h"
#include "..\neural_net\neural_network.h"

#define GRID 14

int main(void) {
    float res = -1;

    Matrix passed_digit = mt_create(1, GRID * GRID);

    FILE* file = fopen("temp_digit.txt", "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open temp_digit.txt\n");
        return 1;
    }

    int r, c, s;
    fscanf(file, "%d %d %d\n", &r, &c, &s);
    for (int j = 0; j < passed_digit.cols; j++) {
        fscanf(file, "%f", &mt_pos(passed_digit, 0, j));
    }
    fclose(file);

    int arch[] = {GRID * GRID, GRID, GRID, 1};
    int arch_len = sizeof(arch) / sizeof(arch[0]);
    int l_count = arch_len - 1;

    N_Net nn = create_n_net(l_count, arch_len, arch);
    load_n_net(nn, "nn_config.txt");

    set_n_net_input(nn, mt_row(passed_digit, 0));
    forward_n_net(nn);

    res = mt_pos(output_n_net(nn), 0, 0) * 9.0f;

    printf("%f\n", res); 

    return 0;
}
