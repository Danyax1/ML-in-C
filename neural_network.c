#include "neural_network.h"

float ReLU (float val){
    if (val <= 0){
        return 0.0f;
    } else {
        return val;
    }
}

float sigmoid(float val){
    return 1 / (1 +expf(-val));
}

float step(float val){
    if (val < 0){
        return 0.0f;
    }else {
        return 1.0f;
    }
}
void mt_activate(Matrix m, float (*activation)(float)){
    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++){
            mt_pos(m, i, j) = activation(mt_pos(m, i, j));
        }
    }
}

N_Net create_n_net (int l_count, int arch_len, int* arch){
    assert(l_count + 1 == arch_len);

    Matrix* mat_arr = malloc((arch_len - 1) * sizeof(Matrix) * 3);

    for (int i = 0; i < l_count; i++) {
        mat_arr[i * 3]     = mt_create(arch[i], arch[i + 1]); // weights
        mat_arr[i * 3 + 1] = mt_create(1, arch[i + 1]);       // biases
        mat_arr[i * 3 + 2] = mt_create(1, arch[i + 1]);       // activations
    }

    N_Net nn={
        .l_count  = l_count,
        .arch_len = arch_len,
        .arch     = arch,
        .mats     = mat_arr
    };
    return nn;
};

void rand_n_net(N_Net nn, float low, float high){
    for (int i = 0; i < 3* nn.l_count; i++){
        mt_rand(nn.mats[i], low, high);
    }
};


void N_NET_PRINT(N_Net nn, const char *name){
    printf("%s = {{\n", name);

    for (int i = 1; i < nn.arch_len; i++){
        printf("    ");
        MATRIX_PRINT(nn.mats[(i-1)*3], "w_n");
        printf("\n");

        printf("    ");
        MATRIX_PRINT(nn.mats[(i-1)*3 + 1], "b_n");
        printf("\n");

        printf("    ");
        MATRIX_PRINT(nn.mats[(i-1)*3 + 2], "a_n");
        printf("\n");
    }
    printf("}}\n");
};