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

    Matrix* w_n = malloc((arch_len - 1) * sizeof(Matrix) * 3);
    assert(w_n);
    Matrix* b_n = malloc((arch_len - 1) * sizeof(Matrix) * 3);
    assert(b_n);
    Matrix* a_n = malloc((arch_len) * sizeof(Matrix) * 3);
    assert(a_n);

    for (int i = 0; i < l_count; i++) {
        w_n[i] = mt_create(arch[i], arch[i + 1]); // weights
        b_n[i] = mt_create(1, arch[i + 1]);       // biases
    }
    for (int i = 0; i < arch_len; i++){
        a_n[i] = mt_create(1, arch[i]);       // activations
    }
    N_Net nn={
        .l_count  = l_count,
        .arch_len = arch_len,
        .arch     = arch,
        .w_n      = w_n,
        .b_n      = b_n,
        .a_n      = a_n,
    };
    return nn;
};

void rand_n_net(N_Net nn, float low, float high){
    for (int i = 0; i < nn.l_count; i++){
        mt_rand(nn.w_n[i], low, high);
        mt_rand(nn.b_n[i], low, high);
    }
};


void N_NET_PRINT(N_Net nn, const char *name){
    printf("%s = {{\n", name);

    printf("    ");
    MATRIX_PRINT(input_n_net(nn), "a_in");

    char label[64];

    for (int i = 0; i < nn.l_count; i++) {
        printf("    ");
        snprintf(label, sizeof(label), "w_%d", i);
        MATRIX_PRINT(nn.w_n[i], label);
        printf("\n");

        printf("    ");
        snprintf(label, sizeof(label), "b_%d", i);
        MATRIX_PRINT(nn.b_n[i], label);
        printf("\n");

    }
    printf("    ");
    MATRIX_PRINT(output_n_net(nn), "a_out");

    printf("}}\n");
};

void set_n_net_input(N_Net nn, Matrix m){
    assert(m.rows == 1);
    assert(nn.arch[0] == m.cols);

    mt_copy(nn.a_n[0], m);
};
void forward_n_net(N_Net nn){
    for(int i = 0; i < nn.l_count; i++){
        mt_mult(nn.a_n[i+1], nn.a_n[i], nn.w_n[i]);
        mt_add(nn.a_n[i+1], nn.b_n[i]);
        mt_activate(nn.a_n[i+1], sigmoid);
    }
};