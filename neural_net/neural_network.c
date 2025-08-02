#include "neural_network.h"

inline float ReLU (const float val){
    return (val <= 0) ?  0.0f : val;
}

inline float sigmoid(const float val){
    return 1 / (1 +expf(-val));
}

inline float step(const float val){
    return (val <= 0) ?  0.0f : 1.0f;
}
void mt_activate(const Matrix m, float (*activation)(float)){
    for (int i = 0; i < m.rows; i++){
        for (int j = 0; j < m.cols; j++){
            mt_pos(m, i, j) = activation(mt_pos(m, i, j));
        }
    }
}

N_Net create_n_net (const int l_count, const int arch_len, int* arch){
    assert(l_count + 1 == arch_len);

    Matrix* w_n = malloc((arch_len - 1) * sizeof(Matrix));
    assert(w_n);
    Matrix* b_n = malloc((arch_len - 1) * sizeof(Matrix));
    assert(b_n);
    Matrix* a_n = malloc((arch_len) * sizeof(Matrix));
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
void free_n_net(N_Net *nn) {
    if (!nn) return;

    for (int i = 0; i < nn->l_count; i++) {
        mt_free(&nn->w_n[i]);
        mt_free(&nn->b_n[i]);
    }

    for (int i = 0; i < nn->arch_len; i++) {
        mt_free(&nn->a_n[i]);
    }

    free(nn->w_n);
    free(nn->b_n);
    free(nn->a_n);

    nn->w_n = NULL;
    nn->b_n = NULL;
    nn->a_n = NULL;
    nn->l_count = 0;
    nn->arch_len = 0;
}

void rand_n_net(const N_Net nn, const float low, const float high){
    for (int i = 0; i < nn.l_count; i++){
        mt_rand(nn.w_n[i], low, high);
        mt_rand(nn.b_n[i], low, high);
    }
};


void N_NET_PRINT(const N_Net nn, const char *name){
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

void set_n_net_input(const N_Net nn, const Matrix input){
    assert(input.rows == 1);
    assert(nn.arch[0] == input.cols);

    mt_copy(nn.a_n[0], input);
};
void forward_n_net(const N_Net nn){
    for(int i = 0; i < nn.l_count; i++){
        mt_mult(nn.a_n[i+1], nn.a_n[i], nn.w_n[i]);
        mt_add(nn.a_n[i+1], nn.b_n[i]);
        mt_activate(nn.a_n[i+1], sigmoid);
    }
};

float loss_n_net(const N_Net nn, const Matrix expect){
    //note: neural net should be pre-forwarded
    assert(expect.rows == 1);
    assert(expect.cols == (output_n_net(nn)).cols);

    float diff = 0.0f;
    for (int i = 0; i < expect.cols; i++){
        const float v1 = mt_pos(output_n_net(nn), 0, i);
        const float v2 = mt_pos(expect, 0, i);
        diff += (v1 - v2)*(v1 - v2);
    }
    return diff;
};

void backprop_n_net(const N_Net nn, const N_Net grad, const Matrix input, const Matrix output) {
    assert(input.rows == output.rows);
    const int n = input.rows;

    // Clear gradients
    for (int i = 0; i < grad.l_count; i++) {
        mt_fill(grad.w_n[i], 0);
        mt_fill(grad.b_n[i], 0);
    }

    for (int sample = 0; sample < n; sample++) {
        set_n_net_input(nn, mt_row(input, sample));
        forward_n_net(nn);

        for (int i = 0; i <= nn.l_count; i++) {
            mt_fill(grad.a_n[i], 0);
        }

        for (int j = 0; j < output.cols; j++) {
            float const a = mt_pos(nn.a_n[nn.l_count], 0, j);
            float const y = mt_pos(output, sample, j);
            float const delta = 2 * (a - y) * a * (1 - a);
            mt_pos(grad.a_n[nn.l_count], 0, j) = delta;
        }

        for (int l = nn.l_count; l > 0; l--) {
            Matrix const a_prev = nn.a_n[l - 1];
            Matrix const delta_curr = grad.a_n[l];

            for (int j = 0; j < nn.a_n[l].cols; j++) {
                float const delta = mt_pos(delta_curr, 0, j);

                // Bias gradient
                mt_pos(grad.b_n[l - 1], 0, j) += delta;

                for (int k = 0; k < a_prev.cols; k++) {
                    float const a = mt_pos(a_prev, 0, k);
                    mt_pos(grad.w_n[l - 1], k, j) += delta * a;

                    float const w = mt_pos(nn.w_n[l - 1], k, j);
                    float const prev_a = mt_pos(a_prev, 0, k);
                    mt_pos(grad.a_n[l - 1], 0, k) += delta * w * prev_a * (1 - prev_a);
                }
            }
        }
    }

    for (int i = 0; i < grad.l_count; i++) {
        for (int j = 0; j < grad.w_n[i].rows; j++) {
            for (int k = 0; k < grad.w_n[i].cols; k++) {
                mt_pos(grad.w_n[i], j, k) /= (float)n;
            }
        }
        for (int j = 0; j < grad.b_n[i].cols; j++) {
            mt_pos(grad.b_n[i], 0, j) /= (float)n;
        }
    }
}


void learn_n_net(const N_Net nn, const N_Net grad, const float rate){
    //applies gradient to neural network
    assert(nn.arch_len == grad.arch_len);

    for (int i = 0; i < nn.l_count; i++){
        mt_scale(grad.w_n[i], -rate);
        mt_scale(grad.b_n[i], -rate);
        mt_add(nn.w_n[i], grad.w_n[i]);
        mt_add(nn.b_n[i], grad.b_n[i]);
    }
}

void train_n_net(const N_Net nn, const N_Net grad, const Matrix inputs, const Matrix outputs, int samples, float rate, int iters, bool show_proc){
    for(int iter = 0; iter < iters; iter++){

        float loss = 0;

        for(int i = 0; i < samples; i++){
            set_n_net_input(nn, mt_row(inputs, i));
            forward_n_net(nn);
            const float mse = loss_n_net(nn, mt_row(outputs, i));
            loss += mse;
        }
        if (show_proc){
            printf("%-6d loss: %f\n", iter, loss/(float)samples);
        }
        backprop_n_net(nn, grad, inputs, outputs);
        learn_n_net(nn, grad, rate);
    }
};

void save_n_net(const N_Net nn, const char *filepath){
    FILE *config = fopen(filepath, "w");
    assert(config != NULL);
    fprintf(config, "%d\n", nn.arch_len);

    for(int i = 0; i < nn.arch_len; i++){
        fprintf(config, "%d ", nn.arch[i]);
    }
    fprintf(config, "\n");

    for(int i = 0; i < nn.l_count; i++){
        for(int j = 0; j < nn.w_n[i].rows; j++){
            for(int k = 0; k < nn.w_n[i].cols; k++){
                fprintf(config, "%f ", mt_pos(nn.w_n[i], j, k));
            }
            fprintf(config, "\n");
        }
        fprintf(config, "\n");
    }
    fprintf(config, "\n");
    for(int i = 0; i < nn.l_count; i++){
        for(int j = 0; j < nn.b_n[i].rows; j++){
            for(int k = 0; k < nn.b_n[i].cols; k++){
                fprintf(config, "%f ", mt_pos(nn.b_n[i], j, k));
            }
            fprintf(config, "\n");
        }
        fprintf(config, "\n");
    }

    fclose(config);
};

void load_n_net(const N_Net nn, const char *filepath){
    FILE *config = fopen(filepath, "r");
    assert(config != NULL);
    
    int arch_len;
    fscanf(config, "%d\n", &arch_len);
    assert(arch_len == nn.arch_len);

    int n_count;
    for(int i = 0; i < arch_len; i++){
        fscanf(config, "%d ", &n_count);
        assert(n_count == nn.arch[i]);
    }
    fscanf(config, "\n");

    for(int i = 0; i < nn.l_count; i++){
        for(int j = 0; j < nn.w_n[i].rows; j++){
            for(int k = 0; k < nn.w_n[i].cols; k++){
                fscanf(config, "%f ", &mt_pos(nn.w_n[i], j, k));
            }
            fscanf(config, "\n");
        }
        fscanf(config, "\n");
    }
    fscanf(config, "\n");
    for(int i = 0; i < nn.l_count; i++){
        for(int j = 0; j < nn.b_n[i].rows; j++){
            for(int k = 0; k < nn.b_n[i].cols; k++){
                fscanf(config, "%f ", &mt_pos(nn.b_n[i], j, k));
            }
            fscanf(config, "\n");
        }
        fscanf(config, "\n");
    }

    fclose(config);
};