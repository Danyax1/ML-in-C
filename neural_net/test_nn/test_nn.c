#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "..\neural_network.h"

#define EPSILON 1e-4f

void test_create_and_free() {
    int arch[] = {2, 4, 1};
    N_Net nn = create_n_net(2, 3, arch);
    assert(nn.arch_len == 3);
    assert(nn.l_count == 2);
    free_n_net(&nn);
    printf("test_create_and_free passed\n");
}

void test_forward_pass() {
    int arch[] = {2, 3, 1};
    N_Net nn = create_n_net(2, 3, arch);
    rand_n_net(nn, -0.5f, 0.5f);

    Matrix input = mt_create(1, 2);
    mt_pos(input, 0, 0) = 0.5f;
    mt_pos(input, 0, 1) = -0.3f;

    set_n_net_input(nn, input);
    forward_n_net(nn);

    Matrix out = output_n_net(nn);
    assert(out.cols == 1);
    assert(out.rows == 1);
    assert(out.data != NULL);

    mt_free(&input);
    free_n_net(&nn);
    printf("test_forward_pass passed\n");
}

void test_loss_function() {
    int arch[] = {2, 2, 1};
    N_Net nn = create_n_net(2, 3, arch);
    rand_n_net(nn, -0.1f, 0.1f);

    Matrix input = mt_create(1, 2);
    mt_fill(input, 0.5f);

    Matrix output = mt_create(1, 1);
    mt_fill(output, 1.0f);

    set_n_net_input(nn, input);
    forward_n_net(nn);
    float loss = loss_n_net(nn, output);
    assert(loss >= 0.0f);

    mt_free(&input);
    mt_free(&output);
    free_n_net(&nn);
    printf("test_loss_function passed\n");
}

void test_train_simple_xor() {
    int arch[] = {2, 4, 1};
    N_Net nn = create_n_net(2, 3, arch);
    N_Net grad = create_n_net(2, 3, arch);
    rand_n_net(nn, -1.0f, 1.0f);

    Matrix inputs = mt_create(4, 2);
    Matrix outputs = mt_create(4, 1);

    // XOR input/output
    mt_pos(inputs, 0, 0) = 0; mt_pos(inputs, 0, 1) = 0; mt_pos(outputs, 0, 0) = 0;
    mt_pos(inputs, 1, 0) = 0; mt_pos(inputs, 1, 1) = 1; mt_pos(outputs, 1, 0) = 1;
    mt_pos(inputs, 2, 0) = 1; mt_pos(inputs, 2, 1) = 0; mt_pos(outputs, 2, 0) = 1;
    mt_pos(inputs, 3, 0) = 1; mt_pos(inputs, 3, 1) = 1; mt_pos(outputs, 3, 0) = 0;

    train_n_net(nn, grad, inputs, outputs, 4, 0.5f, 3000, false);

    // Check that network learned something (loss is low)
    float loss = 0;
    for (int i = 0; i < 4; i++) {
        set_n_net_input(nn, mt_row(inputs, i));
        forward_n_net(nn);
        loss += loss_n_net(nn, mt_row(outputs, i));
    }
    loss /= 4.0f;
    assert(loss < 0.1f);

    mt_free(&inputs);
    mt_free(&outputs);
    free_n_net(&nn);
    free_n_net(&grad);
    printf("test_train_simple_xor passed\n");
}

void test_save_and_load() {
    int arch[] = {3, 2, 1};
    N_Net nn1 = create_n_net(2, 3, arch);
    rand_n_net(nn1, -0.1f, 0.1f);
    const char* file = "nn_test_save.txt";
    save_n_net(nn1, file);

    N_Net nn2 = create_n_net(2, 3, arch);
    load_n_net(nn2, file);

    for (int i = 0; i < nn1.l_count; i++) {
        for (int j = 0; j < nn1.w_n[i].rows; j++) {
            for (int k = 0; k < nn1.w_n[i].cols; k++) {
                float diff = fabs(mt_pos(nn1.w_n[i], j, k) - mt_pos(nn2.w_n[i], j, k));
                assert(diff < EPSILON);
            }
        }
    }

    free_n_net(&nn1);
    free_n_net(&nn2);
    printf("test_save_and_load passed\n");
}

int main() {
    test_create_and_free();
    test_forward_pass();
    test_loss_function();
    test_train_simple_xor();
    test_save_and_load();
    printf("All neural network tests passed!\n");
    return 0;
}
