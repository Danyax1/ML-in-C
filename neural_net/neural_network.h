#ifndef NN_LIB_H
#define NN_LIB_H

#include <math.h>
#include <stdbool.h>
#include "matrix_lib.h"

float ReLU (float val);
float sigmoid (float val);
float step (float val);

void mt_activate(const Matrix m, float (*activation)(float));

typedef struct
{
    int l_count;    // how many layers
    int arch_len;   // layers + 1
    int *arch;      // {5, 3, 2, 3}
    Matrix *w_n;      // array of weights
    Matrix *b_n;      // array of biases
    Matrix *a_n;      // array of activations

} N_Net;

#define print_n_net(nn) N_NET_PRINT((nn), (#nn));
#define input_n_net(nn) ((nn).a_n[0])
#define output_n_net(nn) ((nn).a_n[(nn).l_count])

N_Net create_n_net(const int l_count, const int arch_len, int* arch);
void free_n_net(N_Net *nn);
void rand_n_net(const N_Net nn, const float low, const float high);
void N_NET_PRINT(const N_Net nn, const char *name);

void set_n_net_input(const N_Net nn, const Matrix input);
void forward_n_net(const N_Net nn);
float loss_n_net(const N_Net nn, const Matrix expect);
void backprop_n_net(const N_Net nn, const N_Net grad, const Matrix input, const Matrix output);
void learn_n_net(const N_Net nn, const N_Net grad, const float rate);
void train_n_net(const N_Net nn, const N_Net grad, const Matrix inputs,const  Matrix outputs, int samples, float rate, int iters, bool show_proc);

void save_n_net(const N_Net nn, const char *filepath);
void load_n_net(const N_Net nn, const char *filepath);
#endif //NN_LIB_H