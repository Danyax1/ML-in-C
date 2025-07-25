#ifndef NN_LIB_H
#define NN_LIB_H

#include <math.h>
#include "matrix_lib.h"

float ReLU (float val);
float sigmoid (float val);
float step (float val);

void mt_activate(Matrix m, float (*activation)(float));

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

N_Net create_n_net(int l_count, int arch_len, int* arch);
void rand_n_net(N_Net nn, float low, float high);
void N_NET_PRINT(N_Net nn, const char *name);


#endif //NN_LIB_H