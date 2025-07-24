#ifndef NN_LIB_H
#define NN_LIB_H

#include <math.h>
#include "matrix_lib.h"

float ReLU (float val);
float sigmoid (float val);
float step (float val);

void matrix_activate(Matrix m, float (*activation)(float));

typedef struct
{
    int l_count;    // how many layers
    int arch_len;   // layers + 1
    int *arch;      // {5, 3, 2, 3}
    Matrix *mats;      // array of matrices

} N_Net;

#define n_net_print(nn) N_NET_PRINT((nn), (#nn));

N_Net create_n_net(int l_count, int arch_len, int* arch);
void rand_n_net(N_Net nn, float low, float high);
void print_n_net(N_Net nn);
void N_NET_PRINT(N_Net nn, const char *name);


#endif //NN_LIB_H