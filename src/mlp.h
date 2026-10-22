#ifndef MLP_H_
#define MLP_H_

#include <stddef.h>
#include <stdint.h>

// NOTE: layers and its first pointer also need to free.
struct mlp {
  float learn_rate;
  size_t layer_size_count;           // N
  size_t *layer_sizes;               // n1, n2...
  float **layers;                    // input & activated ouput (N)
  float **output_layers;             // output (N-1)
  float **backward_layers;           // passes gradiants (N-1)
  float **bias;                      // bias (N-1)
  float **weights;                   // weights(N-1)
  float (**activation_funcs)(float); // activation functions(N-1)
};

// layer_sizes will keep and store in mlp. DO NOT free layer_sizes before mlp
// been free. or you can use `free_mlp` to free savely.
struct mlp *mlp_create(float learn_rate, size_t *layer_sizes,
                       float (**activation_funcs)(float),
                       size_t layer_size_count);

// free mlp savely.
void mlp_destroy(struct mlp *mlp);

// fill all bias with a value.
int mlp_fill_bias(struct mlp *mlp, float value);

// fill all weights with a value.
int mlp_fill_weights(struct mlp *mlp, float value);
int mlp_fill_random_weights(struct mlp *mlp, uint32_t seed);

// mlp and input cannot be NULL.
// non-zero return means error.
int mlp_forward(struct mlp *mlp, float const *input, float *output);

// mlp and answer cannot be NULL.
// non-zero return means error.
int mlp_backward(struct mlp *mlp, float const *answer);

#endif /* MLP_H_ */