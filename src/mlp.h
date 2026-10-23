#ifndef MLP_H_
#define MLP_H_

#include <stddef.h>
#include <stdint.h>

// NOTE: layers and its first pointer also need to free.
struct mlp {
  float learning_rate;
  size_t layer_size_count;               // N
  const size_t *layer_sizes_ref;         // n1, n2...
  float **layers;                        // input & activated ouput (N)
  float **output_layers;                 // output (N-1)
  float **backward_layers;               // passes gradiants (N-1)
  float **weights;                       // weights(N-1)
  float **biases;                        // bias (N-1)
  float (**activation_funcs_ref)(float); // activation functions(N-1)
};

// layer_sizes_ref will keep and store in mlp. DO NOT free layer_sizes_ref
// before mlp been free. or you can use `free_mlp` to free savely.
struct mlp *mlp_create(const size_t *layer_sizes_ref, size_t layer_size_count);

// free mlp savely.
void mlp_destroy(struct mlp *mlp);

int mlp_set_learning_rate(struct mlp *mlp, float learning_rate);

int mlp_set_activation_fns_ref(struct mlp *mlp,
                               float (**activation_fns_ref)(float));

// fill all bias with a value.
int mlp_fill_biases(struct mlp *mlp, float value);

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