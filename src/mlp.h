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

// create a mlp.
// layer_sizes_ref will keep and store in mlp. DO NOT free layer_sizes_ref
// before mlp been free.
// layer_sizes_ref cannot be NULL and have no zero in it.
// also, layer_size_count cannot be zero.
struct mlp *mlp_create(const size_t *layer_sizes_ref, size_t layer_size_count);

// free mlp savely.
// NOTE: not include `*_ref`.
void mlp_destroy(struct mlp *mlp);

// set learning rate.
// the MLP cannot be NULL.
int mlp_set_learning_rate(struct mlp *mlp, float learning_rate);

// set activation functions reference.
// size must same as layer count of the MLP.
// the MLP and the activatioin functions reference cannot be NULL.
int mlp_set_activation_fns_ref(struct mlp *mlp,
                               float (**activation_fns_ref)(float));

// fill all bias with a value.
// MLP and it's biases cannot be NULL.
int mlp_fill_biases(struct mlp *mlp, float value);

// fill all weights with a value.
// MLP and it's weights field cannot be NULL.
int mlp_fill_weights(struct mlp *mlp, float value);

// fill all weights with random value within `(0.0, 1.0]`.
// MLP and it's weights field cannot be NULL.
int mlp_fill_random_weights(struct mlp *mlp, uint32_t seed);

// do forward propagation.
// the MLP(and it's members) and the input cannot be NULL, but the output can.
// the output will be fill in values if it is not NULL.
int mlp_forward(struct mlp *mlp, float const *input, float *output);

// do backpropagation.
// the MLP(and it's members) and answer cannot be NULL.
int mlp_backward(struct mlp *mlp, float const *answer);

#endif /* MLP_H_ */