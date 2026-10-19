#ifndef MLP_H_
#define MLP_H_

#include <stddef.h>
#include <stdint.h>

float activate(float value);
float d_activate(float value);

// NOTE: layers and its first pointer also need to free.
struct mlp_t {
  float learn_rate;
  size_t layer_size_count; // N
  size_t *layer_sizes;     // n1, n2...
  float **layers;          // input & activated ouput (N)
  float **output_layers;   // output (N-1)
  float **backward_layers; // passes gradiants (N-1)
  float **bias;            // bias (N-1)
  float **weights;         // weights(N-1)
};

// layer_sizes will keep and store in mlp. DO NOT free layer_sizes before mlp
// been free. or you can use `free_mlp` to free savely.
struct mlp_t *make_mlp(float learn_rate, size_t *layer_sizes,
                       size_t layer_size_count);

// free mlp savely.
void free_mlp(struct mlp_t *mlp);

// fill all bias with a value.
int fill_bias(struct mlp_t *mlp, float const value);

// fill all weights with a value.
int fill_weights(struct mlp_t *mlp, float const value);
int fill_random_weights(struct mlp_t *mlp, uint32_t seed);

// mlp and input cannot be NULL.
// non-zero return means error.
int mlp_forward(struct mlp_t *mlp, float const *const input, float *output);

// mlp and answer cannot be NULL.
// non-zero return means error.
int mlp_backward(struct mlp_t *mlp, float const *const answer);

#endif /* MLP_H_ */