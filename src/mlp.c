#include <stdlib.h>

#include "mlp.h"

// WARN: mlp will store pointer layer_sizes, so DO NOT free it before mlp been
// free.
struct mlp *mlp_create(const size_t *layer_sizes, size_t layer_size_count) {
  if (layer_sizes == NULL) {
    return NULL;
  }

  size_t layer_full_size = layer_size_count * 5 - 4;

  float **layers = (float **)malloc(layer_full_size * sizeof(float *));
  if (layers == NULL) {
    return NULL;
  }

  const size_t output_layer_offset = layer_size_count;
  const size_t backward_layer_offset = layer_size_count * 2U - 1U;
  const size_t weight_offset = layer_size_count * 3U - 2U;
  const size_t bias_offset = layer_size_count * 4U - 3U;

  // get offsets before malloc at once.
  size_t i, offset_sum = 0, current_size, current_layer_idx;
  for (i = 0; i < layer_full_size; ++i) {
    if (i < output_layer_offset) { // offsets for layers
      current_size = layer_sizes[i];
    } else if (i < backward_layer_offset) { // offsets for output_layers
      current_size = layer_sizes[i - layer_size_count + 1U];
    } else if (i < weight_offset) { // offsets for backward_layers
      current_size = layer_sizes[i - backward_layer_offset + 1U];
    } else if (i < bias_offset) { // offsets for weights
      current_layer_idx = i - weight_offset + 1U;
      current_size =
          layer_sizes[current_layer_idx] * layer_sizes[current_layer_idx - 1U];
    } else { // offsets for bias
      current_size = layer_sizes[i - bias_offset + 1U];
    }

    if (SIZE_MAX - current_size < offset_sum) {
      free(layers);
      return NULL;
    }
    // get address offset
    layers[i] = (float *)offset_sum;
    offset_sum += current_size;
  }

  // size in this malloc
  // = size sum of each layer of layers
  // + size sum of each layer(except first layer) of output_layers
  // + size sum of each layer(except first layer) of backward_layers
  // + size sum of each layer(except first layer) of bias
  // + size sum of each layer(except first layer) of weights
  float *layer_elements = (float *)malloc(offset_sum * sizeof(float));
  if (layer_elements == NULL) {
    free(layers);
    return NULL;
  }
  // offset to real pointer
  for (i = 0; i < layer_full_size; ++i) {
    layers[i] = layer_elements + (size_t)layers[i];
  }

  struct mlp *mlp = (struct mlp *)malloc(sizeof(struct mlp));
  if (mlp == NULL) {
    free(layer_elements);
    free(layers);
    return NULL;
  }

  mlp->layer_size_count = layer_size_count;
  mlp->layer_sizes_ref = layer_sizes;
  mlp->layers = layers;
  mlp->output_layers = layers + output_layer_offset;
  mlp->backward_layers = layers + backward_layer_offset;
  mlp->bias = layers + bias_offset;
  mlp->weights = layers + weight_offset;

  return mlp;
}

void mlp_destroy(struct mlp *mlp) {
  if (mlp == NULL) {
    return;
  }
  float **layers = mlp->layers;
  float *layer_elements = layers[0];
  free(layer_elements);
  free(layers);
  free(mlp);
}

int mlp_set_learning_rate(struct mlp *mlp, float learning_rate) {
  if (mlp == NULL) {
    return 1;
  }
  mlp->learning_rate = learning_rate;
  return 0;
}

int mlp_set_activation_fns_ref(struct mlp *mlp,
                               float (**activation_funcs)(float)) {
  if (mlp == NULL) {
    return 1;
  }
  if (activation_funcs == NULL) {
    return 2;
  }
  mlp->activation_funcs_ref = activation_funcs;
  return 0;
}

int mlp_fill_bias(struct mlp *mlp, const float value) {
  if (mlp == NULL) {
    return 1;
  }
  uint32_t i, j;
  size_t size;
  for (i = 0; i < mlp->layer_size_count - 1U; ++i) {
    size = mlp->layer_sizes_ref[i + 1];
    for (j = 0; j < size; ++j) {
      mlp->bias[i][j] = value;
    }
  }
  return 0;
}

int mlp_fill_weights(struct mlp *mlp, const float value) {
  if (mlp == NULL) {
    return 1;
  }
  uint32_t i, j;
  size_t size;
  for (i = 0; i < mlp->layer_size_count - 1U; ++i) {
    size = mlp->layer_sizes_ref[i] * mlp->layer_sizes_ref[i + 1];
    for (j = 0; j < size; ++j) {
      mlp->weights[i][j] = value;
    }
  }
  return 0;
}

int mlp_fill_random_weights(struct mlp *mlp, uint32_t seed) {
  if (mlp == NULL) {
    return 1;
  }
  uint32_t i, j;
  size_t size;
  for (i = 0; i < mlp->layer_size_count - 1U; ++i) {
    size = mlp->layer_sizes_ref[i] * mlp->layer_sizes_ref[i + 1];
    for (j = 0; j < size; ++j) {

      seed ^= seed << 7;
      seed ^= seed >> 19;
      seed ^= seed << 13;

      mlp->weights[i][j] = 2 * ((float)seed / UINT32_MAX) - 1;
    }
  }
  return 0;
}

int mlp_forward(struct mlp *mlp, const float *const input, float *output) {
  if (mlp == NULL || input == NULL) {
    return 1;
  }
  uint32_t i, j, k;
  const size_t *layer_sizes = mlp->layer_sizes_ref;
  size_t back_layer_size;
  size_t front_layer_size;
  float **const layers = mlp->layers;
  float **const output_layers = mlp->output_layers;
  float **const bias = mlp->bias;
  float **const weights = mlp->weights;
  float *const input_layer = layers[0];
  size_t j_idx;
  for (i = 0; i < layer_sizes[0]; ++i) {
    input_layer[i] = input[i];
  }

  size_t max_input_layer_idx = mlp->layer_size_count - 1U;
  for (i = 0; i < max_input_layer_idx; ++i) {
    back_layer_size = layer_sizes[i];
    front_layer_size = layer_sizes[i + 1];

    for (j = 0; j < front_layer_size; ++j) {
      output_layers[i][j] = bias[i][j];
      j_idx = j * back_layer_size;

      for (k = 0; k < back_layer_size; ++k) {
        output_layers[i][j] += weights[i][j_idx + k] * layers[i][k];
      }

      layers[i + 1][j] = mlp->activation_funcs_ref[i](output_layers[i][j]);
    }
  }
  if (output != NULL) {
    float *last_layer = layers[max_input_layer_idx];
    for (i = 0; i < layer_sizes[max_input_layer_idx]; ++i) {
      output[i] = last_layer[i];
    }
  }
  return 0;
}

int mlp_backward(struct mlp *mlp, const float *const answer) {
  if (mlp == NULL || answer == NULL) {
    return 1;
  }

  uint32_t i, j, k;
  const float learn_rate = mlp->learning_rate;
  const size_t layer_size_count = mlp->layer_size_count;
  const size_t last_layer_idx = layer_size_count - 1U;

  const size_t *layer_sizes = mlp->layer_sizes_ref;
  float **const layers = mlp->layers;
  float **const output_layers = mlp->output_layers;
  float **const backward_layers = mlp->backward_layers;
  float **const bias = mlp->bias;
  float **const weights = mlp->weights;
  float (**activation_funcs)(float) = mlp->activation_funcs_ref;

  float *const last_output_layer = layers[last_layer_idx];
  float *const last_backward_layer = backward_layers[last_layer_idx - 1U];
  for (i = 0; i < layer_sizes[last_layer_idx]; ++i) {
    // L(a) = 0.5 * (a - y)^2
    // L'(a) = a - y
    // y is for expect output
    last_backward_layer[i] = (last_output_layer[i] - answer[i]);
  }

  size_t back_layer_size;
  size_t front_layer_size;
  size_t j_idx;
  for (i = last_layer_idx; i > 0;) {
    --i;

    front_layer_size = layer_sizes[i + 1U];
    back_layer_size = layer_sizes[i];
    for (j = 0; j < front_layer_size; ++j) {
      // dL/da * da/dz * dz/dn
      // L is for loss function
      // a is for activate function
      // z is for output
      // n is for any part, there are weight(w), input(x) and bias(b).

      // = dL/da * da/dz
      backward_layers[i][j] *=
          (activation_funcs[i](output_layers[i][j] + 4.76837e-7f) -
           activation_funcs[i](output_layers[i][j] - 4.76837e-7f)) /
          9.53674e-7f;

      j_idx = j * back_layer_size;
      for (k = 0; k < back_layer_size; ++k) {

        // pass gradiant to last layer first
        // dz/dx = w
        if (i > 0) {
          backward_layers[i - 1U][k] =
              (j == 0 ? 0.0f : backward_layers[i - 1U][k]) +
              weights[i][j_idx + k] * backward_layers[i][j];
        }

        // dz/dw = x
        weights[i][j_idx + k] -=
            learn_rate * layers[i][k] * backward_layers[i][j];
      }

      // dz/db = 1
      bias[i][j] -= learn_rate * backward_layers[i][j];
    }
  }

  return 0;
}