#include <stdlib.h>

#include "mlp.h"

// WARN: mlp will store pointer layer_sizes, so DO NOT free it before mlp been
// free.
struct mlp *mlp_create(const size_t *layer_sizes, size_t layer_size_count) {
  if (layer_sizes == NULL || layer_size_count == 0U) {
    return NULL;
  }

  size_t layer_full_size = layer_size_count * 5 - 4;

  size_t *layer_addr_offsets =
      (size_t *)malloc(layer_full_size * sizeof(size_t));
  if (layer_addr_offsets == NULL) {
    return NULL;
  }
  float **layers = (float **)malloc(layer_full_size * sizeof(float *));
  if (layers == NULL) {
    free(layer_addr_offsets);
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
      // this branch reads all element of layer_size, so do layer size check
      // (cannot be zero) here.
      if (current_size == 0) {
        free(layer_addr_offsets);
        free(layers);
        return NULL;
      }
    } else if (i < backward_layer_offset) { // offsets for output_layers
      current_size = layer_sizes[i - layer_size_count + 1U];
    } else if (i < weight_offset) { // offsets for backward_layers
      current_size = layer_sizes[i - backward_layer_offset + 1U];
    } else if (i < bias_offset) { // offsets for weights
      current_layer_idx = i - weight_offset + 1U;
      if (SIZE_MAX / layer_sizes[current_layer_idx] <
          layer_sizes[current_layer_idx - 1U]) {
        free(layer_addr_offsets);
        free(layers);
        return NULL;
      }
      current_size =
          layer_sizes[current_layer_idx] * layer_sizes[current_layer_idx - 1U];
    } else { // offsets for bias
      current_size = layer_sizes[i - bias_offset + 1U];
    }

    if (SIZE_MAX - current_size < offset_sum) {
      free(layer_addr_offsets);
      free(layers);
      return NULL;
    }
    // get address offset
    layer_addr_offsets[i] = offset_sum;
    offset_sum += current_size;
  }

  // size in this malloc
  // = size sum of each layer of layers
  // + size sum of each layer(except first layer) of output_layers
  // + size sum of each layer(except first layer) of backward_layers
  // + size sum of each layer(except first layer) of bias
  // + size sum of each layer(except first layer) of weights
  float *layer_neurons = (float *)malloc(offset_sum * sizeof(float));
  if (layer_neurons == NULL) {
    free(layer_addr_offsets);
    free(layers);
    return NULL;
  }
  // offset to real pointer
  for (i = 0; i < layer_full_size; ++i) {
    layers[i] = layer_neurons + layer_addr_offsets[i];
  }
  free(layer_addr_offsets);

  struct mlp *mlp = (struct mlp *)malloc(sizeof(struct mlp));
  if (mlp == NULL) {
    free(layer_neurons);
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
  mlp->activation_funcs_ref = NULL;

  mlp_fill_bias(mlp, 0.0f);
  mlp_fill_weights(mlp, 0.0f);

  return mlp;
}

void mlp_destroy(struct mlp *mlp) {
  if (mlp == NULL) {
    return;
  }

  float **layers = mlp->layers;
  free(mlp);
  if (layers == NULL) {
    return;
  }

  float *layer_elements = layers[0];
  free(layers);
  if (layer_elements == NULL) {
    return;
  }

  free(layer_elements);
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
    if (mlp->bias[i] == NULL) {
      return 2;
    }
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
    if (mlp->weights[i] == NULL) {
      return 2;
    }
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
    if (mlp->weights[i] == NULL) {
      return 2;
    }
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
  float **const biases = mlp->bias;
  float **const weights = mlp->weights;
  float (**activation_funcs_ref)(float) = mlp->activation_funcs_ref;
  if (layers == NULL || output_layers == NULL || biases == NULL ||
      weights == NULL || activation_funcs_ref == NULL) {
    return 2;
  }
  float *const input_layer = layers[0];
  if (input_layer == NULL) {
    return 3;
  }
  size_t j_idx;
  for (i = 0; i < layer_sizes[0]; ++i) {
    input_layer[i] = input[i];
  }

  size_t max_input_layer_idx = mlp->layer_size_count - 1U;
  float *current_layer = NULL, *next_layer = NULL, *output_layer = NULL,
        *weight = NULL, *bias = NULL;
  float (*activation_function)(float) = NULL;

  for (i = 0; i < max_input_layer_idx; ++i) {
    back_layer_size = layer_sizes[i];
    front_layer_size = layer_sizes[i + 1];

    current_layer = layers[i];
    next_layer = layers[i + 1U];
    output_layer = output_layers[i];
    weight = weights[i];
    bias = biases[i];
    activation_function = activation_funcs_ref[i];

    if (current_layer == NULL || next_layer == NULL || output_layer == NULL ||
        weight == NULL || bias == NULL || activation_function == NULL) {
      return 3;
    }

    for (j = 0; j < front_layer_size; ++j) {
      output_layer[j] = bias[j];
      j_idx = j * back_layer_size;

      for (k = 0; k < back_layer_size; ++k) {
        output_layer[j] += weight[j_idx + k] * current_layer[k];
      }

      next_layer[j] = activation_function(output_layers[i][j]);
    }
  }
  if (output != NULL) {
    float *final_layer = layers[max_input_layer_idx];
    for (i = 0; i < layer_sizes[max_input_layer_idx]; ++i) {
      output[i] = final_layer[i];
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
  float **const biases = mlp->bias;
  float **const weights = mlp->weights;
  float (**activation_funcs_ref)(float) = mlp->activation_funcs_ref;
  if (layers == NULL || output_layers == NULL || backward_layers == NULL ||
      biases == NULL || weights == NULL || weights == NULL ||
      activation_funcs_ref == NULL) {
    return 2;
  }

  float *const final_output_layer = layers[last_layer_idx];
  float *const final_backward_layer = backward_layers[last_layer_idx - 1U];
  if (final_output_layer == NULL || final_backward_layer == NULL) {
    return 3;
  }
  for (i = 0; i < layer_sizes[last_layer_idx]; ++i) {
    // L(a) = 0.5 * (a - y)^2
    // L'(a) = a - y
    // y is for expect output
    final_backward_layer[i] = (final_output_layer[i] - answer[i]);
  }

  size_t back_layer_size;
  size_t front_layer_size;
  size_t j_idx;
  float *layer = NULL, *output_layer = NULL, *current_backward_layer = NULL,
        *last_backward_layer = NULL, *current_weights = NULL,
        *current_biases = NULL;
  float (*activation_function)(float) = NULL;
  for (i = last_layer_idx; i > 0;) {
    --i;

    front_layer_size = layer_sizes[i + 1U];
    back_layer_size = layer_sizes[i];

    layer = layers[i];
    current_backward_layer = backward_layers[i];
    output_layer = output_layers[i];
    current_weights = weights[i];
    current_biases = biases[i];
    activation_function = activation_funcs_ref[i];
    if (layer == NULL || current_backward_layer == NULL ||
        current_weights == NULL || current_biases == NULL ||
        activation_function == NULL) {
      return 3;
    }

    if (i > 0) {
      last_backward_layer = backward_layers[i - 1U];
      if (last_backward_layer == NULL) {
        return 3;
      }
    }

    for (j = 0; j < front_layer_size; ++j) {
      // dL/da * da/dz * dz/dn
      // L is for loss function
      // a is for activate function
      // z is for output
      // n is for any part, there are weight(w), input(x) and bias(b).

      // = dL/da * da/dz
      current_backward_layer[j] *=
          (activation_function(output_layer[j] + 4.76837e-7f) -
           activation_function(output_layer[j] - 4.76837e-7f)) /
          9.53674e-7f;

      j_idx = j * back_layer_size;
      for (k = 0; k < back_layer_size; ++k) {

        // pass gradiant to last layer first
        // dz/dx = w
        if (i > 0) {
          last_backward_layer[k] =
              (j == 0 ? 0.0f : last_backward_layer[k]) +
              current_weights[j_idx + k] * current_backward_layer[j];
        }

        // dz/dw = x
        current_weights[j_idx + k] -=
            learn_rate * layer[k] * current_backward_layer[j];
      }

      // dz/db = 1
      current_biases[j] -= learn_rate * current_backward_layer[j];
    }
  }

  return 0;
}