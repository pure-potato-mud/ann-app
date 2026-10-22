#include <stdbool.h>
#include <stddef.h>

#include <mlp.h>

#define TEST_FUNC(x) int x(void)
#define MLP_ARRANGE_FAIL_IF(mlp, expr)                                         \
  do {                                                                         \
    if (expr) {                                                                \
      mlp_destroy(mlp);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define MLP_ASSERT(mlp, expr)                                                  \
  do {                                                                         \
    if (!(expr)) {                                                             \
      mlp_destroy(mlp);                                                        \
      return 2;                                                                \
    }                                                                          \
  } while (0)

static float leaky_relu(float value) {
  return (value > 0) ? value : value * 0.03125f;
}

bool is_approx(float a, float b, float ref) {
  float tmp = a - b;
  tmp = tmp > 0 ? tmp : -tmp;
  return tmp <= ref;
}

bool is_all_approx(float *a, float b, size_t size, float ref) {
  int i;
  for (i = 0; i < size; ++i) {
    if (is_approx(a[i], b, ref)) {
      continue;
    }
    return false;
  }
  return true;
}

bool is_each_approx(float *a, float *b, size_t size, float ref) {
  int i;
  for (i = 0; i < size; ++i) {
    if (is_approx(a[i], b[i], ref)) {
      continue;
    }
    return false;
  }
  return true;
}

static const size_t layer_sizes[3] = {2, 3, 2};
static const size_t layer_size_count = 3;
static const float learning_rate = 0.125f;

TEST_FUNC(test_mlp_create) {
  struct mlp *mlp = mlp_create(layer_sizes, layer_size_count);

  if (mlp == NULL) {
    return 2;
  }
  mlp_destroy(mlp);
  return 0;
}

TEST_FUNC(test_mlp_create_null) {
  struct mlp *mlp = mlp_create(NULL, layer_size_count);

  if (mlp != NULL) {
    mlp_destroy(mlp);
    return 2;
  }
  return 0;
}

TEST_FUNC(test_mlp_set_learning_rate) {
  struct mlp *mlp = mlp_create(layer_sizes, layer_size_count);
  if (mlp == NULL) {
    return 1;
  }

  int result = mlp_set_learning_rate(mlp, learning_rate);

  MLP_ASSERT(mlp, !result);
  MLP_ASSERT(mlp, mlp->learning_rate == learning_rate);
  mlp_destroy(mlp);
  return 0;
}

TEST_FUNC(test_mlp_set_activation_fns_ref) {
  float (*activation_funcs[3])(float) = {leaky_relu, leaky_relu, leaky_relu};
  struct mlp *mlp = mlp_create(layer_sizes, layer_size_count);
  if (mlp == NULL) {
    return 1;
  }

  int result = mlp_set_activation_fns_ref(mlp, activation_funcs);

  MLP_ASSERT(mlp, !result);
  mlp_destroy(mlp);
  return 0;
}

TEST_FUNC(test_mlp_fill_bias) {
  struct mlp *mlp = mlp_create(layer_sizes, layer_size_count);
  if (mlp == NULL) {
    return 1;
  }

  int result = mlp_fill_bias(mlp, 1.0f);

  MLP_ASSERT(mlp, !result);
  MLP_ASSERT(mlp, is_all_approx(mlp->bias[0], 1.0f, 3, 0.0f));
  MLP_ASSERT(mlp, is_all_approx(mlp->bias[1], 1.0f, 2, 0.0f));
  mlp_destroy(mlp);
  return 0;
}

TEST_FUNC(test_mlp_fill_weights) {
  struct mlp *mlp = mlp_create(layer_sizes, layer_size_count);
  if (mlp == NULL) {
    return 1;
  }

  int result = mlp_fill_weights(mlp, 1.0f);

  MLP_ASSERT(mlp, !result);
  MLP_ASSERT(mlp, is_all_approx(mlp->weights[0], 1.0f, 6, 0.0f));
  MLP_ASSERT(mlp, is_all_approx(mlp->weights[1], 1.0f, 6, 0.0f));
  mlp_destroy(mlp);
  return 0;
}

TEST_FUNC(test_mlp_fill_random_weights) {
  struct mlp *mlp = mlp_create(layer_sizes, layer_size_count);
  if (mlp == NULL) {
    return 1;
  }

  int result = mlp_fill_weights(mlp, 1.0f);

  MLP_ASSERT(mlp, !result);
  int i;
  for (i = 0; i < 6; ++i) {
    float a = mlp->weights[0][i], b = mlp->weights[1][i];
    if (a > -1 && a <= 1 && b > -1 && b <= 1) {
      continue;
    }
    mlp_destroy(mlp);
    return 2;
  }
  mlp_destroy(mlp);
  return 0;
}

TEST_FUNC(test_mlp_forward) {
  float (*activation_funcs[3])(float) = {leaky_relu, leaky_relu, leaky_relu};
  struct mlp *mlp = mlp_create(layer_sizes, layer_size_count);
  if (mlp == NULL) {
    return 1;
  }
  MLP_ARRANGE_FAIL_IF(mlp, mlp_set_activation_fns_ref(mlp, activation_funcs));
  MLP_ARRANGE_FAIL_IF(mlp, mlp_set_learning_rate(mlp, 0.125f));
  MLP_ARRANGE_FAIL_IF(mlp, mlp_fill_bias(mlp, -0.5f));
  MLP_ARRANGE_FAIL_IF(mlp, mlp_fill_weights(mlp, 1));
  float input[2] = {0.5f, 2.0f};
  float output[2], except_output[2] = {5.5f, 5.5f};

  mlp_forward(mlp, input, output);

  MLP_ASSERT(mlp, is_each_approx(output, except_output, 2, 1e-7f));
  mlp_destroy(mlp);
  return 0;
}

TEST_FUNC(test_mlp_backward) {
  float (*activation_funcs[3])(float) = {leaky_relu, leaky_relu, leaky_relu};
  struct mlp *mlp = mlp_create(layer_sizes, layer_size_count);
  if (mlp == NULL) {
    return 1;
  }
  MLP_ARRANGE_FAIL_IF(mlp, mlp_set_activation_fns_ref(mlp, activation_funcs));
  MLP_ARRANGE_FAIL_IF(mlp, mlp_set_learning_rate(mlp, 0.125f));
  MLP_ARRANGE_FAIL_IF(mlp, mlp_fill_bias(mlp, -0.5f));
  MLP_ARRANGE_FAIL_IF(mlp, mlp_fill_weights(mlp, 1));
  float input[2] = {0.5f, 2.0f};
  float output[2], answer[2] = {1.0f, 0.0f};
  float except_output[2] = {-4133.0f / 131072, -4207.0f / 131072};

  mlp_forward(mlp, input, output);
  mlp_backward(mlp, answer);

  float weight_1[6] = {0.375f, -1.5f, 0.375f, -1.5f, 0.375f, -1.5f},
        bias_1[3] = {-1.75f, -1.75f, -1.75f},
        weight_2[6] = {-0.125f, -0.125f, -0.125f, -0.375f, -0.375f, -0.375f},
        bias_2[2] = {-17 / 16.0f, -19 / 16.0f};
  MLP_ASSERT(mlp, is_each_approx(mlp->weights[0], weight_1, 6, 3e-6f));
  MLP_ASSERT(mlp, is_each_approx(mlp->bias[0], bias_1, 3, 3e-6f));
  MLP_ASSERT(mlp, is_each_approx(mlp->weights[1], weight_2, 6, 3e-6f));
  MLP_ASSERT(mlp, is_each_approx(mlp->bias[1], bias_2, 2, 3e-6f));
  mlp_destroy(mlp);
  return 0;
}

int (*test_functions[])(void) = {
    test_mlp_create,
    test_mlp_create_null,
    test_mlp_set_learning_rate,
    test_mlp_set_activation_fns_ref,
    test_mlp_fill_bias,
    test_mlp_fill_weights,
    test_mlp_fill_random_weights,
    test_mlp_forward,
    test_mlp_backward,
};

int main(int argc, char **argv) {
  // select a test
  if (argc != 2) {
    return 1;
  }

  // string to int (only for unsigned)
  int test_index = 0, i;
  for (i = 0; argv[1][i] != '\0'; ++i) {
    if (argv[1][i] < '0' || argv[1][i] > '9') {
      return 1;
    }
    test_index = 10 * test_index + argv[1][i] - '0';
  }

  return test_functions[test_index]();
}
