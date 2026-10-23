#include <stdbool.h>

#include <unity.h>

#include <mlp.h>

static float leaky_relu(float value) {
  return (value > 0) ? value : value * 0.03125f;
}

static struct mlp *mlp;

static const size_t layer_sizes[3] = {2, 3, 2};
static const size_t layer_size_count = 3;
static const float learning_rate = 0.125f;

void setUp() {
  mlp = NULL;
}

void tearDown() {
  mlp_destroy(mlp);
}

void test_mlp_create(void) {
  mlp = mlp_create(layer_sizes, layer_size_count);

  TEST_ASSERT_NOT_NULL(mlp);
}

void test_mlp_create_zero_size(void) {
  mlp = mlp_create(layer_sizes, 0);

  TEST_ASSERT_NULL(mlp);
}

void test_mlp_create_null(void) {
  mlp = mlp_create(NULL, layer_size_count);

  TEST_ASSERT_NULL(mlp);
}

void test_mlp_set_learning_rate(void) {
  mlp = mlp_create(layer_sizes, layer_size_count);
  TEST_ASSERT_NOT_NULL(mlp);

  TEST_ASSERT_UNLESS(mlp_set_learning_rate(mlp, learning_rate));

  TEST_ASSERT_EQUAL_FLOAT(learning_rate, mlp->learning_rate);
}

void test_mlp_set_activation_fns_ref(void) {
  float (*activation_funcs[3])(float) = {leaky_relu, leaky_relu, leaky_relu};
  mlp = mlp_create(layer_sizes, layer_size_count);
  TEST_ASSERT_NOT_NULL(mlp);

  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(mlp, activation_funcs));

  TEST_ASSERT_EQUAL_PTR(activation_funcs, mlp->activation_funcs_ref);
}

void test_mlp_fill_bias(void) {
  float except_layer1_biases[3] = {1.0f, 1.0f, 1.0f};
  float except_layer2_biases[2] = {1.0f, 1.0f};
  mlp = mlp_create(layer_sizes, layer_size_count);
  TEST_ASSERT_NOT_NULL(mlp);

  TEST_ASSERT_UNLESS(mlp_fill_bias(mlp, 1.0f));

  TEST_ASSERT_EQUAL_FLOAT_ARRAY(except_layer1_biases, mlp->bias[0], 3);
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(except_layer2_biases, mlp->bias[1], 2);
}

void test_mlp_fill_weights(void) {
  mlp = mlp_create(layer_sizes, layer_size_count);
  TEST_ASSERT_NOT_NULL(mlp);

  TEST_ASSERT_UNLESS(mlp_fill_weights(mlp, 1.0f));

  int i;
  for (i = 0; i < 6; ++i) {
    float weight_first_layer = mlp->weights[0][i],
          weight_second_layer = mlp->weights[1][i];
    TEST_ASSERT_EQUAL_FLOAT(1.0f, weight_first_layer);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, weight_second_layer);
  }
}

void test_mlp_fill_random_weights(void) {
  mlp = mlp_create(layer_sizes, layer_size_count);
  TEST_ASSERT_NOT_NULL(mlp);

  TEST_ASSERT_UNLESS(mlp_fill_random_weights(mlp, 0x32a9807bU));

  int i;
  for (i = 0; i < 6; ++i) {
    float weight_first_layer = mlp->weights[0][i],
          weight_second_layer = mlp->weights[1][i];
    TEST_ASSERT_GREATER_THAN_FLOAT(-1.0f, weight_first_layer);
    TEST_ASSERT_LESS_OR_EQUAL_FLOAT(1.0f, weight_first_layer);
    TEST_ASSERT_GREATER_THAN_FLOAT(-1.0f, weight_second_layer);
    TEST_ASSERT_LESS_OR_EQUAL_FLOAT(1.0f, weight_second_layer);
  }
}

void test_mlp_forward(void) {
  float input[2] = {0.5f, 2.0f};
  float output[2], except_output[2] = {5.5f, 5.5f};
  float (*activation_funcs[3])(float) = {leaky_relu, leaky_relu, leaky_relu};
  mlp = mlp_create(layer_sizes, layer_size_count);
  TEST_ASSERT_NOT_NULL(mlp);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(mlp, activation_funcs));
  TEST_ASSERT_UNLESS(mlp_set_learning_rate(mlp, 0.125f));
  TEST_ASSERT_UNLESS(mlp_fill_bias(mlp, -0.5f));
  TEST_ASSERT_UNLESS(mlp_fill_weights(mlp, 1));

  TEST_ASSERT_UNLESS(mlp_forward(mlp, input, output));

  TEST_ASSERT_FLOAT_ARRAY_WITHIN(1e-7f, except_output, output, 2);
}

void test_mlp_backward(void) {
  float input[2] = {0.5f, 2.0f};
  float output[2], answer[2] = {1.0f, 0.0f};
  float weight_1[6] = {0.375f, -1.5f, 0.375f, -1.5f, 0.375f, -1.5f},
        bias_1[3] = {-1.75f, -1.75f, -1.75f},
        weight_2[6] = {-0.125f, -0.125f, -0.125f, -0.375f, -0.375f, -0.375f},
        bias_2[2] = {-17 / 16.0f, -19 / 16.0f};
  float (*activation_funcs[3])(float) = {leaky_relu, leaky_relu, leaky_relu};
  mlp = mlp_create(layer_sizes, layer_size_count);
  TEST_ASSERT_NOT_NULL(mlp);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(mlp, activation_funcs));
  TEST_ASSERT_UNLESS(mlp_set_learning_rate(mlp, 0.125f));
  TEST_ASSERT_UNLESS(mlp_fill_bias(mlp, -0.5f));
  TEST_ASSERT_UNLESS(mlp_fill_weights(mlp, 1));

  mlp_forward(mlp, input, output);
  mlp_backward(mlp, answer);

  TEST_ASSERT_FLOAT_ARRAY_WITHIN(3e-6f, weight_1, mlp->weights[0], 6);
  TEST_ASSERT_FLOAT_ARRAY_WITHIN(3e-6f, bias_1, mlp->bias[0], 3);
  TEST_ASSERT_FLOAT_ARRAY_WITHIN(3e-6f, weight_2, mlp->weights[1], 6);
  TEST_ASSERT_FLOAT_ARRAY_WITHIN(3e-6f, bias_2, mlp->bias[1], 2);
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_mlp_create);
  RUN_TEST(test_mlp_create_zero_size);
  RUN_TEST(test_mlp_create_null);
  RUN_TEST(test_mlp_set_learning_rate);
  RUN_TEST(test_mlp_set_activation_fns_ref);
  RUN_TEST(test_mlp_fill_bias);
  RUN_TEST(test_mlp_fill_weights);
  RUN_TEST(test_mlp_fill_random_weights);
  RUN_TEST(test_mlp_forward);
  RUN_TEST(test_mlp_backward);
  return UNITY_END();
}
