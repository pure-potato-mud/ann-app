add_executable(test_mlp
  test_mlp.c
)

target_link_libraries(test_mlp
  PRIVATE mlp
)

add_test(NAME test_mlp_create COMMAND test_mlp 0)
add_test(NAME test_mlp_create_null COMMAND test_mlp 1)
add_test(NAME test_mlp_set_learning_rate COMMAND test_mlp 2)
add_test(NAME test_mlp_set_activation_fns_ref COMMAND test_mlp 3)
add_test(NAME test_mlp_fill_bias COMMAND test_mlp 4)
add_test(NAME test_mlp_fill_weights COMMAND test_mlp 5)
add_test(NAME test_mlp_fill_random_weights COMMAND test_mlp 6)
add_test(NAME test_mlp_forward COMMAND test_mlp 7)
add_test(NAME test_mlp_backward COMMAND test_mlp 8)
