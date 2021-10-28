from cos_similarity import count_context
import numpy as np


def test_count_context_1():
  vocab_size = 3
  samples = [
    [1, 1, 1, 1, 0, 0],
  ]
  true_counts = [
    [2, 2, 0],
    [2, 6, 0],
    [0, 0, 0],
  ]
  calculated_counts = count_context(samples, vocab_size, 1, 1)
  np.testing.assert_array_equal(calculated_counts, true_counts)


def test_count_context_2():
  vocab_size = 3
  samples = [
    [1, 2, 1, 1, 2],
  ]
  true_counts = [
    [1, 1, 0],
    [1, 2, 3],
    [0, 2, 0],
  ]
  calculated_counts = count_context(samples, vocab_size, 1, 1)
  np.testing.assert_array_equal(calculated_counts, true_counts)

# test_count_context_1()