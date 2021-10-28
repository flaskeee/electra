# cython: infer_types=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def count_context(
        samples, vocab_size,
        look_back, look_forward
):
  cdef np.ndarray counts = np.zeros((vocab_size,) * 2, dtype=np.int32)
  cdef int[:, :] counts_view = counts

  cdef Py_ssize_t window_size = look_back + look_forward + 1
  cdef Py_ssize_t curr_window_i = 0
  cdef Py_ssize_t curr_window_j
  cdef int[:] curr_window = np.zeros(window_size, dtype=np.int32)
  cdef Py_ssize_t curr_window_center
  cdef int[:] counts_to_update

  for s in samples:
    curr_window[:] = 0
    for curr_id in s:
      curr_window_center = (curr_window_i - look_forward) % window_size
      counts_to_update = counts_view[curr_window[curr_window_center]]
      curr_window[curr_window_i] = curr_id
      for curr_window_j in range(window_size):
        if curr_window_j != curr_window_center:
          counts_to_update[curr_window[curr_window_j]] += 1
      curr_window_i = (curr_window_i + 1) % window_size

  return counts




