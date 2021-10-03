# cython: infer_types=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION


import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline Py_ssize_t sample_from(const float[:] distr) nogil:
    cdef Py_ssize_t max_i = len(distr) - 1
    if max_i <= 0:
        return 0
    cdef float r = <float> rand() / <float> RAND_MAX
    cdef float cumulated_density = distr[0]
    cdef Py_ssize_t i = 0
    while r > cumulated_density and i < max_i:
        i += 1
        cumulated_density += distr[i]
    return i


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def sample_zerogram(
        np.ndarray input_ids,
        int max_id,
        float mask_prob,
):
    cdef np.ndarray fake_inputs_ids = np.copy(input_ids)

    cdef int[:] fake_inputs_ids_view = fake_inputs_ids.reshape(-1)

    cdef Py_ssize_t i
    cdef int input_len = fake_inputs_ids_view.size
    for i in range(input_len):
        if rand() < mask_prob * RAND_MAX:
            fake_inputs_ids_view[i] = <int> ((<float> rand() / <float> RAND_MAX) * max_id)

    return fake_inputs_ids


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def sample_monogram(
        np.ndarray input_ids,
        np.ndarray monogram,
        float mask_prob,
):
    cdef np.ndarray fake_inputs_ids = np.copy(input_ids)

    cdef int[:] fake_inputs_ids_view = fake_inputs_ids.reshape(-1)
    cdef float[:] monogram_view = monogram

    cdef Py_ssize_t i
    cdef int input_len = fake_inputs_ids_view.size
    for i in range(input_len):
        if rand() < mask_prob * RAND_MAX:
            fake_inputs_ids_view[i] = sample_from(monogram)

    return fake_inputs_ids


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def sample_bigram(
        np.ndarray input_ids,
        np.ndarray bigram,
        float mask_prob,
):
    cdef np.ndarray fake_inputs_ids = np.copy(input_ids)

    cdef int[:] input_ids_view = input_ids.reshape(-1)
    cdef int[:] fake_inputs_ids_view = fake_inputs_ids.reshape(-1)
    cdef float[:,:] bigram_view = bigram

    cdef Py_ssize_t i
    cdef int prev_token_id
    cdef int input_len = input_ids_view.size
    for i in range(1, input_len):
        if rand() < mask_prob * RAND_MAX:
            prev_token_id = input_ids_view[i-1]
            print(prev_token_id,<Py_ssize_t> prev_token_id)
            fake_inputs_ids_view[i] = <int> sample_from(
                bigram_view[<Py_ssize_t> prev_token_id]
            )

    return fake_inputs_ids