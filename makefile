cython:
	rm *.so
	python3 setup_cython.py build_ext --inplace
	rm sampler.c cos_similarity.c
