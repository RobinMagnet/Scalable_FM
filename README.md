# Scalable_FM
This is the code for our paper [Scalable and Efficient Functional Map Computations on Dense Meshes](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14746)

The code is not very clean yet, and relies on some tricks to compute efficiently local dijkstra distance, because scipy's dijkstra function always returns a dense matrix, with NaN at points further away than the given radius, making the function very slow in our case.
Instead, we first subdivide the shape in several parts and compute a local dijkstra on a subshape. 

This repository requires the package multiprocess
