<p align="center">
<img src="images/texture_d4d_upsampled.png" width="400">
</p>

# Scalable_FM
This is the code for our paper [Scalable and Efficient Functional Map Computations on Dense Meshes](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14746)

The code relies on some tricks to handle memory issues with the implementation of the dijkstra algorithm with fixed radius in `scipy`. In particular, this function always returns a dense matrix (potentially filled with `NaN`). To handle this, we first subdivisde the shape into several clusters and handle each cluster separately. This makes the code quite harder to read sadly but this provides major speed improvements.

The only nonstandard dependency is the `pymeshlab` package, only for its implementation of Poisson Disk Sampling.
