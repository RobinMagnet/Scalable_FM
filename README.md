<p align="center">
<img src="images/texture_d4d_upsampled.png" width="400">
</p>

# Scalable and Efficient Functional Map Computations on Dense Meshes
This is the official implementation of the paper [Scalable and Efficient Functional Map Computations on Dense Meshes](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14746).

This paper relies on my [pyFM package](https://github.com/RobinMagnet/pyFM), which you can install or simply clone (it is very lightweight).
The demo also uses my [VisualizationTools](https://github.com/RobinMagnet/VisualizationTools) repo but that's simply for visualization, and is not needed in the main codebase `large_mesh.py`.

To use the repo, clone using

```
git clone --recurse-submodules git@github.com:RobinMagnet/Scalable_FM.git
```

You can run [```Demo.ipynb```](https://github.com/RobinMagnet/Scalable_FM/blob/main/Demo.ipynb) to see code for spectrum approximation and shape matching.

The only nonstandard dependency is the `pymeshlab` package, only for its implementation of Poisson Disk Sampling.

# Note on the code

The code follows the notations in the paper, and all functions have docstrings for documentation.

- It requires to work around some surprising behaviours of [scipy.sparse.csgraph.dijkstra](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html) when using maximal distance after which to stop.
This function indeed always returns a dense matrix, with NaN at points further away than the given radius instead of a sparse one.
Since this dense matrix usually doesn't fit in memory, we first subdivide the shape in several parts using K-Means and BallTree in 3D, and compute a local dijkstra on each subshape
- Upgrading the radius requires some tricks in order not to recompute all entries at each step.

If you have any question about the code, don't hesitate to reach out to me at robin.magnet [at] inria.fr


# Citations

If you use our work, please cite us

```
@article{magnetScalableEfficientFunctional2023,
author = {Magnet, Robin and Ovsjanikov, Maks},
year = {2023},
title = {Scalable and Efficient Functional Map Computations on Dense Meshes},
journal = {Computer Graphics Forum},
publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
}

```