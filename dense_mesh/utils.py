import numpy as np
import scipy.sparse as sparse

from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors

import pymeshlab


def poisson_sample_meshlab(
    path=None, vertices=None, faces=None, mesh=None, n_samples=1000, verbose=False
):
    """
    Compute a sample using poisson disk sampling from several possible format, either:
    - A path to a mesh file
    - A set of vertices and list of faces
    - A pyFM.TriMesh object

    Uses the pymeshlab function.

    Parameters:
    ----------------------
    path : str
        path to mesh file
    vertices : np.ndarray
        (n,3) vertices of mesh
    faces : np.ndarray
        list of faces of the mesh
    mesh : pyFM.TriMesh
        TriMesh object
    n_samples : int
        number of samples to target

    Output
    ----------------------
    samples : np.ndarray
        (n_samples, 3) coordinates of samples
    """
    ms = pymeshlab.MeshSet()
    if mesh is not None:
        ms.add_mesh(
            pymeshlab.Mesh(vertex_matrix=mesh.vertlist, face_matrix=mesh.facelist)
        )
    elif path is not None:
        ms.load_new_mesh(path)
    else:
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))

    ms.generate_sampling_poisson_disk(samplenum=n_samples)
    return ms.current_mesh().vertex_matrix()


def poisson_sample_mesh(mesh, n_samples=1000, force_n_samples=False):
    """
    Samples points on a mesh using poisson disk sampling.

    If necessary, the number of samples is increased until the desired number of samples is reached.

    Parameters:
    ----------------------
    mesh : pyFM.TriMesh
        mesh to sample
    n_samples : int
        number of samples to target
    force_n_samples : bool
        whether to force the number of samples to be exactly n_samples

    Output
    ----------------------
    subsample : np.ndarray
        indices of the subsample in the mesh
    """

    n_samples_arg = n_samples
    sample_vert = poisson_sample_meshlab(mesh=mesh, n_samples=n_samples_arg)
    subsample = np.unique(knn_query(mesh.vertlist, sample_vert))

    n_samples_curr = subsample.size

    while force_n_samples and n_samples_curr < n_samples:

        n_samples_arg = n_samples_arg * 2
        sample_vert = poisson_sample_meshlab(mesh=mesh, n_samples=n_samples_arg)
        subsample = np.unique(knn_query(mesh.vertlist, sample_vert))

        n_samples_curr = subsample.size

        subsample = subsample[:n_samples]

    return subsample


def knn_query(X, Y, k=1, return_distance=False, n_jobs=1):
    """
    Nearest neighbors query. Finds the nearest x in X for each y in Y.

    The order of the output is somewhat unusual, but follows the skikit learn convention.

    Parameters
    -------------------------------
    X: np.ndarray
        (n1,p) first collection
    Y: np.ndarray
        (n2,p) second collection
    k: int
        number of neighbors to look for
    return_distance: bool
        whether to return the nearest neighbor distance
    n_jobs: int
        number of parallel jobs. Set to -1 to use all processes

    Output
    -------------------------------
    dists: np.ndarray
        (n2,k) or (n2,) if k=1 - ONLY if return_distance is True. Nearest neighbor distance.
    matches: np.ndarray
        (n2,k) or (n2,) if k=1 - nearest neighbor
    """

    tree = NearestNeighbors(
        n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs
    )
    tree.fit(X)
    dists, matches = tree.kneighbors(Y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches


def my_tqrange(*args, verbose=False, **kwargs):
    if verbose:
        return tqdm(range(*args), **kwargs)
    return range(*args)


def linear_compact(x):
    """
    Linearly decreasing function between 0 and 1

    Parameters:
    ----------------------
    x : float or np.ndarray
        value between 0 and 1

    Output
    ----------------------
    y: float or np.ndarray
        1 - x
    """
    return 1 - x


def poly_compact(x):
    """
    Polynomial decreasing function between 0 and 1

    Parameters:
    ----------------------
    x : float or np.ndarray
        value between 0 and 1

    Output
    ----------------------
    y: float or np.ndarray
        1 - 3 x^2 + 2 x^3
    """
    return 1 - 3 * x**2 + 2 * x**3


def exp_compact(x):
    """
    Exponential decreasing function between 0 and 1

    Parameters:
    ----------------------
    x : float or np.ndarray
        value between 0 and 1

    Output
    ----------------------
    y: float or np.ndarray
        exp(1 - 1 / (1 - x^2))
    """
    return np.exp(1 - 1 / (1 - x**2))


def local_f(vals, rho, chi="exp", return_zeros_indices=False):
    """
    Apply a local function to a set of values

    Parameters:
    ----------------------
    vals : np.ndarray
        values to apply the local function to
    rho : float
        radius in which the local function is supported
    chi : str
        type of local function to use: 'exp', 'linear', 'poly
    return_zeros_indices : bool
        whether to return zero-values indices

    Output
    ----------------------
    newv : np.ndarray
        new values
    zeros_indices : np.ndarray
        indices of zero-values
    """
    newv = np.zeros_like(vals)

    # Check indices in radius
    inds = vals < rho

    # Compute local function
    if chi.lower() == "exp":
        newv[inds] = exp_compact(vals[inds] / rho)
    elif chi.lower() == "poly":
        newv[inds] = poly_compact(vals[inds] / rho)
    elif chi.lower() == "linear":
        newv[inds] = linear_compact(vals[inds] / rho)
    else:
        raise ValueError(
            'Only "poly", "exp" and "linear" accepted as local interpolation functions'
        )

    if return_zeros_indices:
        return newv, ~inds
    return newv


def build_edge_length_graph(vertices, edges):
    """
    Build the edge length graph from a set of vertices and edges.

    Parameters
    ----------------------
    vertices : np.ndarray
        (n,3) coordinates of vertices
    edges : np.ndarray
        (p,2) indices of edges

    Output
    ----------------------
    graph : scipy.sparse.csc_matrix
        (n,n) sparse matrix of the graph
    """
    I = edges[:, 0]  # (p,) # noqa E741
    J = edges[:, 1]  # (p,)
    V = np.linalg.norm(vertices[J] - vertices[I], axis=1)  # (p,)
    return sparse.csc_matrix((V, (I, J)), shape=(vertices.shape[0], vertices.shape[0]))


def get_data_inds_of_columns_inds_csc(indptr, col_inds):
    """
    Given indptr of a csc matrix and indices of columns, return the indices of entries of the columns
    in the list of data of a csc-matrix


    Parameters:
    ----------------------
    indptr : np.ndarray
        (n+1,) indptr of a csc matrix
    col_inds : np.ndarray
        (p,) indices of columns for which to extract the data indices

    Output
    ----------------------
    data_inds : np.ndarray
        (m,) indices of dataof columns in the csc matrix

    """
    # Starting indices for each column
    start_vals = indptr[col_inds]  # (p,)

    # Number of values in each column
    n_vals = np.diff(indptr)[col_inds]  # (p,)

    data_inds = np.concatenate(
        [start_vals[i] + np.arange(n_vals[i]) for i in range(len(col_inds))]
    )

    return data_inds
