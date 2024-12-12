import time

import numpy as np
import scipy.sparse as sparse

from sklearn.neighbors import KDTree

from .local_dijkstra import compute_sparse_dijkstra
from .sample_update import update_sampled_points
from .radius_update import adapt_rho
from .utils import local_f, build_edge_length_graph, poisson_sample_mesh, knn_query


def build_local_mat(
    mesh1,
    subsample,
    rho,
    correct_dist=True,
    adapt_radius=True,
    self_weight_limit=0.4,
    update_sample=True,
    fast_update=True,
    n_clusters=4,
    interpolation="poly",
    return_dist=False,
    verbose=False,
):
    """
    Compute the local basis (and local distance matrix).

    Parameters:
    ----------------------
    mesh1     : pyFM.TriMesh object with n vertices
    subsample : (p,) indices of current samples
    rho       : radius
    update_sample : whether to add unseen vertices to the sample
    fast_update   : Make the sample update faster by batch-adding vertices
    interpolation  : 'poly', 'linear', 'exp' - type of local function

    Output
    ----------------------
    U : (n,p) sparse local functions at each columns
    subsample : indices of sampled points
    """
    N = mesh1.n_vertices
    rho_dijkstra = 3 * rho if correct_dist else rho

    radius_tree = KDTree(mesh1.vertlist, leaf_size=20)

    edge_length_graph = build_edge_length_graph(mesh1.vertlist, mesh1.edges)

    # First local dijkstra run
    In, Jn, Vn, unseen_inds = compute_sparse_dijkstra(
        mesh1,
        subsample,
        rho_dijkstra,
        rho_real=rho,
        n_clusters=n_clusters,
        correct_dist=correct_dist,
        edge_length_graph=edge_length_graph,
        radius_tree=radius_tree,
        verbose=verbose,
    )

    # Update sample set
    if update_sample:
        n_samples_base = subsample.size
        In, Jn, Vn, subsample = update_sampled_points(
            edge_length_graph,
            subsample,
            unseen_inds,
            rho_dijkstra,
            In,
            Jn,
            Vn,
            mesh1.vertices,
            correct_dist=correct_dist,
            radius_tree=radius_tree,
            rho_real=rho,
            directed=False,
            fast_update=fast_update,
            verbose=verbose,
        )
        if verbose:
            print(
                f"{subsample.size - n_samples_base:d} vertices have been added to the sample"
            )

    if adapt_radius:
        distmat = sparse.csc_matrix((Vn, (Jn, In)), shape=(N, subsample.size))  # (n,p)

        U, dist_data_csc, per_vertex_rho = adapt_rho(
            distmat,
            subsample,
            rho,
            threshold=self_weight_limit,
            rho_decrease_ratio=1.5,
            interpolation=interpolation,
            verbose=verbose,
        )

    if adapt_radius and (return_dist or update_sample):

        # Remove 0 in the distance matrix (but not on the diagonal)
        distmat.data = dist_data_csc
        distmat += sparse.csc_matrix(
            (
                np.full(subsample.size, 1e-10),
                (subsample, np.arange(subsample.size)),
            ),
            shape=distmat.shape,
        )
        distmat.eliminate_zeros()

        # Update unseen indices
        unseen_inds = np.where(U.sum(1) == 0)[0]
        if update_sample and len(unseen_inds) > 0:
            # Find unseen inds and compute new neighbords
            newI, newJ, newV, subsample = update_sampled_points(
                edge_length_graph,
                subsample,
                unseen_inds,
                per_vertex_rho.min(),
                [],
                [],
                [],
                mesh1.vertlist,
                correct_dist=correct_dist,
                radius_tree=radius_tree,
                rho_real=rho,
                directed=False,
                fast_update=fast_update,
                verbose=verbose,
            )

            U = U.tocoo(copy=False)

            # Add new values to U with the small rho
            Vn = np.concatenate(
                [U.data, local_f(newV, per_vertex_rho.min(), chi=interpolation)]
            )
            Jn = np.concatenate([U.row, newJ])
            In = np.concatenate([U.col, newI])
            U = sparse.csr_matrix(
                (Vn, (Jn, In)), shape=(mesh1.n_vertices, subsample.size)
            )

            # Update distance matrix accordingly
            if return_dist:
                distmat = distmat.tocoo(copy=False)

                Vn = np.concatenate([distmat.data, newV])
                Jn = np.concatenate([distmat.row, newJ])
                In = np.concatenate([distmat.col, newI])
                distmat = sparse.csr_matrix(
                    (Vn, (Jn, In)), shape=(mesh1.n_vertices, subsample.size)
                )
                # distmat += sparse.csc_matrix((newV, (newJ, newI)), shape=(mesh1.n_vertices, subsample.size))

        distmat = distmat.tocsr(copy=False)

    elif adapt_radius:
        del distmat

    else:
        # Build distance matrix
        U = sparse.csr_matrix((Vn, (Jn, In)), shape=(N, subsample.size))
        if return_dist:
            distmat = U.copy()

        # U is a simple transformation of the distance matrix
        U.data = local_f(U.data, rho, chi=interpolation)

    # Normalize U
    U = sparse.diags(1 / np.asarray(U.sum(1)).squeeze(), 0) @ U

    if return_dist:
        return U, subsample, distmat

    return U, subsample


def build_red_matrices(
    mesh1,
    subsample,
    dist_ratio=3,
    update_sample=True,
    interpolation="poly",
    correct_dist=True,
    return_dist=False,
    adapt_radius=False,
    fast_update=True,
    self_weight_limit=0.1,
    n_clusters=4,
    verbose=False,
):
    """
    Build matrices U, A_bar; W_bar and distance matrix

    Parameters
    ----------------------
    mesh1 : pyFM.mesh.TriMesh
        object with n vertices
    dist_ratio : float
        rho is set as rho = dist_ratio * average_radius
    update_sample : bool
        whether to add unseen vertices to the sample
    interpolation : str
        'poly', 'linear', 'exp' - type of local function
    correct_dist : bool
        If True, Replace dijkstra dist with euclidean after dijkstra
    return_dist : bool
        If True, return the sparse distance matrix
    adapt_radius : bool
        Whether to use the adaptive radius strategy
    fast_update : bool
        Whether to use the fast update heuristic (for debugging)
    self_limit : float
        Minimum value for self weight (for adaptive radius)
    n_clusters : int
        Number of cluster to use to first divide the shape (memory issues)

    Output
    ----------------------
    U : scipy.sparse.csr_matrix
        (n,p) sparse local functions at each columns
    A_bar : scipy.sparse.csr_matrix
        U^T A U with A the diagonal mass matrix
    W_bar : scipy.sparse.csr_matrix
        U^T W U with W the diagonal weight matrix
    subsample : np.ndarray
        (p,) indices of sampled points
    distmat : bool
        if return_dist is True, the sparse distance matrix (before applying local function)
    """
    n_samples = len(subsample)
    avg_radius = np.sqrt(mesh1.area / (np.pi * n_samples))
    rho = dist_ratio * avg_radius

    result_local = build_local_mat(
        mesh1,
        subsample,
        rho,
        correct_dist=correct_dist,
        adapt_radius=adapt_radius,
        self_weight_limit=self_weight_limit,
        update_sample=update_sample,
        fast_update=fast_update,
        n_clusters=n_clusters,
        interpolation=interpolation,
        return_dist=return_dist,
        verbose=verbose,
    )

    if return_dist:
        U, subsample, distmat = result_local
    else:
        U, subsample = result_local

    A_bar = U.T @ mesh1.A @ U
    W_bar = U.T @ mesh1.W @ U

    if return_dist:
        return U, A_bar, W_bar, subsample, distmat
    return U, A_bar, W_bar, subsample


def process_mesh(
    mesh1,
    n_samples=3000,
    force_n_samples=False,
    dist_ratio=3,
    update_sample=True,
    interpolation="poly",
    correct_dist=True,
    return_dist=False,
    adapt_radius=False,
    self_weight_limit=0.5,
    n_jobs=1,
    n_clusters=4,
    verbose=False,
):
    """
    Build matrices U, A_bar; W_bar and distance matrix

    Parameters:
    ----------------------
    mesh1 : pyFM.mesh.TriMesh
        mesh with n vertices
    n_samples : int
        Approximate number of samples to extract
    dist_ratio : float
        rho is set as rho = dist_ratio * average_radius
    update_sample : bool
        whether to add unseen vertices to the sample
    interpolation : str
        'poly', 'linear', 'exp' - type of local function
    correct_dist : bool
        If True, Replace dijkstra dist with euclidean after dijkstra
    return_dist : bool
        If True, return the sparse distance matrix
    batch_size : int
        Size of batches to use


    Output
    ----------------------
    U : (n,p) sparse local functions at each columns
    A_bar : U^T A U
    W_bar : U^T W U
    subsample : indices of sampled points
    distmat : if return_dist is True, the sparse distance matrix (before applying local function)
    """
    if verbose:
        print(f"Sampling {n_samples} vertices out of {mesh1.n_vertices}...")
        start_time = time.time()

    subsample = poisson_sample_mesh(
        mesh=mesh1, n_samples=n_samples, force_n_samples=force_n_samples
    )
    # subsample = np.unique(knn_query(mesh1.vertlist, vert_sub1))

    if verbose:
        print(
            f"\t{subsample.size} samples extracted in {time.time() - start_time:.2f}s"
        )

    result_red = build_red_matrices(
        mesh1,
        subsample,
        dist_ratio=dist_ratio,
        update_sample=update_sample,
        interpolation=interpolation,
        correct_dist=correct_dist,
        return_dist=return_dist,
        adapt_radius=adapt_radius,
        fast_update=True,
        self_weight_limit=self_weight_limit,
        n_clusters=n_clusters,
        verbose=verbose,
    )

    return result_red


def get_approx_spectrum(Wb, Ab, k=150, verbose=False):
    """
    Eigendecopositionusing Wbar and Abar
    """
    if verbose:
        print(f"Computing {k} eigenvectors")
        start_time = time.time()
    eigenvalues, eigenvectors = sparse.linalg.eigsh(Wb, k=k, M=Ab, sigma=-0.01)
    if verbose:
        print(f"\tDone in {time.time()-start_time:.2f} s")
    return eigenvalues, eigenvectors
