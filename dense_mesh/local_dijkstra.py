import numpy as np
import scipy.sparse as sparse

from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

from .utils import my_tqrange, build_edge_length_graph


def get_samples_in_cluster(labels, sample, class_index, return_indices=True):
    """
    Given a set M samples with labels, and a specific label l to consider,
    return the M' samples belonging to one label (and potentially their indices).

    Parameters:
    ----------------------
    labels : np.ndarray
        (m,) labels of the samples
    sample : np.ndarray
        (m,) or (m,p) values on the samples
    class_index : int
        index of the cluster to consider
    return_indices : bool
        whether to return the indices of the selected samples

    Output
    ----------------------
    samples_in_cluster : np.ndarray
        (m',) or (m',p) sample values in the cluster
    indices_in_cluster : np.ndarray
        (m',) indices of the samples in the original array
    """

    is_in_cluster = labels == class_index

    indices_in_cluster = np.where(is_in_cluster)[0]
    samples_in_cluster = sample[is_in_cluster]

    if return_indices:
        return samples_in_cluster, indices_in_cluster

    return samples_in_cluster


def find_nearby_points(sample_coords, radius_tree, rho):
    """Find points within a certain radius from a given set of points.
    Returns indices in sorted order.

    Parameters
    ----------------------
    sample_coords : np.ndarray
        (m,3) coordinates of the sample points
    radius_tree : sklearn.neighbors.KDTree or sklearn.neighbors.BallTree
        Tree on the complete set of vertices, had a `query_radius` method
    rho : float
        radius

    Output
    ----------------------
    nearby_points : np.ndarray
        (n',) indices of points within the radius
    """
    res_test = radius_tree.query_radius(sample_coords, r=rho)
    return np.unique(np.concatenate(res_test))


def get_subset_inds_from_sorted(indices1, indices_sub):
    """Given a SORTED set of indices, and a SORTED subset, finds the indices of the subset in the original set.

    Parameters
    ----------------------
    indices1 : np.ndarray
        (n,) indices of the original set
    indices_sub : np.ndarray
        (m,) indices of the subset

    Output
    ----------------------
    sourceinds : np.ndarray
        (m,) indices of the subset in the original set

    """

    sourceinds = np.searchsorted(indices1, indices_sub)
    return sourceinds


def extract_subgraph(graph, samples):
    """Extract rows and columns of a graph corresponding to a set of samples.

    Parameters
    ----------------------
    graph : scipy.sparse.csc_matrix
        (n,n) sparse matrix of the graph
    samples : np.ndarray
        (m,) indices of samples

    Output
    ----------------------
    subgraph : scipy.sparse.csc_matrix
        (m,m) subgraph of the graph corresponding to the samples
    """
    return graph[np.ix_(samples, samples)]


def correct_and_filter_distances(vertices, samples, I, J, rho=None):  # noqa C741
    """
    Given a cluster of points, correct the distances and filter them if necessary.
    The distances is given by the L2 distance in euclidean space (corrected distance compared to dijkstra).
    The filtering step simply removes distances that are bigger than a certain radius.

    Parameters
    ----------------------
    vertices : np.ndarray
        (n,3) coordinates of all vertices
    samples : np.ndarray
        (m,) indices of the sample points
    I : np.ndarray
        (m',) rows in the (m,n) distance matrix
    J : np.ndarray
        (m',) columns in the (m,n) distance matrix
    rho : float
        local radius, to filter the distances

    Output
    ----------------------
    I : np.ndarray
        (m'',) rows in the (m,n) distance matrix
    J : np.ndarray
        (m'',) columns in the (m,n) distance matrix
    V : np.ndarray
        (m'',) values of the distance
    """

    V = np.linalg.norm(
        vertices[J] - vertices[samples][I],
        axis=1,
    )  # (m',)

    if rho is not None:
        test_dist = V < rho  # (m',)
        I = I[test_dist]  # (m'',)  # noqa C741
        J = J[test_dist]  # (m'',)
        V = V[test_dist]  # (m'',)

    return I, J, V


def compute_local_dijkstra_with_tree(
    graph,
    vertices,
    subsample,
    rho_dijkstra,
    radius_tree=None,
    rho_real=None,
    correct_dist=False,
):
    """
    Compute local dijkstra distances from a set of samples to the vertices of a mesh.

    Parameters:
    ----------------------
    graph : scipy.sparse.csc_matrix
        (n,n) sparse matrix of the mesh graph (edge lengths)
    vertices : np.ndarray
        (n,3) coordinates of ALL vertices
    subsample : np.ndarray
        (m,) indices of the current samples
    rho_dijkstra : float
        radius for the dijkstra run
    radius_tree : sklearn.neighbors.KDTree or sklearn.neighbors.BallTree
        Tree on the complete set of vertices, had a `query_radius` method
    rho_real : float
        real radius for the correction (if correct_dist is True).  Since the dijkstra distance is bigger than the euclidean distance, rho_dijkstra should be bigger than the real radius.
    correct_dist : bool
        whether to correct the distance with L2 a posteriori (instead of dijkstra)

    Output
    ----------------------
    I,J,V : new entries of the sparse distance matrix
    I : np.ndarray
        (m',) rows in the (m,n) sparse distance matrix
    J : np.ndarray
        (m',) columns in the (m,n) sparse distance matrix
    V : np.ndarray
        (m',) values of the distance

    """
    if radius_tree is None:
        radius_tree = KDTree(vertices, leaf_size=20)

    # Step 1: Find all vertices near the current samples
    vinds_in_neigh = find_nearby_points(
        vertices[subsample], radius_tree, 1.01 * rho_dijkstra
    )  # (n_curr,)

    # Step 2: Indices of current samples in the vertex cluster
    sources_in_curr = get_subset_inds_from_sorted(
        vinds_in_neigh, subsample
    )  # (m_curr,)

    # Step 3: Extract the subgraph for the vertex cluster
    subgraph = extract_subgraph(graph, vinds_in_neigh)  # (n_curr, n_curr)

    # Step 4: Compute local Dijkstra distances from the samples in the subgraph
    dists_curr = sparse.csgraph.dijkstra(
        subgraph, directed=False, indices=sources_in_curr, limit=rho_dijkstra
    )  # (m_curr, n_curr)

    # Step 5: Find finite distances
    finite_test = dists_curr < np.inf  # (m_curr, n_curr)
    I, J = np.where(finite_test)  # noqa E741

    # Step 8: Correct distances if necessary
    if correct_dist:
        I, J, V = correct_and_filter_distances(  # noqa E741
            vertices[vinds_in_neigh], sources_in_curr, I, J, rho=rho_real
        )
    else:
        V = dists_curr[(I, J)]

    I_global = I
    J_global = vinds_in_neigh[J]

    return I_global, J_global, V


def in_cluster_dijkstra(
    kmeans_labels,
    vertices,
    subsample,
    radius_tree,
    graph,
    rho_dijkstra,
    class_index,
    correct_dist=True,
    rho_real=None,
):
    """
    Compute local dijkstra in a cluster.

    Takes as input the KMeans labels, the vertices, the subsample, and the label to consider.

    Parameters:
    ----------------------
    kmeans_labels : np.ndarray
        (m,) labels of the KMeans clustering
    vertices : np.ndarray
        (n,3) coordinates of all vertices
    subsample : np.ndarray
        (m,) indices of the sample points
    radius_tree : sklearn.neighbors.KDTree or sklearn.neighbors.BallTree
        Tree on the complete set of vertices, had a `query_radius` method
    graph : scipy.sparse.csc_matrix
        adjacency matrix of the mesh, with edge lengths
    rho_dijkstra : float
        radius for local dijkstra
    class_index : int
        index of the cluster to consider
    correct_dist : bool
        whether to correct the distance with L2 a posteriori (instead of dijkstra)
    rho_real : float
        real radius for the correction.

    Output
    ----------------------
    I : np.ndarray
        (m',) rows in the (m,n) distance matrix
    J : np.ndarray
        (m',) columns ion the (m,n) distance matrix
    V : np.ndarray
        (m',) values of the distance

    """
    if correct_dist and (rho_real is None):
        raise ValueError("Can only correct distances if real radius is given")

    # Step 1: Identify samples in the given cluster
    subsample_curr, sinds_curr = get_samples_in_cluster(
        kmeans_labels, subsample, class_index, return_indices=True
    )  # (m_curr,), (m_curr)

    # Step 2: Early exit if the cluster is empty
    if sinds_curr.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    # Step 3: Compute local dijkstra on local sample
    I, J_global, V = compute_local_dijkstra_with_tree(  # noqa E741
        graph,
        vertices,
        subsample_curr,
        rho_dijkstra,
        radius_tree=radius_tree,
        rho_real=rho_real,
        correct_dist=correct_dist,
    )

    # Step 4: Go back to global sample index.
    I_global = sinds_curr[I]
    return I_global, J_global, V


def compute_sparse_dijkstra(
    mesh1,
    subsample,
    rho_dijkstra,
    rho_real=None,
    n_clusters=4,
    correct_dist=True,
    edge_length_graph=None,
    radius_tree=None,
    verbose=False,
):
    """
    Given a mesh a set of samples, compute the local distance matrix between samples and vertices using dijkstra algorithm with fixed radius.

    This only outputs the new entries of the distance matrix, not the full matrix.

    Internally, to handle large meshes, the shape is divided in n_clusters parts using KMeans clustering on the samples.
    The clusters are then extended to the vertices, but note that a vertex can be in multiple clusters.

    Parameters
    ----------------------
    mesh1 : pyFM.TriMesh
        TriMesh object. Source mesh
    subsample : np.ndarray
        (p,) indices of the current sample
    rho_dijkstra : float
        radius for the dijkstra run
    rho_real : float
        real radius for the correction.  Since the dijkstra distance is bigger than the euclidean distance, the dijkstra_rho should be bigger than the real radius.
    n_clusters : int
        number of clusters to divide the shape in (memory issues). This is run on the samples then extended to vertices.
    correct_dist : bool
        whether to correct the distance with L2 a posteriori (instead of dijkstra)
    edge_length_graph : scipy.sparse.csc_matrix
        adjacency matrix of the mesh, with edge lengths
    radius_tree : sklearn.neighbors.KDTree or sklearn.neighbors.BallTree
        Tree on the complete set of vertices, had a `query_radius` method
    verbose : bool
        whether to display progress

    Output
    ----------------------
    I,J,V : np.ndarray
        new entries of the distance matrix
    unseen_inds : np.ndarray
        indices of unseen vertices by the dijkstra
    """
    if n_clusters > subsample.size:
        print(
            "Warning: n_clusters > subsample.size, setting n_clusters = subsample.size"
        )
        n_clusters = subsample.size
    kmeans = KMeans(n_clusters=n_clusters, max_iter=10).fit(
        mesh1.vertlist[subsample].copy()
    )
    # kmeans_labels = np.random.choice(n_clusters, size=subsample.shape[0], replace=True)
    if edge_length_graph is None:
        graph = build_edge_length_graph(mesh1.vertlist, mesh1.edges)
    else:
        graph = edge_length_graph

    # Build Tree for radius search
    if radius_tree is None:
        tree = KDTree(mesh1.vertlist, leaf_size=20)
    else:
        tree = radius_tree

    seen = np.full(mesh1.n_vertices, False)
    In = np.array([], dtype=int)
    Jn = np.array([], dtype=int)
    Vn = np.array([], dtype=float)

    for class_index in my_tqrange(n_clusters, verbose=verbose):

        I, J, V = in_cluster_dijkstra(  # noqa E741
            kmeans.labels_,
            mesh1.vertlist,
            subsample,
            tree,
            graph,
            rho_dijkstra,
            class_index,
            correct_dist=True,
            rho_real=rho_real,
        )

        seen[J] = True

        In = np.concatenate([In, I])
        Jn = np.concatenate([Jn, J])
        Vn = np.concatenate([Vn, V])

    unseen_inds = np.where(~seen)[0]

    return In, Jn, Vn, unseen_inds
