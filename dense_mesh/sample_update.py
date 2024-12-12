import numpy as np

from sklearn.neighbors import KDTree

from .local_dijkstra import compute_local_dijkstra_with_tree


def select_distant_vertices(available_inds, vertices, rho, return_vert_inds=True):
    """
    Given a set of vertices, return a subset of vertices which are at a (euclidean) distance bigger than rho.

    Since dijktra distance is larger than euclidean distance, this can be used to select a subset of vertices
    which will not interact in the dense mesh algorithm.


    This has no guarentee of selecting the maximum number of vertices, but is a simple heuristic.

    Parameters
    ----------------------
    available_inds : np.ndarray
        indices of vertices to consider
    vertices : np.ndarray
        (n,3) coordinates of vertices
    rho : float
        radius
    return_vert_inds : bool
        whether to return the indices of the vertices instead of indices in the input array (default is False)

    Output
    ----------------------
    selection : np.ndarray
        (p,) indices of selected vertices.
    """

    assert available_inds.ndim == 1, "Input should be a list of indices"

    vertices_set = vertices[available_inds]  # (p,3)

    # Select a first random source
    rng = np.random.default_rng()
    inds = [rng.integers(available_inds.size)]

    # Iteratively add unseen vertices who lie at an euclidean distance bigger than rho to the current selected vertices

    dists = np.linalg.norm(
        vertices_set[inds[0]][None, :] - vertices_set, axis=1
    )  # (p,)
    while np.any(dists > rho):
        newid = np.argmax(dists)
        inds.append(newid)

        new_distance = np.linalg.norm(
            vertices_set[newid][None, :] - vertices_set, axis=1
        )

        dists = np.minimum(dists, new_distance)

    if return_vert_inds:
        return available_inds[inds]

    return inds


def update_sampled_points(
    graph,
    subsample,
    unseen_inds,
    rho_dijkstra,
    I,  # noqa E741
    J,
    V,
    vertices,
    correct_dist=True,
    radius_tree=None,
    rho_real=None,
    directed=True,
    fast_update=True,
    verbose=False,
):
    """
    Add some unseen vertices to the sample set and compute the local distances.

    Parameters:
    ----------------------
    graph : scipy.sparse.csc_matrix
        (n,n) sparse matrix of the mesh graph (edge lengths)
    subsample : np.ndarray
        (m,) indices of the current samples
    unseen_inds : np.ndarray
        (u,) indices of unseen vertices
    rho_dijkstra : float
        radius for the dijkstra run
    I : np.ndarray
        Current rows in the (m,n) distance matrix
    J : np.ndarray
        Current columns in the (m,n) distance matrix
    V : np.ndarray
        Current values of the distance
    vertices : np.ndarray
        (n,3) coordinates of ALL vertices
    correct_dist : bool
        whether to correct the distance with L2 a posteriori (instead of dijkstra)
    radius_tree : sklearn.neighbors.KDTree or sklearn.neighbors.BallTree
        Tree on the complete set of vertices, had a `query_radius` method
    rho_real : float
        real radius for the correction.  Since the dijkstra distance is bigger than the euclidean distance, the dijkstra_rho should be bigger than the real radius.
    directed : bool
        whether the graph is directed
    fast_update : bool
        whether to use the fast update heuristic
    verbose : bool
        whether to display progress

    Output
    ----------------------
    I,J,V : np.ndarray
        new entries of the (m',n) distance matrix
    subsample : np.ndarray
        (m',) indices of the new sample
    """
    if radius_tree is None:
        radius_tree = KDTree(vertices, leaf_size=20)

    n_samples = subsample.size
    n_samples_init = n_samples

    if verbose and len(unseen_inds) > 0:
        print(f"{len(unseen_inds)} unseen vertices")
    # Loop while some vertices are unseen
    all_seen = len(unseen_inds) == 0
    iteration = 0
    while not all_seen:
        iteration += 1

        # if verbose:
        #     print(f"\tIteration {iteration} : " f"{unseen_inds.size} unseen vertices:")

        # Select one or multiple new sources
        if fast_update:
            rho_selection = rho_dijkstra if not correct_dist else rho_real
            new_inds = select_distant_vertices(unseen_inds, vertices, rho_selection)
        else:
            new_inds = [unseen_inds[0]]

        if verbose:
            print(
                f"Iteration {1+iteration}: Adding {len(new_inds)} samples"
                f" ({n_samples+len(new_inds) - n_samples_init} added in total)"
            )

        # Add vertices to the sample
        subsample = np.append(subsample, new_inds)

        # Compute values from new source
        I_new, J_new, V_new = compute_local_dijkstra_with_tree(
            graph,
            vertices,
            new_inds,
            rho_dijkstra,
            radius_tree=radius_tree,
            rho_real=rho_real,
            correct_dist=correct_dist,
        )
        # Add the values to the (I,J,V) COO entries
        I = np.concatenate([I, I_new + n_samples])  # noqa E741
        J = np.concatenate([J, J_new])
        V = np.concatenate([V, V_new])

        n_samples += len(new_inds)

        # Update unseen vertices
        # unseen_inds = np.array([x for x in unseen_inds if x not in J_new])
        unseen_inds = np.setdiff1d(unseen_inds, J_new)
        all_seen = len(unseen_inds) == 0

    return I, J, V, subsample
