import numpy as np
import scipy.sparse as sparse

from .utils import local_f, get_data_inds_of_columns_inds_csc


def initialize_Umat(distmat, rho, interpolation="poly"):
    """
    Generate the unnormalized matrix of local functions

    Parameters:
    ----------------------
    distmat : scipy.sparse.csc_matrix
        (n,p) sparse matrix of distances
    rho : float
        radius
    interpolation : str
        'poly', 'linear', 'exp' - type of local function

    Output
    ----------------------
    U : scipy.sparse.csr_matrix
        (n,p) sparse matrix of (unnormalized) local functions
    U_data : np.ndarray
        (L,) values of the local functions (for csc format)
    dist_data : np.ndarray
        (L,) values of the distances (for csc format)
    """
    U_data = local_f(distmat.data, rho, chi=interpolation)  # (L,)

    dist_data = distmat.data.copy()  # (L,)
    U = sparse.csc_matrix(
        (U_data, distmat.indices, distmat.indptr), shape=distmat.shape
    ).tocsr()
    return U, U_data, dist_data


def identify_samples_to_adapt(U, subsample, max_val, verbose=False):
    """
    Identify samples to adapt in the local functions.
    These are determined so that the sum of the values of the (unnormalized) local functions is greater than max_val at this point.

    Parameters
    ----------------------
    U : scipy.sparse.csr_matrix
        (n,p) sparse matrix of unnormalized local functions
    subsample : np.ndarray
        (p,) indices of the current sample
    max_val : float
        threshold for the sum of the local functions

    Output
    ----------------------
    subinds_toadapt : np.ndarray
        indices of the samples to adapt

    """
    sum_of_chi = np.asarray(U[subsample].sum(1)).squeeze()
    flagged_samples = np.where(sum_of_chi > max_val)[0]

    return flagged_samples


def compute_secondary_influence(U, subsample, flagged_samples):
    """
    Extract the sample point with the highest influence for each sample needing adaptation (excluding itself which is the biggest)

    We use the fact that the unnormalized local function is 1 at the sample point itself.

    Parameters
    ----------------------
    U : scipy.sparse.csr_matrix
        (n,p) sparse matrix of unnormalized local functions
    subsample : np.ndarray
        (p,) indices of the sample in the complete shape
    flagged_samples : np.ndarray
        (m,) indices of the samples to adapt

    Output
    ----------------------
    wide_samples : np.ndarray
        indices of the secondary samples to adapt
    """
    M = flagged_samples.size

    self_weight_mat = sparse.csr_matrix(
        (np.ones(M, dtype=float), (np.arange(M), flagged_samples)),
        shape=(M, subsample.size),
    )
    red_mat = U[subsample[flagged_samples]] - self_weight_mat

    return np.unique(np.asarray(red_mat.argmax(axis=1)).flatten())


def update_radius_and_recompute(
    distmat, U, U_data_csc, dist_data_csc, wide_samples, rho_curr, interpolation="poly"
):
    """
    Reduce the radius of given samples and recompute values in `U` for affected indices.

    Parameters
    ----------------------
    distmat : scipy.sparse.csc_matrix
        (n,p) sparse matrix of distances (from initial dijkstra)
    U : scipy.sparse.csr_matrix
        (n,p) sparse matrix of unnormalized local functions
    U_data_csc : np.ndarray
        (L,) values of the local functions (FOR CSC FORMAT)
    dist_data_csc : np.ndarray
        (L',) values of the distances (FOR CSC FORMAT)
    wide_samples : np.ndarray
        (p') indices of the samples to adapt
    rho_curr : float
        current radius to apply to the wide samples
    interpolation : str
        'poly', 'linear', 'exp' - type of local function

    Output
    ----------------------
    U : scipy.sparse.csr_matrix
        (n,p) sparse matrix of updated unnormalized local functions
    U_data : np.ndarray
        (L,) values of U.data (FOR CSC FORMAT)
    distance_data : np.ndarray
        (L',) values of the distances (FOR CSC FORMAT)
    """

    # Find the indices where the values of distances for the local functions to consider are stored
    data_indices = get_data_inds_of_columns_inds_csc(distmat.indptr, wide_samples)

    # Recompute local function values for modified radius
    new_local_f, new_zeros = local_f(
        distmat.data[data_indices],
        rho_curr,
        chi=interpolation,
        return_zeros_indices=True,
    )

    # Update local function (potentially with new zeros)
    U_data_csc[data_indices] = new_local_f
    # Set new zeros to distance data (will be remove by sparsity)
    dist_data_csc[data_indices[new_zeros]] = 0
    # print(dist_data_csc)

    # Rebuild U in CSR format
    U = sparse.csc_matrix(
        (U_data_csc, distmat.indices, distmat.indptr), shape=distmat.shape
    ).tocsr()

    return U, U_data_csc, dist_data_csc


def adapt_rho(
    distmat,
    subsample,
    rho,
    threshold=0.1,
    rho_decrease_ratio=2,
    interpolation="poly",
    verbose=False,
):
    """
    Modify radius locally.

    A sample j must see his local radius reduced if:
        1/(Sum_{e_ij} w_ij) < threshold        <->      Sum_{e_ij} w_ij > 1/threshold

    where w_ij are defined as chi(d_ij/rho) with d_ij the distance between i and j (computed with rectified dijkstra).

    Parameters:
    ----------------------
    distmat : scipy.sparse.csc_matrix
        (n,p) sparse matrix of distances
    subsample : np.ndarray
        (p,) indices of the current sample
    rho : float
        initial radius
    threshold : float
        threshold for the self weight
    rho_decrease_ratio : float
        ratio to decrease the radius between iterations
    interpolation : str
        'poly', 'linear', 'exp' - type of local function


    Output
    ----------------------
    U : scipy.sparse.csr_matrix
        (n,p) sparse matrix of updated unnormalized local functions
    dist_data_csc : np.ndarray
        (L,) new values in the distance matrix (FOR CSC FORMAT)
    per_vertex_rho : np.ndarray
        (p,) updated radius for each sample point
    """

    max_self_val = 1 / threshold

    U, U_data_csc, dist_data_csc = initialize_Umat(
        distmat, rho, interpolation=interpolation
    )  # (n,p), (L,) (L,)

    # Step 2: Identify initial indices that need radius adaptation
    flagged_samples = identify_samples_to_adapt(U, subsample, max_self_val)  # (m',)
    wide_samples = compute_secondary_influence(U, subsample, flagged_samples)  # (m'',)

    per_vertex_rho = np.full(subsample.size, rho)
    rho_curr = rho
    iteration = 0

    print(dist_data_csc)

    # While still indices to add (and limit radius to rho/2**7)
    while len(wide_samples) > 0 and iteration < 7:
        iteration += 1
        if verbose:
            print(
                f"Iteration {iteration} : "
                f"Modifying {wide_samples.size} sampled points"
            )

        # compute new rho
        rho_curr /= rho_decrease_ratio
        per_vertex_rho[wide_samples] = rho_curr

        U, U_data_csc, dist_data_csc = update_radius_and_recompute(
            distmat,
            U,
            U_data_csc,
            dist_data_csc,
            wide_samples,
            rho_curr,
            interpolation=interpolation,
        )

        # Identify new samples to adapt
        flagged_samples = identify_samples_to_adapt(U, subsample, max_self_val)
        wide_samples = compute_secondary_influence(U, subsample, flagged_samples)

    if verbose:
        nnz_before = U.nnz
    U.eliminate_zeros()
    if verbose:
        nnz_after = U.nnz
        print(f"Removed {nnz_before-nnz_after} values")

    return U, dist_data_csc, per_vertex_rho
