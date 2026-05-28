#import packages
import networkx as nx 
import numpy as np
import random
import matplotlib.pyplot as plt 

#ensure that probabilities stay below one
def generate_dependent_vector(x, Threshold,eps = 1e-12):
    # upper bounds for each coordinate
    upper = np.minimum(1.0, 1.0 - Threshold * x)

    # safety check (important)
    if np.any(upper <= 0):
        raise ValueError("Some entries have non-positive upper bound. Check M and x.")

    # sample uniformly in (0, upper_i)
    y = upper * np.random.rand(len(x))

    # optional: keep strictly away from 0
    y = np.maximum(y, eps)

    return y

#generate c1 and c2 vectors for the model
def generate_c1_c2(N, M, eps=1e-12):
    c2 = (1/M - eps) * np.random.rand(N)
    c1 = generate_dependent_vector(c2, M)
    return c1, c2

#generate directed graph

def generate_network(
    N,
    p,
    weight_min=0.1,
    weight_max=1.0,
    directed=True,
    seed=None,
):
    """
    Generate a random weighted directed graph and its
    row-stochastic adjacency matrix.

    Parameters
    ----------
    N : int
        Number of nodes.

    p : float
        Edge probability.

    weight_min : float
        Minimum edge weight.

    weight_max : float
        Maximum edge weight.

    directed : bool
        Whether graph is directed.

    seed : int or None
        Random seed.

    Returns
    -------
    G : networkx graph
        Generated graph.

    A : ndarray
        Row-stochastic adjacency matrix.
    """

    rng = np.random.default_rng(seed)

    # generate graph
    G = nx.gnp_random_graph(
        N,
        p,
        directed=directed,
        seed=seed
    )

    # relabel nodes from 1 to N
    G = nx.relabel_nodes(G, {i: i + 1 for i in range(N)})

    # assign random weights
    for u, v in G.edges():
        G[u][v]["weight"] = rng.uniform(weight_min, weight_max)

    # adjacency matrix
    A = nx.to_numpy_array(G, weight="weight")

    # normalize rows
    row_sums = A.sum(axis=1, keepdims=True)

    A = np.divide(
        A,
        row_sums,
        out=np.zeros_like(A),
        where=row_sums != 0
    )

    return G, A

#memory sets
# each element of the memory set is an N\times N indicator matrix
# there are L (memory depth) elements in this set.
def generate_memory_sets(N, L, M):
    """
    Generate L binary N x N matrices such that
    sum over l of entry (i,j) equals M for all i,j.
    """
    if M > L:
        raise ValueError("M must be <= L")

    rng = np.random.default_rng()

    # initialize L matrices
    memory_sets = [np.zeros((N, N), dtype=int) for _ in range(L)]

    # for each (i,j), choose M distinct layers to place a 1
    for i in range(N):
        for j in range(N):
            layers = rng.choice(L, size=M, replace=False)
            for ell in layers:
                memory_sets[ell][i, j] = 1

    return memory_sets



def simulate_trajectory(
    N,
    horizon,
    T,
    L,
    A,
    c1,
    c2,
    memory_sets,
    rng=None,
):
    """
    Simulate one trajectory of the opinion dynamics model.

    Parameters
    ----------
    N : int
        Number of nodes.

    horizon : int
        Final simulation time.

    T : int
        Time at which memory activates.

    L : int
        Memory depth.

    A : ndarray
        Row-stochastic interaction matrix.

    c1 : ndarray
        Drift vector.

    c2 : ndarray
        Scaling vector.

    memory_sets : list
        List of memory indicator matrices.

    rng : numpy random generator
        Random number generator.

    Returns
    -------
    X_history : list
        bias matrices over time.

    Z_history : list
        expressed opinion vectors over time.
    """

    if rng is None:
        rng = np.random.default_rng()

    # ==================================================
    # Initial state
    # ==================================================

    X_t = np.tile(c1.reshape(N, 1), (1, N))

    Z_t = np.zeros(N)

    X_history = [X_t.copy()]
    Z_history = [Z_t.copy()]

    # ==================================================
    # Time evolution
    # ==================================================

    for t in range(1, horizon):

        # ----------------------------------------------
        # Generate activation vector
        # ----------------------------------------------

        interaction = A @ X_t

        activation_prob = interaction[:, 0]

        random_vals = rng.random(N)

        Z_next = (random_vals < activation_prob).astype(float)

        # ----------------------------------------------
        # Pre-memory dynamics
        # ----------------------------------------------

        if t < T:

            X_next = X_t + np.outer(c2, Z_next)

        # ----------------------------------------------
        # Memory dynamics
        # ----------------------------------------------

        else:

            X_next = np.zeros((N, N))

            for i in range(N):

                for j in range(N):

                    memory_sum = 0.0

                    for ell in range(L):

                        memory_sum += (
                            memory_sets[ell][i, j]
                            * Z_history[-(ell + 1)][j]
                        )

                    X_next[i, j] = (
                        c1[i]
                        + c2[i] * memory_sum
                    )

        # ----------------------------------------------
        # Update state
        # ----------------------------------------------

        X_t = X_next
        Z_t = Z_next

        X_history.append(X_t.copy())
        Z_history.append(Z_t.copy())

    return X_history, Z_history

