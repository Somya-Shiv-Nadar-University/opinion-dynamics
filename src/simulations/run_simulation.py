from src.models.od_model import *
from src.plotting.network_plot import draw_network

# ==================================================
# Parameters
# ==================================================

N = 10
L = 10
M = 5
T = 10
Time = 100

# ==================================================
# Generate network
# ==================================================

def generate_network(
    N,
    p,
    weight_min=0.1,
    weight_max=1.0,
    directed=True,
    seed=None,
):
    """
    Generate weighted row-stochastic network.
    """

    rng = np.random.default_rng(seed)

    G = nx.gnp_random_graph(
        N,
        p,
        directed=directed,
        seed=seed,
    )

    G = nx.relabel_nodes(
        G,
        {i: i + 1 for i in range(N)}
    )

    for u, v in G.edges():

        G[u][v]["weight"] = rng.uniform(
            weight_min,
            weight_max,
        )

    A = nx.to_numpy_array(
        G,
        weight="weight",
    )

    row_sums = A.sum(axis=1, keepdims=True)

    A = np.divide(
        A,
        row_sums,
        where=row_sums != 0,
    )

    return G, A


# ==================================================
# Generate model ingredients
# ==================================================

c1, c2 = generate_c1_c2(N, M)

G, A = generate_network(
    N=N,
    p=0.5,
    seed=42
)

memory_sets = generate_memory_sets(
    N=N,
    L=L,
    M=M
)

# ==================================================
# Draw network
# ==================================================

draw_network(G)

# ==================================================
# Simulate empirical sums 
# ==================================================

nodewise_means = []

rng = np.random.default_rng(42)

for horizon in range(3, Time + 3):

    monte_carlo_vals = []

    for sim in range(100):

        X_history, Z_history = simulate_trajectory(
            N=N,
            horizon=horizon,
            T=T,
            L=L,
            A=A,
            c1=c1,
            c2=c2,
            memory_sets=memory_sets,
            rng=rng,
        )

        # shape: (N,)
        ergodic_vals = np.mean(
            Z_history,
            axis=0,
        )

        # store full nodewise vector
        monte_carlo_vals.append(
            ergodic_vals
        )

    # average over simulations
    mean_vals = np.mean(
        monte_carlo_vals,
        axis=0,
    )

    nodewise_means.append(mean_vals)

# ==================================================
# Plotting
# ==================================================

nodewise_means = np.array(nodewise_means)

time_axis = np.arange(3, Time + 3)

plt.figure(figsize=(10,6))

for node in range(N):

    plt.plot(
        time_axis,
        nodewise_means[:, node],
        label=f"Node {node+1}",
    )

plt.xlabel("Time ")

plt.ylabel("Empirical averages")

plt.title(
    "Nodewise empirical averages for a 10-node non-homogeneous network"
)

plt.grid(True)

plt.legend()

plt.show()