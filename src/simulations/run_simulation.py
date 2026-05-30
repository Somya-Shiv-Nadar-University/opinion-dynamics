from src.models.od_model import *
from src.plotting.network_plot import *

# ==================================================
# Parameters
# ==================================================

N = 10
L = 10
M = 2
T = 10
Time = 200

# ==================================================
# Generate model ingredients
# ==================================================

c1 = np.load("data/c1_N10_M5.npy")
c2 = np.load("data/c2_N10_M5.npy")
memory_sets = generate_memory_sets(
    N=N,
    L=L,
    M=M
)

# ==================================================
# Draw network
# ==================================================
A = np.load("data/A_N10_M5.npy")
# ==================================================
# Simulate empirical sums for all agents
# ==================================================

all_means = np.zeros((Time,N))

rng = np.random.default_rng(42)

for horizon in range(3, Time + 3):

    monte_carlo_vals = []

    for sim in range(5000):

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

        ergodic_vals = np.mean(Z_history, axis=0)

        monte_carlo_vals.append(
            ergodic_vals
        )

    all_means[horizon - 3] = np.mean(
    monte_carlo_vals,
    axis=0
)
#save the empirical sum data
np.save(
    "data/all_means_N10_M2.npy",all_means
)

# ==================================================
# Plotting
# ==================================================

time = np.arange(3, Time + 3)

plt.figure(figsize=(10, 4.5))

for node in range(N):

    plt.plot(
        time,
        all_means[:, node],
        linewidth=2,
        label=f"Agent {node+1}"
    )

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0.
)
plt.tight_layout()

plt.xlabel("Time")
plt.ylabel("Average empirical sum")

plt.savefig(
    "figures/all_means_N10_M2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()