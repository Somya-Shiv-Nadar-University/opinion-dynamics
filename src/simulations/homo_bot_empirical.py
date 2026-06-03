from src.models.od_model import *
from src.plotting.network_plot import *

# ==================================================
# Parameters
# ==================================================

N = 10
L = 10
M = 5
T = 10
Time = 20

# ==================================================
# Generate model ingredients
# ==================================================

c1 = 0.05 * np.ones(N)
c2 = 0.1 * np.ones(N)

A = np.load("data/A_N10_M5.npy")
memory_sets = generate_memory_sets(
    N=N,
    L=L,
    M=M
)


# ==================================================
# Simulate empirical sums for all agents
# ==================================================

all_means = np.zeros((Time,N))

rng = np.random.default_rng(42)

for horizon in range(3, Time + 3):

    monte_carlo_vals = []

    for sim in range(100):

        X_history, Z_history = simulate_trajectory_bot(
            N=N,
            horizon=horizon,
            T=T,
            L=L,
            A=A,
            c1=c1,
            c2=c2,
            memory_sets=memory_sets,
            beta =0.20,
            eta = 0.8,
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
    "data/all_means_N10_M5_homo_bot.npy",all_means
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

plt.axhline(
    y=0.1,
    color="black",
    linestyle="--",
    linewidth=2,
    label=r"$p^{(*)}=0.1$"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0.
)
plt.tight_layout()

plt.xlabel("Time")
plt.ylabel("Average ergodic sum")
plt.title(
    rf"Average ergodic sum: $\beta = 0.2,\ \eta_B = 0.8$"
)

plt.savefig(
    "figures/all_means_N10_M5_homo_bot.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()