from src.models.od_model import *

# ==================================================
# Parameters
# ==================================================

N = 10
L = 10
M = 5
T = 10
Time = 100

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
# Monte Carlo experiment
# ==================================================

all_means = []

rng = np.random.default_rng(42)

for horizon in range(3, Time + 3):

    monte_carlo_vals = []

    for sim in range(500):

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
            np.mean(ergodic_vals)
        )

    all_means.append(
        np.mean(monte_carlo_vals)
    )

# ==================================================
# Plot
# ==================================================

time = np.arange(3, Time + 3)

plt.figure(figsize=(10, 4.5))

plt.plot(time, all_means)

plt.xlabel("Time")
plt.ylabel("Empirical ergodic average")

plt.show()