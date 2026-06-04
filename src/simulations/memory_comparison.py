import numpy as np
import matplotlib.pyplot as plt 

all_means_M2 = np.load("data/all_means_N10_M2.npy")
all_means_M3 = np.load("data/all_means_N10_M3.npy")
all_means_M5 = np.load("data/all_means_N10_M5.npy")
time = np.arange(3, all_means_M2.shape[0] + 3)

colors = ["C0", "C1", "C2"]

plt.figure(figsize=(10,5))

for agent in range(3):

    plt.plot(
        time,
        all_means_M2[:, agent],
        color=colors[agent],
        linestyle=":",
        linewidth=2,
        label=f"Agent {agent+1}, M=2"
    )

    plt.plot(
        time,
        all_means_M3[:, agent],
        color=colors[agent],
        linestyle="--",
        linewidth=2,
        label=f"Agent {agent+1}, M=3"
    )

    plt.plot(
        time,
        all_means_M5[:, agent],
        color=colors[agent],
        linestyle="-",
        linewidth=2,
        label=f"Agent {agent+1}, M=5"
    )

plt.xlabel("Time")
plt.ylabel("Average empirical sum")

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.savefig(
    "figures/memory_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()