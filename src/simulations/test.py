import numpy as np
import matplotlib.pyplot as plt 
Time = 200
N=10
all_means = np.load("data/all_means_N10_M5_homo_bot.npy")

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
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

plt.title(
    rf"Average ergodic sum: $\beta = 0.2,\ \eta_B = 0.8$"
)

plt.savefig(
    "figures/all_means_N10_M5_homo_bot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
