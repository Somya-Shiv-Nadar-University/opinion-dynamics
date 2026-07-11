import numpy as np
import matplotlib.pyplot as plt 
Time = 200
N=10
all_means = np.load("data/all_means_N10_M5_homo_hub_spoke_bot_.npy")

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
    loc="lower right",
    fontsize=10,
    frameon=True
)

plt.tight_layout()

plt.xlabel("Time",fontsize=15)
plt.ylabel("Average ergodic sum",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

plt.title(
    rf"Average ergodic sum: $\beta = 0.2,\ \eta_B = 0.05$",fontsize=15

)

plt.savefig(
    "figures/all_means_N10_M5_homo_hub_spoke_bot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
