import numpy as np
import matplotlib.pyplot as plt

# load data
all_means = np.load(
    "data/all_means_N10_M5_nonhomo_bot.npy"
)

# time axis
time = np.arange(3, all_means.shape[0] + 3)

# plot
plt.figure(figsize=(10, 4.5))

for node in range(all_means.shape[1]):

    plt.plot(
        time,
        all_means[:, node],
        linewidth=2,
        label=f"Agent {node+1}"
    )

plt.xlabel("Time")
plt.ylabel("Average empirical sum")

plt.title(
    r"Non-homogeneous model with bot: $\beta=0.2,\ \eta_B=0.8$"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.savefig(
    "figures/all_means_N10_M5_nonhomo_bot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()