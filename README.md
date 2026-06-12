# Opinion Dynamics with Memory Loss and Communication delay

This repository contains code for the simulation results in paper #insert link# .

## Overview

We study a network of interacting agents, each of whom expresses a binary opinion at every time step. The probability of expressing an opinion depends on the agent's current biases, which evolves over time according to:

1. interactions with neighboring agents,
2. memory of previously expressed opinions,
3. optional external influence through a bot.

We plot the ergodic sum of expressed opinions for all agents, averaged over 5000 experiments.

---
## Repository Structure

```text
opinion-dynamics/

├── data/                  # saved simulation data
├── figures/               # generated figures
├── experiments/           # experiment descriptions and notes  #write this#
├── src/
│   ├── models/
│   │   └── od_model.py    # model and simulation functions
│   ├── plotting/
│   │   └── network_plot.py
│   ├── simulations/
│   │   ├── run_simulation.py
│   │   ├── non_homo_empirical.py
│   │   └── non_homo_bot_empirical.py
│   └── utils/
│
├── requirements.txt
└── README.md
```

---

## Model Components

### Network

A directed weighted graph is generated using NetworkX.

The weighted adjacency matrix is row-stochastic and determines the interaction weights among agents.

### Memory sets

Each agent maintains a collection of memory sets specifying which previously expressed opinions (its own as well as other agents) influence its current biases.
 


### Model Parameters

T : Memory activation time. Before T, each agents updates

Each agent possesses parameters

[
c_1,\qquad c_2,
]

which determine how past opinions affect future activation probabilities.

Both homogeneous and non-homogeneous parameter settings are supported.

### Bot Influence

The model optionally includes a bot characterized by:

* bot weight (\beta),
* bot strength (\eta_B).

The bot modifies the opinion activation probability while keeping the underlying network structure fixed.

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Simulations

### Homogeneous Model

```bash
python -m src.simulations.run_simulation
```

### Non-Homogeneous Model

```bash
python -m src.simulations.non_homo_empirical
```

### Non-Homogeneous Model with Bot

```bash
python -m src.simulations.non_homo_bot_empirical
```

---

## Output

Simulation outputs are saved in the `data/` directory.

Typical outputs include:

* weighted adjacency matrices,
* model parameters,
* empirical ergodic averages.

Figures are saved in the `figures/` directory.

---

## Example Quantities of Interest

The code can be used to study:

* empirical ergodic averages,
* convergence to fixed points,
* influence of memory size,
* effect of network topology,
* effect of external influence,
* heterogeneity across agents.

---

## Author

Somya Singh

Department of Mathematics

Shiv Nadar Univeristy, India


