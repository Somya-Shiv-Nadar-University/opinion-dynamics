from src.models.od_model import generate_network
from src.plotting.network_plot import draw_network

G, A = generate_network(
    N=10,
    p=0.5,
    seed=42
)

draw_network(G)

print(A)
