import networkx as nx
import matplotlib.pyplot as plt

def draw_network(G, save_path=None):

    plt.figure(figsize=(6,6))

    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=600,
        node_color="lightgray"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=15,
        edge_color="black",
        width=1.3,
        connectionstyle="arc3,rad=0.15"
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10
    )

    plt.axis("off")

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    else:

        plt.show()