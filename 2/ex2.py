import numpy as np
import networkx as nx
import pickle as pkl
import matplotlib.pyplot as plt

from ex1 import power_eigen_values


def checks(G):
    # Calculate eigenvalues
    dom_eig_vec = power_eigen_values(G, 1e-9, 1000)
    dom_eig_val = (dom_eig_vec @ G @ dom_eig_vec.T).item()
    eigvals = np.sort(np.linalg.eigvalsh(G))

    # Properties check
    min_eig_val, max_eig_val = eigvals[0], eigvals[-1]
    distinct = 1 + np.sum(np.diff(eigvals) > 1e-9)

    nxg = nx.from_numpy_array(G)
    n = nxg.number_of_nodes()

    # print(min_eig_val, max_eig_val, eigvals)
    print(f"Complet (eigv): {distinct == 2}")
    print(f"Complet (from networkx): {nxg.number_of_edges() == n*(n - 1) // 2}")
    print(f"Bipartit (eigv): {np.isclose(min_eig_val, -max_eig_val)}")
    print(f"Bipartit (from networkx): {nx.is_bipartite(nxg)}")
    print(f"Cliche size (eigv) - upper bound: {np.round(max_eig_val) + 1}")
    print(f"Cliche size (from networkx): {len(nx.algorithms.approximation.clique.max_clique(nxg))}")


if __name__ == "__main__":
    with open("data/grafuri.pickle", "rb") as f:
        graphs = pkl.load(f)
        n_graphs = len(graphs)

        rows, cols = 3, 1
        fig, axes = plt.subplots(
            rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False
        )
        axes = axes.flatten()

        for i, graph in enumerate(graphs):
            nxgraph = nx.from_numpy_array(graph)
            nx.draw(nxgraph, with_labels=True, ax=axes[i])
            axes[i].set_title(f"Graph {i}")

            print(f"------ Graph {i} ------")
            checks(graph)
            print()

        plt.tight_layout()
        plt.savefig("output/graphs.svg")
        plt.show()
