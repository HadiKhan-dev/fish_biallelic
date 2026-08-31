"""Optional plotting for pedigree relationship tables."""

import pandas as pd


def draw_pedigree_tree(relationships_df, output_file="pedigree_tree.png"):
    """Draw scientific parent fields without importing a pedigree engine."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        return None

    has_first = "ScientificParent1" in relationships_df.columns
    has_second = "ScientificParent2" in relationships_df.columns
    if has_first != has_second:
        raise ValueError(
            "relationship tables need both ScientificParent columns or neither"
        )
    frame = relationships_df
    if has_first:
        frame = relationships_df.copy(deep=True)
        frame["Parent1"] = frame["ScientificParent1"]
        frame["Parent2"] = frame["ScientificParent2"]

    graph = nx.DiGraph()
    generation_nodes = {"F1": [], "F2": [], "F3": [], "Unknown": []}
    parents_of = {}
    for _, row in frame.iterrows():
        sample = row["Sample"]
        generation = row["Generation"]
        generation_nodes.setdefault(generation, []).append(sample)
        color = "#999999"
        if generation == "F1":
            color = "#1f77b4"
        elif generation == "F2":
            color = "#ff7f0e"
        elif generation == "F3":
            color = "#2ca02c"
        graph.add_node(sample, color=color, gen=generation)
        if pd.notna(row["Parent1"]):
            graph.add_edge(row["Parent1"], sample)
            parents_of.setdefault(sample, []).append(row["Parent1"])
        if pd.notna(row["Parent2"]):
            graph.add_edge(row["Parent2"], sample)
            parents_of.setdefault(sample, []).append(row["Parent2"])

    positions = {}
    node_y = {}
    layers = sorted(
        [key for key in generation_nodes if key.startswith("F")],
        key=lambda value: int(value[1:]),
    )
    if "Unknown" in generation_nodes:
        layers.append("Unknown")
    for x_index, generation in enumerate(layers):
        nodes = generation_nodes[generation]
        if not nodes:
            continue
        if x_index == 0:
            nodes.sort()
        else:
            nodes.sort(
                key=lambda node: (
                    sum(node_y.get(parent, 0.5) for parent in parents_of[node])
                    / len(parents_of[node])
                    if parents_of.get(node)
                    else 0.5
                ),
                reverse=True,
            )
        for index, node in enumerate(nodes):
            y_position = 1.0 - (index + 0.5) / len(nodes)
            positions[node] = (x_index, y_position)
            node_y[node] = y_position

    plt.figure(
        figsize=(20, max(10, len(generation_nodes.get("F3", [])) * 0.2))
    )
    node_colors = [graph.nodes[node]["color"] for node in graph.nodes()]
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=80,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.5,
    )
    nx.draw_networkx_edges(
        graph, positions, edge_color="gray", alpha=0.3, width=0.5, arrows=False
    )
    labelled = generation_nodes.get("F1", []) + generation_nodes.get("F2", [])
    nx.draw_networkx_labels(
        graph, positions, labels={node: node for node in labelled}, font_size=8
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    return None
