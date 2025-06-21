# plot_reference_graph.py

import matplotlib.pyplot as plt
import networkx as nx

# 论文信息
papers = {
    "P1": "Large Language Models for Text Summarization",
    "P2": "A Survey on Retrieval-Augmented Generation",
    "P3": "Knowledge Graph Construction from Scholarly Papers",
    "P4": "Citation Network Analysis in AI Research",
    "P5": "Semantic Similarity in Scientific Literature"
}

# 引用关系（有向图：被引用 <- 引用者）
reference_graph = {
    "P1": ["P2", "P3"],  # P1 引用了 P2 和 P3
    "P2": ["P4"],  # P2 引用了 P4
    "P3": ["P4", "P5"],  # P3 引用了 P4 和 P5
    "P4": [],  # P4 没有引用其他论文
    "P5": ["P2"]  # P5 引用了 P2
}


def plot_reference_graph(reference_graph):
    G = nx.DiGraph()
    for src, targets in reference_graph.items():
        for tgt in targets:
            G.add_edge(src, tgt)
    nx.draw(G, with_labels=True, node_size=1000, font_size=10)
    plt.show()


# 示例
plot_reference_graph(reference_graph)
