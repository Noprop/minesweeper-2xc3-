import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from typing import List

class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)

        
    def display_graph(self):
        # use the networkx directed graph
        G = nx.DiGraph()
        for node in list(self.adj.keys()):
            for neighbor in self.adj[node]:
                weight = self.w(node, neighbor)
                G.add_edge(node, neighbor, weight=weight)

        # use the spring layout, but increase the distance between the nodes using k
        # pos = nx.spring_layout(G, k=0.1, iterations=5)
        pos = nx.spring_layout(G, k=5/math.sqrt(self.number_of_nodes()), iterations=5)

        # create figure, edge labels, and draw
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, node_shape='o')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # display
        plt.show()

graph = DirectedWeightedGraph()
graph.add_node(0)
graph.add_node(1)
graph.add_node(2)
graph.add_node(3)
graph.add_node(4)
graph.add_node(5)
graph.add_node(6)

graph.add_edge(1, 0, 6)
graph.add_edge(1, 5, 2)
graph.add_edge(1, 3, 3)
graph.add_edge(1, 6, 4)
graph.add_edge(1, 4, 4)
graph.add_edge(2, 6, 7)
graph.add_edge(4, 0, 2)
graph.add_edge(4, 2, 5)
graph.add_edge(4, 5, 6)
graph.add_edge(5, 2, -6)
graph.add_edge(5, 6, 4)

graph.display_graph()

def init_d(G):
    n = G.number_of_nodes()
    d = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.are_connected(i, j):
                d[i][j] = G.w(i, j)
        d[i][i] = 0
    return d

#Assumes G represents its nodes as integers 0,1,...,(n-1)
def unknown(G):
    n = G.number_of_nodes()
    d = init_d(G)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]: 
                    d[i][j] = d[i][k] + d[k][j]
    return d


weights = init_d(graph)
for i in range(len(weights)):
    print("n" + str(i) + ": ", end='')
    print(weights[i])

print()
print()
weights2 = unknown(graph)
for i in range(len(weights2)):
    print("n" + str(i) + ": ", end='')
    print(weights2[i])


def graphGenerator(num_graphs: int, size: int, min_edges: int, max_edges: int) -> list[DirectedWeightedGraph]:
    graphs = []
    for _ in range(num_graphs):
        graph = DirectedWeightedGraph()
        
        # Initialize all nodes in the graph
        for node in range(size):
            graph.add_node(node)
        
        edges = random.randint(min_edges, max_edges)
        while edges > 0:
            node1 = random.randint(0, size-1)
            node2 = random.randint(0, size-1)
            weight = random.randint(1, size)
            
            # Ensure we do not add a self-loop and the edge does not already exist
            if node1 != node2 and not graph.are_connected(node1, node2):
                graph.add_edge(node1, node2, weight)
                edges -= 1
        
        graphs.append(graph)
    return graphs

graphs = graphGenerator(5, 10, 5, 20)