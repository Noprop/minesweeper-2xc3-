from typing import List
import math
import networkx as nx
import matplotlib.pyplot as plt
import random

class WeightedGraph:
    def __init__(self,nodes):
        self.graph=[]
        self.weights={}
        for _ in range(nodes):
            self.graph.append([])

    def add_node(self,node):
        self.graph[node]=[]

    def add_edge(self, node1, node2, weight):
        if node2 not in self.graph[node1]:
            self.graph[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def get_weights(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def are_connected(self, node1, node2):
        for neighbour in self.graph[node1]:
            if neighbour == node2:
                return True
        return False

    def get_neighbors(self, node):
        return self.graph[node]

    def get_number_of_nodes(self,):
        return len(self.graph)
    
    def get_nodes(self,):
        return [i for i in range(len(self.graph))]
    
    def display_graph(self):
        # use the networkx directed graph
        G = nx.DiGraph()
        for node in range(len(self.graph)):
            for neighbor in self.graph[node]:
                weight = self.get_weights(node, neighbor)
                G.add_edge(node, neighbor, weight=weight)

        # use the spring layout, but increase the distance between the nodes using k
        # pos = nx.spring_layout(G, k=0.1, iterations=5)
        pos = nx.spring_layout(G, k=5/math.sqrt(self.get_number_of_nodes()), iterations=5)

        # create figure, edge labels, and draw
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, node_shape='o')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # display
        plt.show()

def graphGenerator(num_graphs: int, size: int, min_edges: int, max_edges: int) -> List[WeightedGraph]:
    graphs = []
    for _ in range(num_graphs):
        graph = WeightedGraph(size)
        edges = random.randint(min_edges, max_edges)
        # this loop decreases in efficiency for large graph sizes
        while edges > 0:
            node1 = random.randint(0, size-1)
            node2 = random.randint(0, size-1)
            weight = random.randint(1, size)
            if node1 != node2 and not graph.are_connected(node1, node2):
                edges -= 1
                graph.add_edge(node1, node2, weight)
        graphs.append(graph)
    return graphs

graphs = graphGenerator(2, 7, 3, 20)
# graphs[0].display_graph()

static_graph = WeightedGraph(7)
static_graph.add_edge(1, 0, 6)
static_graph.add_edge(1, 5, 2)
static_graph.add_edge(1, 3, 3)
static_graph.add_edge(1, 6, 4)
static_graph.add_edge(1, 4, 4)
static_graph.add_edge(2, 6, 7)
static_graph.add_edge(4, 0, 2)
static_graph.add_edge(4, 2, 5)
static_graph.add_edge(4, 5, 6)
static_graph.add_edge(5, 2, -6)
static_graph.add_edge(5, 6, 4)
static_graph.display_graph()


def bellmanFord(g: WeightedGraph, s: int, k: int):
    n = g.get_number_of_nodes()
    dist = [float('inf') for _ in range(n)]
    prev = [None for _ in range(n)]
    relax = [k for _ in range(n)]
    dist[s] = 0
    prev[s] = s

    # v-1 max iterations
    for _ in range(n-1):
        for node in range(n):
            for nb in g.get_neighbors(node):
                if (relax[nb]) == 0:
                    continue
                weight = g.get_weights(node, nb)
                if (dist[node] + weight) < dist[nb]:
                    relax[nb] -= 1
                    dist[nb] = dist[node] + weight
                    prev[nb] = node

    return (dist, prev)

bellmanFord(static_graph, 1, 1)