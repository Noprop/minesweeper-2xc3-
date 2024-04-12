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

graphs = graphGenerator(2, 7, 3, 20) # list length 2

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

static_graph_positive = WeightedGraph(7)
static_graph_positive.add_edge(1, 0, 6)
static_graph_positive.add_edge(1, 5, 2)
static_graph_positive.add_edge(1, 3, 3)
static_graph_positive.add_edge(1, 6, 4)
static_graph_positive.add_edge(1, 4, 4)
static_graph_positive.add_edge(2, 6, 7)
static_graph_positive.add_edge(4, 0, 2)
static_graph_positive.add_edge(4, 2, 5)
static_graph_positive.add_edge(4, 5, 6)
static_graph_positive.add_edge(5, 2, 6)
static_graph_positive.add_edge(5, 6, 4)


def bellmanFord(g: WeightedGraph, s: int):
    n = g.get_number_of_nodes()
    dist = [float('inf') for _ in range(n)]
    prev = [None for _ in range(n)]
    dist[s] = 0
    prev[s] = s

    # v-1 max iterations
    for _ in range(n-1):
        for node in range(n):
            for nb in g.get_neighbors(node):
                weight = g.get_weights(node, nb)
                if (dist[node] + weight) < dist[nb]:
                    dist[nb] = dist[node] + weight
                    prev[nb] = node

    return (dist, prev)

def bellmanFord_relaxes(g: WeightedGraph, s: int, k: int):
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

def dijkstra(Graph,source,destination):
    visited = {}
    distance = {}

    # create empty queue
    Q = MinHeap([])

    for i in range(Graph.get_number_of_nodes()):

        visited[i] = False
        distance[i] = float("inf")

        # insert the nodes in the minheap
        Q.insert(Item(i, float("inf")))

    # assign 0 to source 
    Q.decrease_key(source, 0)
    distance[source] = 0

    while not (Q.is_empty() or visited[destination]):
        # get current node
        current_node = Q.extract_min().value
        visited[current_node] = True

        for neighbour in Graph.graph[current_node]:
            # get weight of current node
            edge_weight = Graph.get_weights(current_node, neighbour)

            temp = distance[current_node] + edge_weight

            # not visited yet
            if not visited[neighbour]:
                if temp < distance[neighbour]:
                    distance[neighbour] = temp
                    Q.decrease_key(neighbour, temp)

    return distance[destination]

def dijkstra_relaxes(Graph,source,k):
    visited = {}
    distance = {}
    path = {}
    relaxed = {}
    
    # create empty queue
    Q = MinHeap([])

    for i in range(Graph.get_number_of_nodes()):

        visited[i] = False
        distance[i] = float("inf")
        relaxed[i] = k
        path[i] = []

        # insert the nodes in the minheap
        Q.insert(Item(i, float("inf")))

    # assign 0 to source 
    Q.decrease_key(source, 0)
    distance[source] = 0
    path[source] = [source]

    while not (Q.is_empty() ):
        # get current node
        current_node = Q.extract_min().value
        visited[current_node] = True

        for neighbour in Graph.graph[current_node]:
            # get weight of current node
            edge_weight = Graph.get_weights(current_node, neighbour)
            temp = distance[current_node] + edge_weight

            # not visited yet
            if not visited[neighbour]:
                if temp < distance[neighbour]:
                    if relaxed[neighbour] > 0:
                        path[neighbour] = path[current_node] + [neighbour]
                        distance[neighbour] = temp
                        Q.decrease_key(neighbour, temp)
                        relaxed[neighbour] -= 1

    dp = {}

    for i in range(Graph.get_number_of_nodes()):
        dp[i] = (distance[i], path[i])
        
    return dp

def heuristic(n: int) -> int:
    # hardcoded
    heuristic_dict = {
        0: 0.0,
        1: 1.0,
        2: -15.0,
        3: 3.0,
        4: -16.0,
        5: 5.0,
        6: 6.0,
        7: 7.0,
    }
    return heuristic_dict[n]

def AStar(g: WeightedGraph, source: int, goal: int, h: callable):
    n = g.get_number_of_nodes()

    # init our pq, and add source to it
    Q = MinHeap([])
    Q.insert(Item(source, 0))

    # we use this to backtrack and calculate the path
    prev = {}
    # this is the score not including the heuristic value
    gScore = {}
    # this is the score including the heuristic
    # this is what determines which node to check next
    fScore = {}
    for i in range(n):
        prev[i] = None
        gScore[i] = float('inf')
        fScore[i] = float('inf')

    prev[source] = source
    gScore[source] = 0
    fScore[source] = h(source)

    while not Q.is_empty():
        current = Q.extract_min().value
        if current == goal:
            return  (prev, get_path(goal, source, prev))
        
        for nb in g.get_neighbors(current):
            temp_gScore = gScore[current] + g.get_weights(current, nb)
            if temp_gScore < gScore[nb]:
                prev[nb] = current
                gScore[nb] = temp_gScore
                fScore[nb] = temp_gScore + h(nb)
                if Q.contains(nb):
                    Q.decrease_key(nb, fScore[nb])
                else:
                    Q.insert(Item(nb, fScore[nb]))
    return None
    
def get_path(start, end, pred):
    path = [start]
    node = start
    while node != end:
        node = pred[node]
        path.append(node)
    path.reverse()
    return path

print(AStar(static_graph_positive, 1, 2, heuristic))
static_graph_positive.display_graph()

class MinHeap:
    def __init__(self, data):
        self.items = data
        self.length = len(data)
        self.build_heap()

        # add a map based on input node
        self.map = {}
        for i in range(self.length):
            self.map[self.items[i].value] = i

    def find_left_index(self,index):
        return 2 * (index + 1) - 1

    def find_right_index(self,index):
        return 2 * (index + 1)

    def find_parent_index(self,index):
        return (index + 1) // 2 - 1  
    
    def sink_down(self, index):
        smallest_known_index = index

        if self.find_left_index(index) < self.length and self.items[self.find_left_index(index)].key < self.items[index].key:
            smallest_known_index = self.find_left_index(index)

        if self.find_right_index(index) < self.length and self.items[self.find_right_index(index)].key < self.items[smallest_known_index].key:
            smallest_known_index = self.find_right_index(index)

        if smallest_known_index != index:
            self.items[index], self.items[smallest_known_index] = self.items[smallest_known_index], self.items[index]
            
            # update map
            self.map[self.items[index].value] = index
            self.map[self.items[smallest_known_index].value] = smallest_known_index

            # recursive call
            self.sink_down(smallest_known_index)

    def build_heap(self,):
        for i in range(self.length // 2 - 1, -1, -1):
            self.sink_down(i) 

    def insert(self, node):
        if len(self.items) == self.length:
            self.items.append(node)
        else:
            self.items[self.length] = node
        self.map[node.value] = self.length
        self.length += 1
        self.swim_up(self.length - 1)

    def insert_nodes(self, node_list):
        for node in node_list:
            self.insert(node)

    def swim_up(self, index):
        
        while index > 0 and self.items[index].key < self.items[self.find_parent_index(index)].key:
            #swap values
            self.items[index], self.items[self.find_parent_index(index)] = self.items[self.find_parent_index(index)], self.items[index]
            #update map
            self.map[self.items[index].value] = index
            self.map[self.items[self.find_parent_index(index)].value] = self.find_parent_index(index)
            index = self.find_parent_index(index)

    def get_min(self):
        if len(self.items) > 0:
            return self.items[0]

    def extract_min(self,):
        #xchange
        self.items[0], self.items[self.length - 1] = self.items[self.length - 1], self.items[0]
        #update map
        self.map[self.items[self.length - 1].value] = self.length - 1
        self.map[self.items[0].value] = 0

        min_node = self.items[self.length - 1]
        self.length -= 1
        self.map.pop(min_node.value)
        self.sink_down(0)
        return min_node

    def decrease_key(self, value, new_key):
        if new_key >= self.items[self.map[value]].key:
            return
        index = self.map[value]
        self.items[index].key = new_key
        self.swim_up(index)

    def get_element_from_value(self, value):
        return self.items[self.map[value]]

    def is_empty(self):
        return self.length == 0
    
    def __str__(self):
        height = math.ceil(math.log(self.length + 1, 2))
        whitespace = 2 ** height + height
        s = ""
        for i in range(height):
            for j in range(2 ** i - 1, min(2 ** (i + 1) - 1, self.length)):
                s += " " * whitespace
                s += str(self.items[j]) + " "
            s += "\n"
            whitespace = whitespace // 2
        return s
    
    def contains(self, value):
        return value in self.map

class Item:
    def __init__(self, value, key):
        self.key = key
        self.value = value
    
    def __str__(self):
        return "(" + str(self.key) + "," + str(self.value) + ")"

