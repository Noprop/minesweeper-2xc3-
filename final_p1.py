from typing import List
import math
import time
import timeit
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

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

# graphs = graphGenerator(2, 7, 3, 20) # list length 2

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

def get_path(start, end, pred):
    path = [start]
    node = start
    while node != end:
        node = pred[node]
        path.append(node)
    path.reverse()
    return path

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
    #prev = [None for _ in range(n)]
    path = [[] for _ in range(n)]
    relax = [k for _ in range(n)]
    dist[s] = 0
    #prev[s] = s
    path[s] = [s]

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
                    #prev[nb] = node
                    path[nb] = path[node] + [nb]

    dp = {}

    for i in range(g.get_number_of_nodes()):
        dp[i] = (dist[i], path[i])

    return dp

def dijkstra(Graph,source,destination):
    visited = {}
    distance = {}
    path = {}

    # create empty queue
    Q = MinHeap([])

    for i in range(Graph.get_number_of_nodes()):

        visited[i] = False
        distance[i] = float("inf")
        path[i] = []

        # insert the nodes in the minheap
        Q.insert(Item(i, float("inf")))

    # assign 0 to source 
    Q.decrease_key(source, 0)
    distance[source] = 0
    path[source] = [source]

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
                    path[neighbour] = path[current_node] + [neighbour]
                    distance[neighbour] = temp
                    Q.decrease_key(neighbour, temp)

    return path[destination]
    # if (distance[destination] < float('inf')):
        # return get_path(destination, source, path)

    # return distance[destination]

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

def experiment1():
    runs = 50
    sizes = [5,10,15,20]
    dijkstra_times = []
    bellman_times = []

    for size in sizes:
        dijkstra_total = 0
        bellman_total = 0
        graphs = graphGenerator(runs, size, size*(size-1)/5, size*(size-1)/5) # keep the number of edges at a static 1/5th of maximum
        
        for graph in graphs:
            start = timeit.default_timer()
            dijkstra_relaxes(graph, 0, size-1)
            stop = timeit.default_timer()
            dijkstra_total += stop-start

        for graph in graphs:
            start = timeit.default_timer()
            bellmanFord_relaxes(graph, 0, size-1)
            stop = timeit.default_timer()
            bellman_total += stop-start
        
        dijkstra_times.append(dijkstra_total / runs)
        bellman_times.append(bellman_total / runs)

    x_axis = np.arange(0, len(sizes),1)
    plt.figure(figsize=(20, 8))
    plt.bar(x_axis-0.1, dijkstra_times, 0.2, color='#b33300', label='Dijkstra')
    plt.bar(x_axis+0.1, bellman_times, 0.2, color='#8bc1c7', label='Bellman Ford')
    plt.xticks(x_axis, sizes)
    plt.title("Time comparison for Dijkstra vs Bellman Ford for varying graph sizes")
    plt.ylabel("Run time in ms in order of 1e-6")
    plt.xlabel("# of nodes")
    plt.legend()
    plt.show()

def experiment2():
    runs = 50
    size = 10
    edges = [5, 10, 50, 90]
    dijkstra_times = []
    bellman_times = []

    for edge in edges:
        dijkstra_total = 0
        bellman_total = 0
        graphs = graphGenerator(runs, size, edge, edge)
        
        for graph in graphs:
            start = timeit.default_timer()
            dijkstra_relaxes(graph, 0, size-1)
            stop = timeit.default_timer()
            dijkstra_total += stop-start

        for graph in graphs:
            start = timeit.default_timer()
            bellmanFord_relaxes(graph, 0, size-1)
            stop = timeit.default_timer()
            bellman_total += stop-start
        
        dijkstra_times.append(dijkstra_total / runs)
        bellman_times.append(bellman_total / runs)

    x_axis = np.arange(0, len(edges),1)
    plt.figure(figsize=(20, 8))
    plt.bar(x_axis-0.1, dijkstra_times, 0.2, color='#b33300', label='Dijkstra')
    plt.bar(x_axis+0.1, bellman_times, 0.2, color='#8bc1c7', label='Bellman Ford')
    plt.xticks(x_axis, edges)
    plt.title("Time comparison for Dijkstra vs Bellman Ford for graphs with varying number of edges (size = 10)")
    plt.ylabel("Run time in ms in order of 1e-6")
    plt.xlabel("# of edges")
    plt.legend()
    plt.show()

def experiment3():
    runs = 50
    size = 20
    edge = 100
    k_values = [1,2,3,4,5,19]
    dijkstra_accuracy = []
    bellman_accuracy = []

    for k in k_values:
        dijkstra_total = 0
        bellman_total = 0
        graphs = graphGenerator(runs, size, edge, edge)
        
        for graph in graphs:
            accurate = dijkstra_relaxes(graph, 0, size-1)


            test = dijkstra_relaxes(graph, 0, k)
            
            for i in range(0,size):
                if accurate[i][1] == test[i][1]:
                    dijkstra_total += 1

        for graph in graphs:
            accurate = bellmanFord_relaxes(graph, 0, size-1)
            test = bellmanFord_relaxes(graph, 0, k)

            for i in range(0,size):
                if accurate[i][1] == test[i][1]:
                    bellman_total += 1
            
        
        dijkstra_accuracy.append(int(dijkstra_total / runs / size * 100))
        bellman_accuracy.append(int(bellman_total / runs / size * 100))

    x_axis = np.arange(0, len(k_values),1)
    plt.figure(figsize=(20, 8))
    plt.bar(x_axis-0.1, dijkstra_accuracy, 0.2, color='#b33300', label='Dijkstra')
    plt.bar(x_axis+0.1, bellman_accuracy, 0.2, color='#8bc1c7', label='Bellman Ford')
    plt.xticks(x_axis, k_values)
    plt.axhline(100,color="red",linestyle="--")
    plt.title("Time comparison for Dijkstra vs Bellman Ford with varying values of k (size = 20, edges = 100)")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("k values")
    plt.legend()
    plt.show()

# experiment3()

def all_pair_positive(Graph : WeightedGraph):
    size = Graph.get_number_of_nodes()
    nodes = [i for i in range(0,size)]
    return_list = [{} for _ in nodes]

    for source in nodes:
        visited = {}
        distance = {}
        prev = {}
        
        # create empty queue
        Q = MinHeap([])

        for i in range(Graph.get_number_of_nodes()):

            visited[i] = False
            distance[i] = float("inf")
            prev[i] = None

            # insert the nodes in the minheap
            Q.insert(Item(i, float("inf")))

        # assign 0 to source 
        Q.decrease_key(source, 0)
        distance[source] = 0

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
                        prev[neighbour] = current_node
                        distance[neighbour] = temp
                        Q.decrease_key(neighbour, temp)

        for i in range(Graph.get_number_of_nodes()):
            return_list[source][i] = (distance[i], prev[i])

    return return_list

def all_pair_negative(g : WeightedGraph):
    size = g.get_number_of_nodes()
    nodes = [i for i in range(0,size)]
    return_list = [{} for _ in nodes]

    for s in nodes:
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

        for i in range(g.get_number_of_nodes()):
            return_list[s][i] = (dist[i], prev[i])
            
    return return_list

def heuristic(source: int, n: int) -> int:
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
    fScore[source] = h(source, source)

    while not Q.is_empty():
        current = Q.extract_min().value
        if current == goal:
            return (prev, get_path(goal, source, prev))

        for nb in g.get_neighbors(current):
            temp_gScore = gScore[current] + g.get_weights(current, nb)
            if temp_gScore < gScore[nb]:
                prev[nb] = current
                gScore[nb] = temp_gScore
                fScore[nb] = temp_gScore + h(source, nb)
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

# print(AStar(static_graph_positive, 1, 2, heuristic))
# static_graph_positive.display_graph()

# Part 4

class Dijkstra_AStar_Analysis:
    def __init__(self):
        self.stations = {}
        self.connections = []
        self.lines = {}

        with open('london_stations.csv', newline='') as file1:
            reader1 = csv.reader(file1)
            next(reader1)
            for row in reader1:
                self.stations[int(row[0])] = {
                    "name": row[3],
                    "lat": float(row[1]),
                    "long": float(row[2]),
                    "zone": row[5],
                    "total_lines": row[6],
                    "rail": row[7],
                }

        with open('london_connections.csv', newline='') as file2:
            reader2 = csv.reader(file2)
            next(reader2)
            for row in reader2:
                s1, s2, line = int(row[0]), int(row[1]), int(row[2])

                self.connections.append({
                    "s1": s1,
                    "s2": s2,
                    "line": row[2],
                    "time": row[3]
                })
                if s1 in self.lines:
                    self.lines[s1][s2] = line
                else:
                    self.lines[s1] = { s2: line }
                if s2 in self.lines:
                    self.lines[s2][s1] = line
                else:
                    self.lines[s2] = { s1: line }

        self.distances = {}
        for sid1 in self.stations:
            for sid2 in self.stations:
                s1 = self.stations[sid1]
                s2 = self.stations[sid2]

                lat1 = s1["lat"]
                lon1 = s1["long"]
                lat2 = s2["lat"]
                lon2 = s2["long"]
                distance = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

                if sid1 in self.distances:
                    self.distances[sid1][sid2] = distance
                else:
                    self.distances[sid1] = {
                        sid2: distance
                    }

    def heuristic(self, s1, s2):
        return self.distances[s1][s2]

    def create_graph(self):
        self.graph = WeightedGraph(len(self.stations)+2)
        for edge in self.connections:
            distance = self.heuristic(edge["s1"], edge["s2"])
            self.graph.add_edge(edge["s1"], edge["s2"], distance)
            self.graph.add_edge(edge["s2"], edge["s1"], distance)
    
    def run_experiment1(self, length="short"):
        dijkstra_speed = []
        astar_speed = []

        station_ids = list(self.stations.keys())
        station_ids.sort()

        if length == "short":
            station_ids = station_ids[:12]

        # for every station, check the path to every other station
        for s1 in station_ids:
            dspeed = []
            aspeed = []

            # test the time to find every other station from s1
            for s2 in self.stations:
                if s1 == s2:
                    continue

                # test dijkstra
                start1 = timeit.default_timer()
                dijkstra(self.graph, s1, s2)
                stop1 = timeit.default_timer()
                dspeed.append(stop1-start1)

                # test astar
                start2 = timeit.default_timer()
                AStar(self.graph, s1, s2, self.heuristic)
                stop2 = timeit.default_timer()
                aspeed.append(stop2-start2)

            print('adding: ', s1)
            dijkstra_speed.append(sum(dspeed))
            astar_speed.append(sum(aspeed))

        # print(dijkstra_speed)
        # print(astar_speed)

        # to avoid 600+ bars we will aggregate at a max size of 20 stations
        num_bins = min(12, len(dijkstra_speed))
        bin_size = len(dijkstra_speed) // num_bins

        # sum the items 1-20, 21-40, 41-60, etc.
        binned_dijkstra_speed = [sum(dijkstra_speed[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)]
        binned_astar_speed = [sum(astar_speed[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)]

        # configure and display the plot
        plt.figure(figsize=(10, 5))
        x_axis = np.arange(num_bins)
        if len(dijkstra_speed) <= num_bins:
            plt.xticks(x_axis, [str(i) for i in range(num_bins)])
            plt.xlabel("Stations IDs")
        else:
            plt.xticks(x_axis, [str((i*bin_size+1)) + '-' + str(((i+1)*bin_size)) for i in range(num_bins)])
            plt.xlabel("Group of Stations (IDs)")

        plt.bar(x_axis - 0.2, binned_dijkstra_speed, 0.4, color='#b33300', label='Dijkstra')
        plt.bar(x_axis + 0.2, binned_astar_speed, 0.4, color='#8bc1c7', label='A*')

        plt.title("Time comparison for Dijkstra vs A* on the London Subway")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.show()

    def run_experiment2(self):
        dspeed = []
        aspeed = []

        # bottom, left station 118: Heathrow Terminal 4
        # top, right station 88: Epping
        # top, right station 256: Theydon Bois (adjacent to Epping)
        # top, right station 153: Leyton (10 stations away from Epping)
        # center station 192: Oxford Circus

        tests = [
            [118, 88], # corner -> corner
            [88, 118],
            [88, 192], # corner -> center
            [192, 88], # center -> corner
            [149, 162], # adjacent
            [162, 149],
            [88, 153], # 10 stations away
            [153, 88],
        ]
        for test in tests:
            # test dijkstra
            start1 = timeit.default_timer()
            dijkstra(self.graph, test[0], test[1])
            stop1 = timeit.default_timer()
            dspeed.append((stop1-start1)/1000)

            # test astar
            start2 = timeit.default_timer()
            AStar(self.graph, test[0], test[1], self.heuristic)
            stop2 = timeit.default_timer()
            aspeed.append((stop2-start2)/1000) 

        # configure and display the plot
        plt.figure(figsize=(10, 8))
        x_axis = np.arange(len(dspeed))
        plt.xticks(x_axis, [
            "Corner to Corner", "Corner to Corner Reversed", 
            "Corner to Center", "Center to Corner",
            "Adjacent", "Adjacent Reversed",
            "10 Stations Away", "10 Stations Away Reversed", 
        ], rotation=45)
        plt.xlabel("Test Conducted")

        plt.bar(x_axis - 0.2, dspeed, 0.4, color='#b33300', label='Dijkstra')
        plt.bar(x_axis + 0.2, aspeed, 0.4, color='#8bc1c7', label='A*')

        plt.title("Time comparison for Dijkstra vs A* on the London Subway")
        plt.ylabel("Time (ms)")
        plt.tight_layout()
        plt.legend()
        plt.show()
    
    def run_experiment3(self, length="short"):
        # 1 is 1 line travelled, 2 is 2 lines travlled, 3 is 3+ lines travelled
        dspeed_total = { 1: [], 2: [], 3: [] }
        aspeed_total = { 1: [], 2: [], 3: [] }

        station_ids = list(self.stations.keys())
        station_ids.sort()

        if length == "short":
            station_ids = station_ids[:10]

        # for every station, check the path to every other station
        for s1 in station_ids:
            print('s1: ', s1)
            dspeed = { 1: [], 2: [], 3: [] }
            aspeed = { 1: [], 2: [], 3: [] }

            # test the time to find every other station from s1
            for s2 in self.stations:
                if s1 == s2:
                    continue
                # test dijkstra
                start1 = timeit.default_timer()
                path_d = dijkstra(self.graph, s1, s2)
                stop1 = timeit.default_timer()

                # check amount of lines taken
                lines_visited_d = {}
                for i in range(0, len(path_d)-1, 2):
                    line = self.lines[path_d[i]][path_d[i+1]]
                    if line not in lines_visited_d:
                        lines_visited_d[line] = True
                lines_d = len(lines_visited_d.keys())

                # test astar
                start2 = timeit.default_timer()
                path_a = AStar(self.graph, s1, s2, self.heuristic)[1]
                stop2 = timeit.default_timer()

                # check amount of lines taken
                lines_visited_a = {}
                for i in range(0, len(path_a)-1, 2):
                    line = self.lines[path_a[i]][path_a[i+1]]
                    if line not in lines_visited_a:
                        lines_visited_a[line] = True
                lines_a = len(lines_visited_a.keys())

                lines = min(lines_a, lines_d) # should be the same

                if lines == 1 or lines == 2:
                    dspeed[lines].append((stop1-start1)/1000)
                    aspeed[lines].append((stop2-start2)/1000) 
                else:
                    dspeed[3].append((stop1-start1)/1000)
                    aspeed[3].append((stop2-start2)/1000)
            for i in range(1, 4):
                dspeed_total[i] = sum(dspeed[i])
                aspeed_total[i] = sum(aspeed[i])

        # configure and display the plot
        plt.figure(figsize=(10, 8))
        x_axis = np.arange(len(dspeed))
        plt.xticks(x_axis, [
            "1 Line",
            "2 Lines",
            "3+ Lines"
        ], rotation=45)
        plt.xlabel("Amount of Lines Travelled on the Shortest Path")

        plt.bar(x_axis - 0.2, [sum(dspeed[i])/len(dspeed[i]) for i in range(1, 4)], 0.4, color='#b33300', label='Dijkstra')
        plt.bar(x_axis + 0.2, [sum(aspeed[i])/len(aspeed[i]) for i in range(1, 4)], 0.4, color='#8bc1c7', label='A*')

        plt.title("Time comparison for Dijkstra vs A* on the London Subway")
        plt.ylabel("Time (ms)")
        plt.tight_layout()
        plt.legend()
        plt.show()


london_subway = Dijkstra_AStar_Analysis()
london_subway.create_graph()
# london_subway.run_experiment1()
# london_subway.run_experiment2()
# london_subway.run_experiment3()

