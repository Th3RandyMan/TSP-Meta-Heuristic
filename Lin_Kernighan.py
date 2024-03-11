#
#   Lin-Kernighan algorithm for the Traveling Salesman Problem
#   Following the documentation from the following link:
#   https://arthur.maheo.net/implementing-lin-kernighan-in-python/
#
from copy import deepcopy
import numpy as np

class LKAlgorithm():
    """
    
    """
    names = [
        "New York",
        "Los Angeles",
        "Chicago",
        "Minneapolis",
        "Denver",
        "Dallas",
        "Seattle",
        "Boston",
        "San Francisco",
        "St. Louis",
        "Houston",
        "Phoenix",
        "Salt Lake City"
        ]   

    matrix = np.array([
        [   0, 2451,  713, 1018, 1631, 1374, 2408,  213, 2571,  875, 1420, 2145, 1972], # New York
        [2451,    0, 1745, 1524,  831, 1240,  959, 2596,  403, 1589, 1374,  357,  579], # Los Angeles
        [ 713, 1745,    0,  355,  920,  803, 1737,  851, 1858,  262,  940, 1453, 1260], # Chicago
        [1018, 1524,  355,    0,  700,  862, 1395, 1123, 1584,  466, 1056, 1280,  987], # Minneapolis
        [1631,  831,  920,  700,    0,  663, 1021, 1769,  949,  796,  879,  586,  371], # Denver
        [1374, 1240,  803,  862,  663,    0, 1681, 1551, 1765,  547,  225,  887,  999], # Dallas
        [2408,  959, 1737, 1395, 1021, 1681,    0, 2493,  678, 1724, 1891, 1114,  701], # Seattle
        [ 213, 2596,  851, 1123, 1769, 1551, 2493,    0, 2699, 1038, 1605, 2300, 2099], # Boston
        [2571,  403, 1858, 1584,  949, 1765,  678, 2699,    0, 1744, 1645,  653,  600], # San Francisco
        [ 875, 1589,  262,  466,  796,  547, 1724, 1038, 1744,    0,  679, 1272, 1162], # St. Louis
        [1420, 1374,  940, 1056,  879,  225, 1891, 1605, 1645,  679,    0, 1017, 1200], # Houston
        [2145,  357, 1453, 1280,  586,  887, 1114, 2300,  653, 1272, 1017,    0,  504], # Phoenix
        [1972,  579, 1260,  987,  371,  999,  701, 2099,  600, 1162,  1200,  504,   0]  # Salt Lake City
    ])


    def __init__(self, edges = np.array([])):  # there is a fast bool parameter?
        if(edges.shape == (0,)):
            self.edges = self.matrix
        else:
            self.edges = np.array(edges)
        self.n_nodes = self.edges.shape[0]
        self.tour = np.arange(self.n_nodes)     # Initial tour is enumaration of nodes
        self.tour_length = self.get_tour_length(self.tour)


    def get_tour_length(self, tour) -> float:
        """
        Calculate the length of a tour
        """
        length = 0
        for i in range(len(tour)):
            length += self.edges[tour[i-1], tour[i]]
        return length
    

    def optimize(self) -> tuple:
        """
        Optimize the tour if not already optimized

        :return: optimized tour and its length
        """
        # tour = str(self.tour)
        # if(tour in self.tours):
        #     print("Tour already optimized")
        #     return self.tours[tour]["tour"], self.tours[tour]["tour_length"]
        # else:
        self._optimize()
        
        return self.tour, self.tour_length
    

    def _optimize(self) -> None:
        """
        Optimize the tour
        """
        self.solutions = set() # Set of solutions with no duplicates
        self.neighbors = {}     # Dictionary of neighbors

        for i in self.tour: # MIGHT REPLACE SINCE NOT REALLY NECESSARY
            self.neighbors[i] = []
            for j, dist in enumerate(self.edges[i]):
                if dist > 0 and j in self.tour:
                    self.neighbors[i].append(j)

        iteration = 0
        improving = True
        while improving: # HOW DOES THIS IMPROVE THE TOUR?
            improving = self._improve()
            self.solutions.add(str(self.tour))
            print(f"Iteration {iteration} complete")
            print(f"Current best tour has length {self.tour_length}")
            print(f"Current best tour is {self.tour}")
            print()
            iteration += 1


    def _improve(self) -> bool:
        """
        Improve the tour
        :return: True if the tour is improved, False otherwise
        """
        self.create_tour_edges(self.tour)
        #DO I NEED TO CREATE A NEW TOUR?

        for node in self.tour: # For each node in the tour
            # Look at locations before and after the target node
            nearby = self._nearby(node)
            for neighbor in nearby:
                removed_edges = set([self._pair(node, neighbor)]) # Set of broken edges
                gain = self.edges[node, neighbor]
                close_nodes = self.closest(neighbor, gain, removed_edges, set())

                attempts = 5  # Limit the number of attempts to find a better solution
                for close_node, (potential_gain, reduced_gain) in close_nodes:
                    # Confirm that the potential new node is not already in the tour
                    if close_node in nearby:
                        continue

                    added_edges = set([self._pair(neighbor, close_node)])
                    if self.remove_edge(node, close_node, reduced_gain, removed_edges, added_edges):
                        return True # Improvement found

                    attempts -= 1
                    if attempts == 0:
                        break
        
        return False    # No improvement found
    

    def _nearby(self, node):
        """
        Return the nodes around a given node
        """
        index = np.where(self.tour == node)[0][0]
        return [self.tour[index-1], self.tour[(index+1)%len(self.tour)]]
    

    def create_tour_edges(self, tour):
        """
        Create a set of edges from a tour
        """
        self.tour_edges = set()
        for i in range(len(tour)):
            self.tour_edges.add(self._pair(tour[i-1], tour[i]))


    def _pair(self, n1, n2):
        """
        Pair two nodes
        """
        return (n1, n2) if n1 < n2 else (n2, n1)


    def closest(self, target, gain, removed_edges, added_edges):
        """
        Find the closest node to the given target node and its potential gain. Sorted by
        potential gain in descending order.
        """
        neighbors = {} # Look at 5 and 7

        # For each neighbor of the target node
        for neighbor in self.neighbors[target]:
            edge = self._pair(target, neighbor) # Group for reference
            reduced_gain = gain - self.edges[target, neighbor] # Calculate change in gain
            
            # If the gain isn't reduced, the edge is broken edges, or the edge is already in the tour, skip
            if reduced_gain <= 0 or edge in removed_edges or edge in self.tour_edges:
                continue

            # Look at locations before and after the target node
            for node in self._nearby(neighbor):
                edge_path = self._pair(neighbor, node)
                if edge_path not in removed_edges and edge_path not in added_edges:
                    # Calculate difference in distance
                    diff = self.edges[neighbor, node] - self.edges[target, neighbor]

                    # If the neighbor is already in the list of neighbors, update the gain if the difference is greater
                    if neighbor in neighbors and diff > neighbors[neighbor][0]:
                        neighbors[neighbor][0] = diff
                    else:   # Otherwise, add the neighbor to the list of neighbors
                        neighbors[neighbor] = [diff, reduced_gain]
        
        return sorted(neighbors.items(), key=lambda x: x[1][0], reverse=True)


    def remove_edge(self, node, close_node, gain, removed_edges, added_edges):
        """
        Remove an edge from the tour
        :return: True if the edge is removed, False otherwise
        """
        nearby = self._nearby(close_node)
        if(len(removed_edges) == 4):
            n1, n2 = nearby

            if self.edges[n1, close_node] > self.edges[n2, close_node]:
                nearby = [n1]
            else:
                nearby = [n2]

        for neighbor in nearby:
            edge = self._pair(close_node, neighbor)
            current_gain = gain + self.edges[close_node, neighbor]

            if edge not in added_edges and edge not in removed_edges:
                added = deepcopy(added_edges)
                removed = deepcopy(removed_edges)
                
                removed.add(edge)
                added.add(self._pair(node, neighbor))

                new_dist = current_gain - self.edges[node, neighbor]
                valid_tour, new_tour = self.create_tour(removed, added)
                if not valid_tour and len(added) > 2:   # Check if valid tour
                    continue

                if str(new_tour) in self.solutions:
                    return False
                
                if valid_tour and new_dist > 0:
                    self.tour = new_tour
                    self.tour_length -= new_dist
                    return True
                else:
                    return self.add_edge(node, neighbor, current_gain, removed, added_edges)

        return False


    def create_tour(self, removed_edges, added_edges):
        """
        Create a new tour
        """
        # Create edges for new tour
        edges = (self.tour_edges - removed_edges) | added_edges

        if len(edges) < len(self.tour):
            return False, []
        
        nodes_visited = {}
        node = 0
        while len(edges) > 0:
            for i,j in edges:
                if i == node:
                    nodes_visited[node] = j
                    node = j
                    break
                elif j == node:
                    nodes_visited[node] = i
                    node = i
                    break
            
            edges.remove((i,j))
        
        if len(nodes_visited) < len(self.tour):
            return False, []
        
        node = nodes_visited[0]
        new_tour = [0]
        visited = set(new_tour)

        while node not in visited:
            new_tour.append(node)
            visited.add(node)
            node = nodes_visited[node]

        return len(new_tour) == len(self.tour), np.array(new_tour)

    def add_edge(self, node, neighbor, gain, removed_edges, added_edges):
        """
        Add an edge to the tour
        """
        close = self.closest(neighbor, gain, removed_edges, added_edges)

        if len(removed_edges) == 2:
            n_neighbors = 5 # Check 5 closest neighbors
        else:
            n_neighbors = 1 # Only check closest neighbor

        for close_node, (potential_gain, reduced_gain) in close[:n_neighbors]:
            edge = self._pair(neighbor, close_node)
            added = deepcopy(added_edges)
            added.add(edge)

            if self.remove_edge(node, close_node, reduced_gain, removed_edges, added):
                return True

        return False
    


# Test the algorithm
if __name__ == "__main__":
    # Create a distance matrix
    #edges = np.array([  [0, 2, 9, 10],
                        # [1, 0, 6, 4],
                        # [15, 7, 0, 8],
                        # [6, 3, 12, 0]   ])
    

    # Create an instance of the LKAlgorithm
    lk = LKAlgorithm()
    tour, dist = lk.optimize()
    print(f"Best path has length {dist}")
    print(f"Best path is {tour}")