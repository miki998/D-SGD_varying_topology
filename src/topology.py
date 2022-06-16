import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Topology:
    def __init__(self, n_workers):
        self.n_workers = n_workers
        
    def get_neighbors(self, worker):
        raise NotImplementedError()

    def to_networkx(self):
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(self.n_workers)) 

        for worker in range(self.n_workers):
            graph.add_edges_from([(worker, neighbor)  for neighbor in self.get_neighbors(worker)])
        
        self.graph = graph
        return graph
    
    def draw_graph(self, with_labels=False):
        if with_labels:
            nx.draw_networkx(self.graph, with_labels=with_labels, node_size=500)
            plt.show()
        else:
            nx.draw(self.graph, node_size=500)
            plt.show()


class ringTopology(Topology):
    def get_neighbors(self, worker):
        client = worker
        n = self.n_workers
        return [(client - 1) % n, (client + 1) % n]

class starTopology(Topology):
    def get_neighbors(self, worker):
        if worker == 0:
            return[j for j in range(1, self.n_workers)]
        else:
            return [0]

class FullConnectedTopology(Topology):
    def get_neighbors(self, worker):
        return [j for j in range(self.n_workers) if j != worker]

class SocialNetworkTopology(Topology):
    def __init__(self):
        social_network = nx.florentine_families_graph()
        self.graph = nx.relabel.convert_node_labels_to_integers(social_network)
        self.n_workers = len(self.graph.nodes)

    def to_networkx(self):
        return self.graph

    def get_neighbors(self, worker):
        return(list(self.graph.neighbors(worker)))

class WheelTopology(Topology):
    def __init__(self, n_workers):
        super().__init__(n_workers)

        wheel = nx.wheel_graph(n=n_workers)
        self.graph = nx.relabel.convert_node_labels_to_integers(wheel)
        self.n_workers = len(self.graph.nodes)

    def to_networkx(self):
        return self.graph

    def get_neighbors(self, worker):
        return(list(self.graph.neighbors(worker)))

class LadderTopology(Topology):
    def __init__(self, n_workers):
        super().__init__(n_workers)

        ladder = nx.ladder_graph(n=n_workers//2)
        self.graph = nx.relabel.convert_node_labels_to_integers(ladder)
        self.n_workers = len(self.graph.nodes)

    def to_networkx(self):
        return self.graph

    def get_neighbors(self, worker):
        return(list(self.graph.neighbors(worker)))

class BarbellTopology(Topology):
    def __init__(self, n_workers):
        super().__init__(n_workers)
        
        path_lenth = n_workers - (2*(n_workers // 2))
        barbell = nx.barbell_graph(m1=(n_workers // 2), m2=path_lenth)
        self.graph = nx.relabel.convert_node_labels_to_integers(barbell)
        self.n_workers = len(self.graph.nodes)

    def to_networkx(self):
        return self.graph

    def get_neighbors(self, worker):
        return(list(self.graph.neighbors(worker)))

class TorusTopology(Topology):
    def __init__(self, n_workers, dimension='2d'):
        super().__init__(n_workers)

        if dimension == '2d':
            torus = nx.generators.lattice.grid_2d_graph(int(np.sqrt(self.n_workers)), int(np.sqrt(self.n_workers)), periodic=True)
            self.graph = nx.relabel.convert_node_labels_to_integers(torus)
            self.n_workers = len(self.graph.nodes)
        elif dimension == '3d':
            torus = nx.grid_graph(dim=(int(np.cbrt(self.n_workers)), int(np.cbrt(self.n_workers)), int(np.cbrt(self.n_workers))), periodic=True)
            self.graph = nx.relabel.convert_node_labels_to_integers(torus)
            self.n_workers = len(self.graph.nodes)          

    def to_networkx(self):
        return self.graph

    def get_neighbors(self, worker):
        return(list(self.graph.neighbors(worker)))
    
    
class ChainTopology(Topology):
    def get_neighbors(self, worker):
        if worker < 1:
            return [1]
        elif worker >= self.n_workers-1:
            return [worker - 1]
        else:
            return [worker - 1, worker + 1]

class HyperCubeTopology(Topology):
    def get_neighbors(self, worker):
        n = self.n_workers
        x = worker

        y = int(np.log2(n))
        assert 2**y == n

        return [x ^ (2**z) for z in range(0, y)]    
    
class BinaryTreeTopology(Topology):
    def get_neighbors(self, worker):
        if self.n_workers == 1:
            return[]
        elif worker >= self.n_workers or worker < 0:
            raise ValueError('Your worker is not in the range [0, {}]'.format(self.n_workers))
        elif worker == 0:
            return [1, 2]
        else:  
            neighbors = []
            neighbors.append((worker-1) // 2)
            children = [child for child in [worker * 2 + 1, worker * 2 + 2] if child < self.n_workers]
            neighbors.extend(children)
            return neighbors

class CustomTopology(Topology):
    
    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.adjency   = np.zeros((n_workers,n_workers))

    def set_neighbours(self,adjency):
        self.adjency = adjency        
        
    def get_neighbors(self, worker):
        client = worker
        neighbors = []
        for k in range(len(self.adjency[client])):
            if self.adjency[client,k] == 1:
                neighbors.append(k)
        return neighbors

def sampleAdj(nbEdge,nbSamp,K,seed=1):
    """
    desc:
    """
    def convertIdx(idx,K):
        I   = 0
        tmp = idx
        row = K - 1 
        while tmp - row >= 0:
            I += 1
            tmp -= row
            row -= 1

        return  I, tmp + I + 1
    
    np.random.seed(seed)
    possibleEdges = (K**2-K)//2

    adj  = np.zeros((K,K))
    comb = np.random.choice(range(possibleEdges),nbEdge, replace=False)
    for entry in comb:
        c = convertIdx(entry, K)
        adj[c[0],c[1]] = 1
        
    return adj + adj.T