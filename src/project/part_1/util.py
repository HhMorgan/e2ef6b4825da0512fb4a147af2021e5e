import random
import networkx as nx

def read_instance(filename: str) -> nx.Graph:
    with open(filename, "r", encoding="utf-8") as f:
        n_nodes = int(f.readline())
        n_edges = int(f.readline())

        G = nx.Graph()
        G.add_nodes_from(range(1, n_nodes+1))

        for line in f:
            values = [int(i) for i in line.split()]
            if len(values) == 4:
                G.add_edge(values[1], values[2], id=values[0], cost=values[3])

        return G
    
def write_instance(filename: str, graph: nx.Graph):
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write(f"{graph.number_of_nodes()}\n")
        f.write(f"{graph.number_of_edges()}\n")

        for i, j, data in graph.edges(data=True):
            f.write(f"{data['id']} {i} {j} {data['cost']}\n")

def write_solution(filename: str, edge_ids: list[int]):
    with open(filename, "w", encoding="utf-8") as f:
        for edge_id in edge_ids:
            f.write(f"{edge_id}\n")


def create_random_instance(n_nodes: int, n_edges: int, random_seed: int = 42) -> nx.Graph:
    assert n_edges <= n_nodes * (n_nodes - 1) / 2
    rnd = random.Random(random_seed)

    nodes = range(1, n_nodes+1)
    connected_nodes = list(nodes)[:1]
    unconnected_nodes = list(nodes)[1:]
    rnd.shuffle(unconnected_nodes)

    edges = set()
    # first n-1 edges: build a tree
    # connect unconnected node to connected node
    while unconnected_nodes:
        i = rnd.choice(connected_nodes)
        j = unconnected_nodes.pop()

        assert j not in connected_nodes
        connected_nodes.append(j)

        i, j = sorted((i, j))
        assert (i,j) not in edges

        edges.add((i, j))

    assert list(sorted(connected_nodes)) == list(nodes)
    assert len(edges) == n_nodes-1

    # remaining edges: connect two random nodes
    while len(edges) < n_edges:
        i, j = rnd.sample(connected_nodes, 2)
        i, j = sorted((i, j))
        assert i != j
        
        if (i,j) in edges:
            continue
        
        edges.add((i, j))

    assert len(edges) == n_edges

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for id, (i, j) in enumerate(sorted(edges)):
        cost = rnd.randrange(1, 1000)
        graph.add_edge(i, j, id=id, cost=cost)

    assert graph.number_of_nodes() == n_nodes
    assert graph.number_of_edges() == n_edges

    return graph
