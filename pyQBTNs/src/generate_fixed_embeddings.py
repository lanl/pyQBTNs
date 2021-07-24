import networkx as nx
import random
from dwave import embedding
from .utils import remove_values_from_list


def iterative_search(CONNECTIVITY_GRAPH, starting_node, target_clique_size, random_state=42):
    """


    Parameters
    ----------
    CONNECTIVITY_GRAPH : networkx.Graph()
        quantum annealer hardware connectivity graph.
    starting_node : integer
        integer that repressents a node in the hardware graph.
    target_clique_size : integer
        small clique we want to embed multiple times.
    random_state : integer, optional
        random seed. The default is 42.

    Returns
    -------
    -1 or netowkrx.Graph()
        -1 if search failed, otherwise a suitable subgraph for the small clique size.

    """
    random.seed(random_state)

    complete = nx.complete_graph(target_clique_size)
    subg = [starting_node]
    if len(CONNECTIVITY_GRAPH.subgraph(subg)) == 0:
        return -1
    emb = embedding.minorminer.find_embedding(
        complete, CONNECTIVITY_GRAPH.subgraph(subg), random_seed=random_state)

    c = 0
    while emb == {}:
        c += 1
        neighbors = []
        for node in subg:
            neighbors += list(CONNECTIVITY_GRAPH[node].keys())
        subg += neighbors
        subg = list(set(subg))
        if len(CONNECTIVITY_GRAPH.subgraph(subg)) == 0:
            return -1
        emb = embedding.minorminer.find_embedding(
            complete, CONNECTIVITY_GRAPH.subgraph(subg), random_seed=random_state)
        if c > 10:
            return -1
    possible_choices = [subg]
    for rep in range(1000):
        choice = random.choice(subg)
        subg.remove(choice)
        if len(CONNECTIVITY_GRAPH.subgraph(subg)) == 0:
            return -1
        emb = embedding.minorminer.find_embedding(
            complete, CONNECTIVITY_GRAPH.subgraph(subg), random_seed=random_state)
        if emb == {}:
            subg.append(choice)
        if emb != {}:
            possible_choices.append(subg)
    lengths = []
    for a in possible_choices:
        lengths.append(len(a))
    second_tier_choices = []
    top_choices = []
    for a in possible_choices:
        if len(a) == min(lengths):
            second_tier_choices.append(a)
            if starting_node in a:
                top_choices.append(a)
    if len(top_choices) == 0:
        return random.choice(second_tier_choices)
    if len(second_tier_choices) == 0:
        return -1
    return random.choice(top_choices)


def iterative_embedding(CONNECTIVITY_GRAPH, target_clique_size, random_state=42):
    """
    This heuristic method computes many disjoint embeddings for cliques of size
    target_clique_size onto CONNECTIVITY_GRAPH.

    Parameters
    ----------
    CONNECTIVITY_GRAPH : networkx.Graph()
        Quantum annealer hardware connectivity graph.
    target_clique_size : integer
        small clique we want to embed as many times as possible onto the hardware.
    random_state : integer, optional
        random seed. The default is 42.

    Returns
    -------
    all_subgraphs : list
        list of  the edges  of the subgraphs, each subgraph can embed a clique
        of size target_clique_size.

    """
    random.seed(random_state)

    all_subgraphs = []
    subg = iterative_search(CONNECTIVITY_GRAPH, random.choice(
        list(CONNECTIVITY_GRAPH.nodes())), target_clique_size)
    all_subgraphs.append(subg)
    new_starting_points = []
    for node in subg:
        ne = list(CONNECTIVITY_GRAPH[node].keys())
        for i in ne:
            if i not in subg:
                new_starting_points.append(i)
    for node in subg:
        CONNECTIVITY_GRAPH.remove_node(node)
    ref = [-1]
    while ref.count(max(ref)) < 1000:
        ref.append(len(all_subgraphs))
        starting_node = random.choice(new_starting_points)
        subg = iterative_search(CONNECTIVITY_GRAPH, starting_node, target_clique_size)
        if subg == -1:
            continue
        missing_node_indicator = True
        for node in subg:
            if CONNECTIVITY_GRAPH.has_node(node) != True:
                missing_node_indicator = False
        if missing_node_indicator is False:
            continue
        all_subgraphs.append(subg)
        for node in subg:
            ne = list(CONNECTIVITY_GRAPH[node].keys())
            for i in ne:
                if i not in subg:
                    new_starting_points.append(i)
        for node in subg:
            CONNECTIVITY_GRAPH.remove_node(node)
        for tmp in all_subgraphs:
            for node in tmp:
                if node in new_starting_points:
                    new_starting_points = remove_values_from_list(new_starting_points, node)
    new_starting_points = list(CONNECTIVITY_GRAPH.nodes())
    ref = [-1]
    while ref.count(max(ref)) < 1000:
        ref.append(len(all_subgraphs))
        starting_node = random.choice(new_starting_points)
        subg = iterative_search(CONNECTIVITY_GRAPH, starting_node, target_clique_size)
        if subg == -1:
            continue
        missing_node_indicator = True
        for node in subg:
            if CONNECTIVITY_GRAPH.has_node(node) != True:
                missing_node_indicator = False
        if missing_node_indicator == False:
            continue
        all_subgraphs.append(subg)
        for node in subg:
            CONNECTIVITY_GRAPH.remove_node(node)
        for tmp in all_subgraphs:
            for node in tmp:
                if node in new_starting_points:
                    new_starting_points = remove_values_from_list(new_starting_points, node)
    return all_subgraphs

def iterative_search_random_tries(hardware_connectivity, CLIQUE):
        all_embeddings = []
        for repetition in range(100):
                emb = embedding.minorminer.find_embedding(nx.complete_graph(CLIQUE), hardware_connectivity, tries=20, max_no_improvement=20, threads=10)
                if emb == {}:
                        #print("Failed")
                        continue
                used_qubits = []
                for a in emb:
                        used_qubits += emb[a]
                for qubit in used_qubits:
                        hardware_connectivity.remove_node(qubit)
                all_embeddings.append(emb)
        return all_embeddings
