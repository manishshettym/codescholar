from functools import reduce
from tqdm import tqdm
import random

from networkx.algorithms.isomorphism import DiGraphMatcher

from codescholar.sast.simplified_ast import get_simplified_ast
from codescholar.sast.sast_utils import remove_node
from codescholar.utils.graph_utils import program_graph_to_nx
from codescholar.utils.search_utils import read_graph, _frontier, _reduce
from codescholar.utils.perf import perftimer

########### IDIOM MINING ############


def init_search_m(args, prog_indices):
    ps = []
    for idx in tqdm(prog_indices, desc="[init_search]"):
        g = read_graph(args, idx)
        ps.append(len(g))
        del g

    ps = np.array(ps, dtype=float)
    ps /= np.sum(ps)
    graph_dist = stats.rv_discrete(values=(np.arange(len(ps)), ps))

    beam_sets = []
    for trial in range(args.n_trials):
        graph_idx = np.arange(len(ps))[graph_dist.rvs()]
        graph_idx = prog_indices[graph_idx]

        graph = read_graph(args, graph_idx)  # TODO: convert to undirected?
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]

        # TODO: convert to undirected search like --mode g
        # find frontier = {neighbors} - {itself} = {supergraphs}
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])

        beam_sets.append([(0, 0, neigh, frontier, visited, graph_idx)])

    return beam_sets


########### SINGLE API SEARCH ############


@perftimer
def init_search_q(args, prog_indices, seed):
    beam_sets = []
    count = 0

    # generate seed graph for query
    seed_sast = get_simplified_ast(seed)
    if seed_sast is None:
        raise ValueError("Seed program is invalid!")

    module_nid = list(seed_sast.get_ast_nodes_of_type("Module"))[0].id
    remove_node(seed_sast, module_nid)

    seed_graph = program_graph_to_nx(seed_sast, directed=True)

    for idx in tqdm(prog_indices, desc="[init_search]"):
        if count >= args.max_init_beams:
            continue

        graph = read_graph(args, idx)

        # find all matches of the seed graph in the program graph
        # uses exact subgraph isomorphism - not that expensive because query is small (2-3 nodes)
        node_match = lambda n1, n2: n1["span"] == n2["span"] and n1["ast_type"] == n2["ast_type"]
        DiGM = DiGraphMatcher(graph, seed_graph, node_match=node_match)
        seed_matches = list(DiGM.subgraph_isomorphisms_iter())

        # no matches
        if len(seed_matches) == 0:
            continue

        # randomly select one of the matches as the starting point
        neigh = list(random.choice(seed_matches).keys())

        # find frontier = {successors} U {predecessors} - {itself} = {supergraphs}
        frontier = set(_reduce(list(_frontier(graph, n, type="radial") for n in neigh))) - set(neigh)
        visited = set(neigh)

        beam_sets.append([(0, 0, neigh, frontier, visited, idx)])
        count += 1

    return beam_sets


########### MULTI API SEARCH ############


@perftimer
def init_search_mq(args, prog_indices, seeds):
    beam_sets = []
    count = 0
    seed_graphs = []

    # generate seed graph for query
    seed_sasts = [get_simplified_ast(s) for s in seeds]
    if seed_sasts is []:
        raise ValueError("Seed programs are invalid!")

    for seed_sast in seed_sasts:
        module_nid = list(seed_sast.get_ast_nodes_of_type("Module"))[0].id
        remove_node(seed_sast, module_nid)
        seed_graph = program_graph_to_nx(seed_sast, directed=True)
        seed_graphs.append(seed_graph)

    def match_seed_graph(seed_graph, target_graph) -> list[dict]:
        """find all matches of the seed graph in the target graph"""
        # uses exact subgraph isomorphism - not that expensive because query is small (2-3 nodes)
        node_match = lambda n1, n2: n1["span"] == n2["span"] and n1["ast_type"] == n2["ast_type"]
        DiGM = DiGraphMatcher(target_graph, seed_graph, node_match=node_match)
        return list(DiGM.subgraph_isomorphisms_iter())

    for idx in tqdm(prog_indices, desc="[init_search]"):
        if count >= args.max_init_beams:
            continue

        graph = read_graph(args, idx)
        no_match_flag = False
        all_seed_matches = []

        for seed_graph in seed_graphs:
            seed_matches = match_seed_graph(seed_graph, graph)

            # > 1 match: choose one
            if len(seed_matches) > 1:
                all_seed_matches.append(random.choice(seed_matches))

            # no match for this seed: skip program
            elif len(seed_matches) == 0:
                no_match_flag = True
                break

            # 1 match: add to list
            else:
                all_seed_matches += seed_matches

        if no_match_flag:
            continue

        neigh = reduce(lambda a, b: {**a, **b}, all_seed_matches)
        neigh = list(neigh.keys())

        # find frontier = {successors} U {predecessors} - {itself} = {supergraphs}
        frontier = set(_reduce(list(_frontier(graph, n, type="radial") for n in neigh))) - set(neigh)
        visited = set(neigh)

        beam_sets.append([(0, 0, neigh, frontier, visited, idx)])
        count += 1

    return beam_sets
