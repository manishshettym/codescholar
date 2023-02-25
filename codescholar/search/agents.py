import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import networkx as nx
from deepsnap.batch import Batch
import scipy.stats as stats
import matplotlib.pyplot as plt

from codescholar.utils.train_utils import get_device, featurize_graph
from codescholar.utils.search_utils import wl_hash


class SearchAgent:
    """Class for search strategies to identify frequent subgraphs
    by walking the embedding space.
    Steps:
    1. Choose a seed node.
    2. Chooses a node to connect to the existing subgraph pattern,
        increasing pattern size by 1
    """
    def __init__(
            self, min_idiom_size, max_idiom_size,
            model, dataset, embs, out_batch_size=20):

        self.min_idiom_size = min_idiom_size
        self.max_idiom_size = max_idiom_size
        self.model = model
        self.dataset = dataset
        self.embs = embs
        self.out_batch_size = out_batch_size

        # size (#nodes) -> List[(score, graph)]
        self.cand_patterns = defaultdict(list)

        # size (#nodes) -> hash(graph) -> List[graph]
        self.idiommine = defaultdict(lambda: defaultdict(list))

    def search(self):
        '''Abstract method to run the search procedure'''
        raise NotImplementedError

    def init_search():
        '''Abstract method to initialize search with seed nodes'''
        raise NotImplementedError

    def grow(self):
        """Abstract method for executing a search step.
        Every step adds a new node to the subgraph pattern.
        """
        raise NotImplementedError
    
    def finish_search(self):
        raise NotImplementedError
    
    def is_search_done(self):
        """Condition to stop search
        """
        raise NotImplemented
    
    def _print_mine(self):
        print("========== CODESCHOLAR MINE ==========")
        print(".")
        for size, hashed_idioms in self.idiommine.items():
            print(f"├── size {size}")
            fin_idx = len(hashed_idioms.keys()) - 1

            for idx, (hash_id, idioms) in enumerate(hashed_idioms.items()):
                if idx == fin_idx:
                    print(f"    └── [{idx}] {len(idioms)} idiom(s)")
                else:
                    print(f"    ├── [{idx}] {len(idioms)} idiom(s)")
        print("==========+================+==========")


class GreedySearch(SearchAgent):
    def __init__(
            self, min_idiom_size, max_idiom_size,
            model, dataset, embs, n_beams=1, out_batch_size=20):
        super().__init__(
            min_idiom_size, max_idiom_size,
            model, dataset, embs, out_batch_size)

        self.n_beams = n_beams
        self.beam_sets = None
    
    def search(self, n_trials):
        self.n_trials = n_trials
        self.init_search()
            
        while not self.is_search_done():
            self.grow()
            self._print_mine()
        
        return self.finish_search()

    def init_search(self):
        ps = np.array([len(g) for g in self.dataset], dtype=np.float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        beams = []
        for trial in range(self.n_trials):
            # choose random graph
            # TODO @manish: choose interesting programs instead?
            graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
            graph = self.dataset[graph_idx]

            # choose random start node
            # TODO @manish: choose interesting node instead?
            start_node = random.choice(list(graph.nodes))
            neigh = [start_node]
            
            # find frontier = {neighbors} - {itself} = {supergraphs}
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])

            beams.append([(0, neigh, frontier, visited, graph_idx)])

        # NOTE: seeds can come from the same graph
        self.beam_sets = beams
    
    def is_search_done(self):
        return len(self.beam_sets) == 0
    
    def grow(self):
        new_beam_sets = []
        # dist_graphs = len(set(b[0][-1] for b in self.beam_sets))
                
        for beam_set in tqdm(self.beam_sets):
            new_beams = []

            # STEP 1: Explore all candidate nodes in the beam_set
            for beam in beam_set:
                _, neigh, frontier, visited, graph_idx = beam
                graph = self.dataset[graph_idx]

                if len(neigh) >= self.max_idiom_size or not frontier:
                    continue

                cand_neighs = []

                # EMBED CANDIDATES
                # candidates := neighbors of the current node
                for cand_node in frontier:
                    cand_neigh = graph.subgraph(neigh + [cand_node])
                    cand_neigh = featurize_graph(cand_neigh, neigh[0])
                    cand_neighs.append(cand_neigh)
                
                cand_batch = Batch.from_data_list(cand_neighs).to(get_device())
                cand_embs = self.model.encoder(cand_batch)
                    
                # SCORE CANDIDATES
                for cand_node, cand_emb in zip(frontier, cand_embs):
                    score, n_embs = 0, 0

                    for emb_batch in self.embs:
                        n_embs += len(emb_batch)

                        ''' NOTE: score = total_violation :=
                            #neighborhoods not containing the pattern.
                        '''
                        score -= torch.sum(torch.argmax(
                            self.model.classifier(
                                self.model.predict((
                                    emb_batch.to(get_device()),
                                    cand_emb)).unsqueeze(1)), axis=1)).item()

                    new_neigh = neigh + [cand_node]
                    new_frontier = list(((
                        set(frontier) | set(graph.neighbors(cand_node)))
                        - visited) - set([cand_node]))
                    new_visited = visited | set([cand_node])
                    new_beams.append((
                        score, new_neigh, new_frontier,
                        new_visited, graph_idx))

            # STEP 2: Sort new beams by score (total_violation)
            new_beams = list(sorted(
                new_beams, key=lambda x: x[0]))[:self.n_beams]
            
            # STEP 3: Select candidates from the top scoring beam
            for new_beam in new_beams[:1]:
                score, neigh, frontier, visited, graph_idx = new_beam
                graph = self.dataset[graph_idx]

                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))

                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                
                # update the results
                self.cand_patterns[len(neigh_g)].append((score, neigh_g))
                self.idiommine[len(neigh_g)][wl_hash(neigh_g)].append(neigh_g)

            if len(new_beams) > 0:
                new_beam_sets.append(new_beams)

        self.beam_sets = new_beam_sets

    def finish_search(self):        
        cand_patterns_uniq = []
        for idiom_size in range(
                self.min_idiom_size,
                self.max_idiom_size + 1):

            hashed_idioms = self.idiommine[idiom_size].items()
            hashed_idioms = list(sorted(
                hashed_idioms, key=lambda x: len(x[1]), reverse=True))

            for _, idioms in hashed_idioms[:self.out_batch_size]:
                # choose any one because they all map to the same hash
                # cand_patterns_uniq.append(random.choice(idioms))
                for idiom in idioms:
                    cand_patterns_uniq.append(idiom)

        return cand_patterns_uniq
