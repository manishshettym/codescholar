from itertools import chain

import os
import yappi
import torch
import numpy as np
import networkx as nx
from deepsnap.batch import Batch
from transformers import RobertaTokenizer, RobertaModel

from codescholar.utils.search_utils import _frontier, read_graph, read_embeddings_batched_redis
from codescholar.representation import models
from codescholar.utils.train_utils import build_model, get_device, featurize_graph



def score_candidate_freq(args, model, embs, cand_emb, device_id=None):
    """score candidate program embedding against target program embeddings
    in a batched manner.

    Algorithm:
    softmax based classifier on top of emb-diff
        score = #nhoods !containing cand.
        = count(cand_emb - embs > 0)
            --> classifier: 0/1
    """
    score = 0

    if cand_emb.shape[0] > 1:
        preds = []

        for comp_emb in cand_emb:
            comp_preds = np.array([])
            for emb_batch in embs:
                with torch.no_grad():
                    is_subgraph_rel = model.predict((emb_batch.to(get_device(device_id)), comp_emb))
                    is_subgraph = model.classifier(is_subgraph_rel.unsqueeze(1))
                    comp_preds = np.concatenate((comp_preds, torch.argmax(is_subgraph, axis=1).cpu().numpy()))

            preds.append(comp_preds)

        assert len(set([len(pred) for pred in preds])) == 1, "component preds have different shapes!"

        preds = np.array(preds)
        merged_preds = np.all(preds, axis=0)
        score = np.sum(merged_preds)
    else:
        for emb_batch in embs:
            with torch.no_grad():
                is_subgraph_rel = model.predict((emb_batch.to(get_device(device_id)), cand_emb))
                is_subgraph = model.classifier(is_subgraph_rel.unsqueeze(1))
                score += torch.sum(torch.argmax(is_subgraph, axis=1)).item()

    return score


embs = None
feat_tokenizer = None
feat_model = None
subg_model = None
def init_grow(args, prog_indices, device_id=None):
    global embs, feat_tokenizer, feat_model, subg_model
    codebert_name = "microsoft/codebert-base"
    
    if embs is None:
        embs = read_embeddings_batched_redis(args, prog_indices)

    if feat_tokenizer is None:
        feat_tokenizer = RobertaTokenizer.from_pretrained(codebert_name)

    if feat_model is None:
        feat_model = RobertaModel.from_pretrained(codebert_name).to(get_device(device_id))
        feat_model.eval()

    if subg_model is None:
        subg_model = build_model(models.SubgraphEmbedder, args, device_id=device_id)
        subg_model.eval()
    
    return embs, feat_tokenizer, feat_model, subg_model

def grow(args, prog_indices, in_queue, out_queue, device_id=None):
    embs, feat_tokenizer, feat_model, model = init_grow(args, prog_indices, device_id=device_id)

    done = False
    while not done:
        msg, beam_set = in_queue.get()

        if msg == "done":
            del embs
            done = True
            break

        new_beams = []

        # STEP 1: Explore all candidates in each beam of the beam_set
        for beam in beam_set:
            _, holes, neigh, frontier, visited, graph_idx = beam
            graph = read_graph(args, graph_idx)

            if len(neigh) >= args.max_idiom_size or not frontier:
                continue

            cand_neighs = []

            # EMBED CANDIDATES
            for i, cand_node in enumerate(frontier):
                cand_neigh = graph.subgraph(neigh + [cand_node])
                connected_comps = list(nx.connected_components(cand_neigh.to_undirected()))

                if len(connected_comps) == 1:
                    cand_neigh = featurize_graph(cand_neigh, feat_tokenizer, feat_model, anchor=neigh[0], device_id=device_id)
                    cand_neighs.append(cand_neigh)
                else:
                    comp_neighs = []
                    for comp in connected_comps:
                        comp_neigh = cand_neigh.subgraph(comp)
                        comp_root = [n for n in comp_neigh.nodes if comp_neigh.in_degree(n) == 0][0]
                        comp_neigh = featurize_graph(comp_neigh, feat_tokenizer, feat_model, anchor=comp_root, device_id=device_id)
                        comp_neighs.append(comp_neigh)

                    cand_neighs.append(comp_neighs)

            flat_cand_neighs = list(chain.from_iterable([x if isinstance(x, list) else [x] for x in cand_neighs]))
            cand_batch = Batch.from_data_list(flat_cand_neighs).to(get_device(device_id))

            with torch.no_grad():
                cand_embs = model.encoder(cand_batch)

            # SCORE CANDIDATES (freq)
            for i, (cand_node, cand_neigh) in enumerate(zip(frontier, cand_neighs)):
                if isinstance(cand_neigh, list):
                    cand_emb = cand_embs[i : i + len(cand_neigh)]
                else:
                    cand_emb = cand_embs[i : i + 1]

                # first, add new holes introduced
                # then, remove hole filled in/by cand_node (incoming/outgoing edge resp)
                new_holes = holes + graph.nodes[cand_node]["span"].count("#") - 1

                # filter out candidates that exceed max_holes
                if new_holes < 0 or new_holes > args.max_holes:
                    continue

                new_neigh = neigh + [cand_node]

                # new frontier = {prev frontier} U {outgoing and incoming neighbors of cand_node} - {visited}
                # note: one can use type='neigh' to add only outgoing neighbors
                new_frontier = list(((set(frontier) | _frontier(graph, cand_node, type="radial")) - visited) - set([cand_node]))

                new_visited = visited | set([cand_node])

                score = score_candidate_freq(args, model, embs, cand_emb, device_id=device_id)
                new_beams.append((score, new_holes, new_neigh, new_frontier, new_visited, graph_idx))

        # STEP 2: Sort new beams by freq_score/#holes
        new_beams = list(sorted(new_beams, key=lambda x: x[0] / x[1] if x[1] > 0 else x[0], reverse=True))

        # print("===== [debugger] new beams =====")
        # for beam in new_beams:
        #     print("freq: ", beam[0])
        #     print("holes: ", beam[1])
        #     print("nodes: ", [graph.nodes[n]['span'] for n in beam[2]])
        #     print("frontier: ", [graph.nodes[n]['span'] for n in beam[3]])
        #     print()
        # print("================================")

        # STEP 3: filter top-k beams
        new_beams = new_beams[: args.n_beams]
        out_queue.put(("complete", new_beams))