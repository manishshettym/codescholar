import os
import argparse

import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from codescholar.representation import models, config, dataset
from codescholar.utils.train_utils import (
    build_model, build_optimizer, get_device)


def init_logger(args):
    log_keys = ["conv_type", "n_layers", "hidden_dim",
                "margin", "dataset", "max_graph_size", "skip"]
    log_str = ".".join(["{}={}".format(k, v)
                        for k, v in sorted(vars(args).items())
                        if k in log_keys])
    return SummaryWriter(comment=log_str)


def get_corpus(args):
    corpus = dataset.Corpus(
        args.dataset,
        node_anchored=args.node_anchored
    )

    return corpus


def make_validation_set(corpus, loaders):
    test_pts = []

    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        pos_q, pos_t, neg_q, neg_t = corpus.gen_batch(
            batch_target,
            batch_neg_target,
            batch_neg_query, False)
        
        if pos_q:
            pos_q = pos_q.to(torch.device("cpu"))
            pos_t = pos_t.to(torch.device("cpu"))
        neg_q = neg_q.to(torch.device("cpu"))
        neg_t = neg_t.to(torch.device("cpu"))
        
        test_pts.append((pos_q, pos_t, neg_q, neg_t))
    
    return test_pts


def train(args, model, corpus, in_queue, out_queue):
    """
    Train the model that was initialized
    
    Args:
        args: Commandline arguments
        model: GNN model
        corpus: dataset to train the GNN model
        in_queue: input queue to an intersection computation worker
        out_queue: output queue to an intersection computation worker
    """
    scheduler, opt = build_optimizer(args, model.parameters())
    opt = optim.Adam(model.classifier.parameters(), lr=args.lr)

    done = False
    while not done:
        # corpus = get_corpus(args)
        loaders = corpus.gen_data_loaders(
            args.eval_interval * args.batch_size,
            args.batch_size)
        
        for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
            msg, _ = in_queue.get()
            if msg == "done":
                done = True
                break

            # train
            model.train()
            model.zero_grad()

            # generate a positive and negative pair
            pos_a, pos_b, neg_a, neg_b = corpus.gen_batch(
                batch_target, batch_neg_target, batch_neg_query, True)
            
            # get embeddings
            emb_pos_a = model.encoder(pos_a)  # pos target
            emb_pos_b = model.encoder(pos_b)  # pos query
            emb_neg_a = model.encoder(neg_a)  # neg target
            emb_neg_b = model.encoder(neg_b)  # neg query

            # concatenate
            emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
            emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)

            labels = torch.tensor(
                [1] * pos_a.num_graphs
                + [0] * neg_a.num_graphs).to(get_device())
            
            # make predictions
            pred = model(emb_as, emb_bs)
            loss = model.criterion(pred, labels)

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if scheduler:
                scheduler.step()
            
            with torch.no_grad():
                pred = model.predict(pred)

            model.classifier.zero_grad()
            pred = model.classifier(pred.unsqueeze(1))
            criterion = nn.NLLLoss()
            clf_loss = criterion(pred, labels)
            clf_loss.backward()
            opt.step()
            
            # metrics
            pred = pred.argmax(dim=-1)
            acc = torch.mean((pred == labels).type(torch.float))
            train_loss = loss.item()
            train_acc = acc.item()

            out_queue.put(("step", (train_loss, train_acc)))


def start_workers(model, corpus, in_queue, out_queue, args):
    workers = []
    for i in range(args.n_workers):
        worker = mp.Process(
            target=train,
            args=(args, model, corpus, in_queue, out_queue)
        )
        worker.start()
        workers.append(worker)
    
    return workers


def train_loop(args):
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    
    print("Using dataset {}".format(args.dataset))
    print("Starting {} workers".format(args.n_workers))
    in_queue, out_queue = mp.Queue(), mp.Queue()

    # init logger
    logger = init_logger(args)

    # build model
    model = build_model(models.SubgraphEmbedder, args)
    print(model)
    model.share_memory()

    # prepare data source
    corpus = get_corpus(args)
    
    # loaders = corpus.gen_data_loaders(
    #     args.eval_interval * args.batch_size,
    #     args.batch_size)
    # validation_pts = make_validation_set(corpus, loaders)

    workers = start_workers(model, corpus, in_queue, out_queue, args)

    # ====== TRAINING ======
    batch_n = 0
    for epoch in range(args.n_batches // args.eval_interval):
        print(f"Epoch #{epoch}")

        for _ in range(args.eval_interval):
            in_queue.put(("step", None))
        
        for _ in range(args.eval_interval):
            _, result = out_queue.get()
            train_loss, train_acc = result
            print(f"Batch {batch_n}. Loss: {train_loss:.4f}. \
                Train acc: {train_acc:.4f}\n")
            
            logger.add_scalar("Loss(train)", train_loss, batch_n)
            logger.add_scalar("Acc(train)", train_acc, batch_n)
            batch_n += 1
                
        # TODO: call validation on validation_pts?

    for _ in range(args.n_workers):
        in_queue.put(("done", None))
    for worker in workers:
        worker.join()


def main():
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    args = parser.parse_args()

    train_loop(args)


if __name__ == "__main__":
    main()
