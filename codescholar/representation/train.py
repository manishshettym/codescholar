import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from deepsnap.batch import Batch

from codescholar.representation.test import validation
from codescholar.representation import models, config, dataset
from codescholar.utils.train_utils import (
    build_model, build_optimizer, get_device)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.multiprocessing.set_sharing_strategy('file_system')


def init_logger(args):
    log_keys = ["conv_type", "n_layers", "hidden_dim",
                "margin", "dataset", "max_graph_size", "skip"]
    log_str = ".".join(["{}={}".format(k, v)
                        for k, v in sorted(vars(args).items())
                        if k in log_keys])
    return SummaryWriter(comment=log_str)


def make_validation_set(dataloader):
    test_pts = []

    for batch in tqdm(dataloader, total=len(dataloader), desc="TestBatches"):
        pos_q, pos_t, neg_q, neg_t = zip(*batch)
        pos_q = Batch.from_data_list(pos_q)
        pos_t = Batch.from_data_list(pos_t)
        neg_q = Batch.from_data_list(neg_q)
        neg_t = Batch.from_data_list(neg_t)
        
        # if pos_q:
        #     pos_q = pos_q.to(torch.device("cpu"))
        #     pos_t = pos_t.to(torch.device("cpu"))
        # neg_q = neg_q.to(torch.device("cpu"))
        # neg_t = neg_t.to(torch.device("cpu"))
        
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
    clf_opt = optim.Adam(model.classifier.parameters(), lr=args.lr)

    done = False
    while not done:
        dataloader = corpus.gen_data_loader(args.batch_size, train=True)

        for batch in dataloader:
            msg, _ = in_queue.get()
            if msg == "done":
                done = True
                break

            model.train()
            model.zero_grad()

            pos_a, pos_b, neg_a, neg_b = zip(*batch)

            # convert to DeepSnap Batch
            pos_a = Batch.from_data_list(pos_a).to(get_device())
            pos_b = Batch.from_data_list(pos_b).to(get_device())
            neg_a = Batch.from_data_list(neg_a).to(get_device())
            neg_b = Batch.from_data_list(neg_b).to(get_device())

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
            clf_opt.step()
            
            # metrics
            pred = pred.argmax(dim=-1)
            acc = torch.mean((pred == labels).type(torch.float))
            train_loss = loss.item()
            train_acc = acc.item()

            out_queue.put(("step", (train_loss, train_acc)))


def start_workers(model, corpus, in_queue, out_queue, args):
    workers = []
    for _ in tqdm(range(args.n_workers), desc="Workers"):
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
    in_queue, out_queue = mp.Queue(), mp.Queue()

    # init logger
    logger = init_logger(args)

    # build model
    model = build_model(models.SubgraphEmbedder, args)
    # print(model)
    model.share_memory()

    print("Moving model to device:", get_device())
    model = model.to(get_device())

    # create a corpus for train and test
    corpus = dataset.Corpus(
        args.dataset, args.n_train, args.n_test,
        train=(not args.test))

    # create validation points
    loader = corpus.gen_data_loader(args.batch_size, train=False)
    validation_pts = make_validation_set(loader)

    # ====== TESTING ======
    if args.test:
        validation(args, model, validation_pts, logger, 0, 0)

    # ====== TRAINING ======
    else:
        workers = start_workers(model, corpus, in_queue, out_queue, args)

        batch_n = 0
        for epoch in range(args.n_batches // args.eval_interval):
            print(f"Epoch #{epoch}")

            for _ in range(args.eval_interval):
                in_queue.put(("step", None))
            
            # loop over #batches in an epoch
            for _ in range(args.eval_interval):
                _, result = out_queue.get()
                train_loss, train_acc = result
                print(f"Batch {batch_n}. Loss: {train_loss:.4f}. \
                    Train acc: {train_acc:.4f}\n")
                
                logger.add_scalar("Loss(train)", train_loss, batch_n)
                logger.add_scalar("Acc(train)", train_acc, batch_n)
                batch_n += 1

            # validation after an epoch
            validation(args, model, validation_pts, logger, batch_n, epoch)
    
        for _ in range(args.n_workers):
            in_queue.put(("done", None))
        for worker in workers:
            worker.join()


def main(testing=False):
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    args = parser.parse_args()

    if testing:
        args.test = True
    
    args.n_train = args.n_batches * args.batch_size
    args.n_test = int(0.2 * args.n_train)
    # args.n_train = 32000
    # args.n_test = 6400
    args.n_test = 10000

    train_loop(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
