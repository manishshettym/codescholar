from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from deepsnap.batch import Batch


from codescholar.utils.train_utils import get_device


def precision(pred, labels):
    if torch.sum(pred) > 0:
        return torch.sum(pred * labels).item() / torch.sum(pred).item()
    else:
        return float("NaN")


def recall(pred, labels):
    if torch.sum(labels) > 0:
        return torch.sum(pred * labels).item() / torch.sum(labels).item()
    else:
        return float("NaN")


def test(args, model, dataloader, logger):
    model.eval()
    all_raw_preds, all_preds, all_labels = [], [], []
    
    for batch in tqdm(dataloader, total=len(dataloader), desc="TestBatches"):
        pos_a, pos_b, neg_a, neg_b = zip(*batch)
        pos_a = Batch.from_data_list(pos_a)
        pos_b = Batch.from_data_list(pos_b)
        neg_a = Batch.from_data_list(neg_a)
        neg_b = Batch.from_data_list(neg_b)

        if pos_a:
            pos_a = pos_a.to(get_device())
            pos_b = pos_b.to(get_device())
        neg_a = neg_a.to(get_device())
        neg_b = neg_b.to(get_device())

        labels = torch.tensor(
            [1] * (pos_a.num_graphs if pos_a else 0) +
            [0] * neg_a.num_graphs
        ).to(get_device())

        with torch.no_grad():
            # forward pass through GNN layers
            emb_neg_a, emb_neg_b = (
                model.encoder(neg_a),
                model.encoder(neg_b))
            if pos_a:
                emb_pos_a, emb_pos_b = (
                    model.encoder(pos_a),
                    model.encoder(pos_b))
                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
            else:
                emb_as, emb_bs = emb_neg_a, emb_neg_b
        
            # prediction from GNN
            pred = model(emb_as, emb_bs)
            raw_pred = model.predict(pred)
            
            # prediction from classifier
            pred = model.classifier(raw_pred.unsqueeze(1)).argmax(dim=-1)

        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
        all_labels.append(labels)

    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)

    # metrics
    acc = torch.mean((pred == labels).type(torch.float))
    prec = precision(pred, labels)
    rec = recall(pred, labels)

    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    auroc = roc_auc_score(labels, pred)
    avg_prec = average_precision_score(labels, pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()

    print("\n{}".format(str(datetime.now())))
    print(
        "Test. Count: {}. Acc: {:.4f}. "
        "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}. AP: {:.4f}.\n     "
        "TN: {}. FP: {}. FN: {}. TP: {}".format(
            len(pred), acc, prec, rec, auroc, avg_prec,
            tn, fp, fn, tp))


def validation(args, model, test_pts, logger, batch_n, epoch):
    model.eval()
    all_raw_preds, all_preds, all_labels = [], [], []

    for pos_a, pos_b, neg_a, neg_b in test_pts:
        if pos_a:
            pos_a = pos_a.to(get_device())
            pos_b = pos_b.to(get_device())
        neg_a = neg_a.to(get_device())
        neg_b = neg_b.to(get_device())

        labels = torch.tensor(
            [1] * (pos_a.num_graphs if pos_a else 0) +
            [0] * neg_a.num_graphs
        ).to(get_device())

        with torch.no_grad():
            # forward pass through GNN layers
            emb_neg_a, emb_neg_b = (
                model.encoder(neg_a),
                model.encoder(neg_b))
            if pos_a:
                emb_pos_a, emb_pos_b = (
                    model.encoder(pos_a),
                    model.encoder(pos_b))
                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
            else:
                emb_as, emb_bs = emb_neg_a, emb_neg_b
        
            # prediction from GNN
            pred = model(emb_as, emb_bs)
            raw_pred = model.predict(pred)
            
            # prediction from classifier
            pred = model.classifier(raw_pred.unsqueeze(1)).argmax(dim=-1)

        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
        all_labels.append(labels)

    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)

    # metrics
    acc = torch.mean((pred == labels).type(torch.float))
    prec = precision(pred, labels)
    rec = recall(pred, labels)

    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    auroc = roc_auc_score(labels, pred)
    avg_prec = average_precision_score(labels, pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()

    print("\n{}".format(str(datetime.now())))
    print(
        "Validation. Epoch {}. Count: {}. Acc: {:.4f}. "
        "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}. AP: {:.4f}.\n     "
        "TN: {}. FP: {}. FN: {}. TP: {}".format(
            epoch, len(pred), acc, prec, rec, auroc, avg_prec,
            tn, fp, fn, tp))
    
    if not args.test:
        logger.add_scalar("Accuracy/test", acc, batch_n)
        logger.add_scalar("Precision/test", prec, batch_n)
        logger.add_scalar("Recall/test", rec, batch_n)
        logger.add_scalar("AUROC/test", auroc, batch_n)
        logger.add_scalar("AvgPrec/test", avg_prec, batch_n)
        logger.add_scalar("TP/test", tp, batch_n)
        logger.add_scalar("TN/test", tn, batch_n)
        logger.add_scalar("FP/test", fp, batch_n)
        logger.add_scalar("FN/test", fn, batch_n)
        print("Saving {}".format(args.model_path))
        torch.save(model.state_dict(), args.model_path)
