import numpy as np

def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade1, fde1, ade, fde, min_idcs


# batch_features, batch_labels, results, loss_dict, mode='eval'
# return lr, loss, cls loss, reg loss, ade1, fde1, ade, fde
def lanegcn_metric(batch_features, batch_labels, results, loss_dict, mode='eval'):

    cls = loss_dict["cls_loss"] / (loss_dict["num_cls"] + 1e-10)
    reg = loss_dict["reg_loss"] / (loss_dict["num_reg"] + 1e-10)
    loss = cls + reg

    preds = np.concatenate([x[0:1].detach().cpu().numpy() for x in results["reg"]], axis=0)
    gt_preds = np.concatenate([x[0:1].detach().cpu().numpy() for x in batch_labels["gt_preds"]], axis=0)
    has_preds = np.concatenate([x[0:1].detach().cpu().numpy() for x in batch_labels["has_preds"]], axis=0)

    ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

    return {
        'loss': loss,
        'cls': cls,
        'reg': reg,
        'ade1': ade1,
        'fde1': fde1,
        'ade': ade,
        'fde': fde,
    }
