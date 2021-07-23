from model.lanegcn_ori.model import Net as OriLaneGCN


def get_model(args, hparams):
    if args.model == 'lanegcn_ori':
        model = OriLaneGCN(args, hparams)
    else:
        raise NotImplementedError
    return model