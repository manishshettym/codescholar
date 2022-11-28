

def init_encoder_configs(parser, arg_str=None):
    enc_args = parser.add_argument_group()

    enc_args.add_argument('--agg_type', type=str,
                          help='type of aggregation/convolution')
    enc_args.add_argument('--batch_size', type=int,
                          help='Training batch size')
    enc_args.add_argument('--n_layers', type=int,
                          help='Number of graph conv layers')
    enc_args.add_argument('--hidden_dim', type=int,
                          help='Training hidden size')
    enc_args.add_argument('--skip', type=str,
                          help='"all" or "last"')
    enc_args.add_argument('--dropout', type=float,
                          help='Dropout rate')
    enc_args.add_argument('--n_iters', type=int,
                          help='Number of training iterations')
    enc_args.add_argument('--n_batches', type=int,
                          help='Number of training minibatches')
    enc_args.add_argument('--margin', type=float,
                          help='margin for loss')
    enc_args.add_argument('--dataset', type=str,
                          help='Dataset')
    enc_args.add_argument('--test_set', type=str,
                          help='test set filename')
    enc_args.add_argument('--eval_interval', type=int,
                          help='how often to eval during training')
    enc_args.add_argument('--val_size', type=int, help='validation set size')
    enc_args.add_argument('--model_path', type=str,
                          help='path to save/load model')
    enc_args.add_argument('--opt_scheduler', type=str, help='scheduler name')
    enc_args.add_argument('--node_anchored', action="store_true",
                          help='whether to use node anchoring in training')
    enc_args.add_argument('--test', action="store_true")
    enc_args.add_argument('--n_workers', type=int)
    enc_args.add_argument('--tag', type=str, help='tag to identify the run')

    enc_args.set_defaults(
        agg_type='GINE',
        dataset='pandas',
        n_layers=7,
        batch_size=64,
        hidden_dim=64,
        skip="learnable",
        dropout=0.0,
        n_iters=5,
        n_batches=10000,
        opt='adam',
        opt_scheduler='none',
        opt_restart=100,
        weight_decay=0.0,
        lr=1e-4,
        margin=0.1,
        test_set='',
        eval_interval=1000,
        n_workers=4,
        model_path="ckpt/model.pt",
        tag='',
        val_size=4096,
        node_anchored=True
    )


def init_optimizer_configs(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
                            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. default: none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                            help='Number of epochs before restart, default: 0')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step',
                            type=int, help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate',
                            type=float, help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
                            help='Optimizer weight decay.')
