
def init_search_configs(parser, arg_str=None):
    search_args = parser.add_argument_group()

    search_args.add_argument('--radius', type=int,
                             help='radius of node neighborhoods')
    search_args.add_argument(
        '--subgraph_sample_size', type=int,
        help='number of nodes to take from each neighborhood')
    search_args.add_argument(
        '--use_full_graphs', action="store_true",
        help="process whole graphs or sampled node neighborhoods")
    
    search_args.add_argument('--min_pattern_size', type=int)
    search_args.add_argument('--max_pattern_size', type=int)
    search_args.add_argument(
        '--n_trials', type=int,
        help='number of search trials to run')

    search_args.set_defaults(
        n_trials=1000,
        min_pattern_size=5,
        max_pattern_size=20,
        subgraph_sample_size=0,
        radius=3)
