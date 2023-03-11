
def init_search_configs(parser, arg_str=None):
    search_args = parser.add_argument_group()

    search_args.add_argument('--radius', type=int,
                             help='radius of node neighborhoods')
    search_args.add_argument(
        '--subgraph_sample_size', type=int,
        help='number of nodes to take from each neighborhood')
    search_args.add_argument('--min_idiom_size', type=int)
    search_args.add_argument('--max_idiom_size', type=int)
    search_args.add_argument(
        '--n_trials', type=int,
        help='number of search trials = #initial program nodes')
    search_args.add_argument(
        '--n_beams', type=int,
        help='number of beams to filter into after each round')
    search_args.add_argument(
        '--prog_samples', type=int,
        help='number of programs in the search space')
    search_args.add_argument(
        '--rank', type=int,
        help='number of most frequent idioms of a particular size')
    

    search_args.set_defaults(
        prog_samples=200000,
        n_trials=50000,
        n_beams=1,
        rank=30,
        min_idiom_size=10,
        max_idiom_size=35,
        subgraph_sample_size=10,
        radius=3)
