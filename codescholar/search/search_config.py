
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
    search_args.add_argument(
        '--mode', type=str, choices=['g',  'm'],
        help='mode in which to run the search agent; m: mine, g: graph (def)'
    )
    search_args.add_argument(
        '--keywords', type=str, nargs='+',
        help='keywords to start idiomatic search in keyword mode'
    )
    search_args.add_argument(
        '--seed', type=str,
        help='graph to start idiomatic search in graph mode'
    )
    search_args.add_argument(
        '--max_holes', type=int,
        help='maximum number of holes that can be added to a candidate program'
    )

    search_args.set_defaults(
        mode='g',
        prog_samples=100000,
        n_trials=10,
        n_beams=1,
        rank=30,
        max_holes=8,
        min_idiom_size=3,
        max_idiom_size=30,
        subgraph_sample_size=10,
        radius=3)
