def init_search_configs(parser, arg_str=None):
    search_args = parser.add_argument_group()

    ########## EMBED SPACE CONFIGS ##########
    search_args.add_argument(
        "--num_neighborhoods", type=int, help="top neighborhoods to fetch sorted by degree of anchor; use `None` for all."
    )
    search_args.add_argument("--radius", type=int, help="max radius (hops) of node neighborhoods")
    search_args.add_argument("--neighborhood_size", type=int, help="number of nodes to sample from each neighborhood")

    ########## SEARCH CONFIGS ##########

    search_args.add_argument("--min_idiom_size", type=int)
    search_args.add_argument("--max_idiom_size", type=int)
    search_args.add_argument("--n_trials", type=int, help="number of search trials = #initial program nodes")
    search_args.add_argument("--n_beams", type=int, help="number of beams to filter into after each round")
    search_args.add_argument("--prog_samples", type=int, help="number of programs in the search space; -1 for all")
    search_args.add_argument(
        "--mode", type=str, choices=["q", "m", "mq"], help="mode in which to run the search agent; m: mine, g: query (def), mq: multi-query"
    )
    search_args.add_argument(
        "--seed", type=str, help="query to search in query/multi-query mode; for multi-query mode, use semi-colon separated seeds"
    )
    search_args.add_argument("--max_holes", type=int, help="maximum number of holes that can be added to a candidate program")
    search_args.add_argument("--max_init_beams", type=int, help="maximum number of beams to filter into during initialization of search")
    search_args.add_argument(
        "--stop_at_equilibrium", action="store_true", help="stop search when we reach equilibrium: diversity = reusability"
    )
    search_args.add_argument("--render", action="store_true", help="render the search graph")

    search_args.set_defaults(
        # embed_space configs
        num_neighborhoods=10,
        neighborhood_size=10,
        radius=3,
        # search configs
        mode="q",
        prog_samples=-1,
        n_trials=10,
        n_beams=1,
        max_holes=8,
        min_idiom_size=3,
        max_idiom_size=30,
        max_init_beams=200,
    )
