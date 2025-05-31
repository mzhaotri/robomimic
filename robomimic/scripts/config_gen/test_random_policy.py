from robomimic.scripts.config_gen.config_gen_utils import *
import json


def make_random_generator_helper(args):
    ckpt_path = args.ckpt

    # get ckpt config file, infer algo name from path
    ckpt_is_dir = os.path.isdir(os.path.expanduser(ckpt_path))
    
    if ckpt_is_dir:
        ckpt_config_path = os.path.join(ckpt_path, "config.json")
    else:
        ckpt_config_path = os.path.join(os.path.dirname(ckpt_path), "../config.json")
    
    with open(ckpt_config_path) as f:
        ckpt_config = json.load(f)
    algo_name_short = ckpt_path.split("/")[-6]

    generator = get_generator(
        algo_name=ckpt_config["algo_name"],
        config_file=ckpt_config_path,
        args=args,
        algo_name_short=algo_name_short,
    )


    
    # set up configs for running evals (do not need to change these lines)
    generator.add_param(
        key="experiment.ckpt_path",
        name="ckpt",
        group=1,
        values=[ckpt_path],
        hidename=True,
    )
    generator.add_param(
        key="experiment.rollout.enabled",
        name="",
        group=-1,
        values=[True],
    )

    generator.add_param(
            key="experiment.rollout.rate",
            name="",
            group=-1,
            values=[200],
        )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to model checkpoint (must be *.pth file)",
        required=True,
    )

    args = parser.parse_args()

    # Make generator creates the environment, policy, etc
    make_generator(args, make_random_generator_helper, skip_helpers=("env", "mod"), extra_flags="--eval_only")
