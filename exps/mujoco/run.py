"""Script that demonstrates how to use BPTT using hallucination."""

import argparse
import importlib

import torch

torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.float32)

from rllib.environment import GymEnvironment
from rllib.model import TransformedModel
from rllib.util import set_random_seed
from rllib.util.training.agent_training import evaluate_agent, train_agent

from exps.util import parse_config_file
from hucrl.environment.hallucination_wrapper import HallucinationWrapper
from hucrl.model.hallucinated_model import HallucinatedModel


def main(args):
    """Run experiment."""
    set_random_seed(args.seed)
    env_config = parse_config_file(args.env_config_file)

    # _, environment = init_experiment(args)

    env_config["action_cost"] = env_config.get("action_cost", 0)
    environment = GymEnvironment(
        env_config["name"], ctrl_cost_weight=env_config["action_cost"], seed=args.seed
    )
    reward_model = environment.env.reward_model()

    if args.exploration == "optimistic":

        dynamical_model = HallucinatedModel.default(environment, beta=args.beta)
        environment.add_wrapper(HallucinationWrapper)
    else:
        dynamical_model = TransformedModel.default(environment)
    kwargs = parse_config_file(args.agent_config_file)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs['device'] = args.device

    dynamical_model = dynamical_model.to(args.device)

    env_version = args.env_config_file.split('/')[0]
    if env_version == 'exp':
        env_version = 'hucrl'

    comment = '-'.join((environment.name.replace('-','_') + '_' + env_version,
                        args.exploration,
                        f'ac{env_config["action_cost"]}',
                        f'b{args.beta}',
                        f'ep{args.train_episodes}',
                        f's{args.seed}'))

    agent = getattr(
        importlib.import_module("rllib.agent"), f"{args.agent}Agent"
    ).default(
        environment=environment,
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        thompson_sampling=args.exploration == "thompson",
        tensorboard=False,
        comment=comment,
        **kwargs,
    )

    agent.to(args.device)
    # agent.policy.to(torch.float32)

    train_agent(
        agent=agent,
        environment=environment,
        max_steps=env_config["max_steps"],
        num_episodes=args.train_episodes,
        render=args.render,
        print_frequency=1,
    )

    evaluate_agent(
        agent=agent,
        environment=environment,
        max_steps=env_config["max_steps"],
        num_episodes=args.test_episodes,
        render=args.render,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for H-UCRL.")
    parser.add_argument(
        "--agent",
        type=str,
        default="BPTT",
        choices=["BPTT", "MVE", "DataAugmentation", "MPC", "MBMPO", 'SAC'],
    )
    parser.add_argument("--agent-config-file", type=str, default="config/agents/bptt.yaml")
    parser.add_argument(
        "--env-config-file", type=str, default="config/envs/half-cheetah.yaml"
    )

    parser.add_argument(
        "--exploration",
        type=str,
        default="optimistic",
        choices=["optimistic", "expected", "thompson"],
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=250)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=1)

    parser.add_argument("--device", type=str, default='cuda')

    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument("--action-cost", type=float, default=0.1)

    parser.add_argument("--beta", type=float, default=1.0)
    main(parser.parse_args())
