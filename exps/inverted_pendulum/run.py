"""Run the inverted-pendulum using MB-MPO."""
from dotmap import DotMap

from exps.inverted_pendulum import (
    ENVIRONMENT_MAX_STEPS,
    TRAIN_EPISODES,
    get_agent_and_environment,
)
from exps.inverted_pendulum.plotters import (
    plot_pendulum_trajectories,
    set_figure_params,
)
from exps.inverted_pendulum.util import get_mbmpo_parser
from exps.util import train_and_evaluate


PLAN_HORIZON, SIM_TRAJECTORIES = 8, 16
if __name__ == "__main__":

    parser = get_mbmpo_parser()
    parser.description = "Run Swing-up Inverted Pendulum using Model-Based MPO."
    parser.set_defaults(
        train_episodes=TRAIN_EPISODES,
        environment_max_steps=ENVIRONMENT_MAX_STEPS,
        plan_horizon=PLAN_HORIZON,
        sim_num_steps=ENVIRONMENT_MAX_STEPS,
        sim_initial_states_num_trajectories=SIM_TRAJECTORIES // 2,
        sim_initial_dist_num_trajectories=SIM_TRAJECTORIES // 2,
        model_kind="ProbabilisticEnsemble",
        model_learn_num_iter=50,
        model_opt_lr=1e-3,
    )

    args = parser.parse_args()
    args.exp_name = \
        '-'.join([
            args.agent,
            args.exploration,
            f'b{args.beta}'
        ])

    params = DotMap(vars(args))

    environment, agent = get_agent_and_environment(params)
    # agent.to('cuda')

    set_figure_params(serif=True, fontsize=9)
    train_and_evaluate(
        agent, environment, params, plot_callbacks=[plot_pendulum_trajectories]
    )