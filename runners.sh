bash run_permutations.sh exps/mujoco/run.py \
--env-config \
rllib/examples/config/envs/cart-pole-mujoco.yaml \
rllib/examples/config/envs/inverted-pendulum.yaml \
--agent-config \
exps/mujoco/config/agents/bptt.yaml \
exps/mujoco/config/agents/data_augmentation.yaml \
 --seed 4 5 6 \
 --exploration optimistic expected thompson \
 --action-cost 0 \
 -o

bash run_permutations.sh rllib/examples/run.py \
SAC \
--env-config \
rllib/examples/config/envs/cart-pole-mujoco.yaml \
rllib/examples/config/envs/inverted-pendulum.yaml \
 --seed 1 2 3 \
 -o
