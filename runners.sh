bash run_permutations.sh exps/mujoco/run.py \
--env-config \
rllib/examples/config/envs/inverted-pendulum.yaml \
exps/mujoco/config/envs/ant.yaml \
exps/mujoco/config/envs/walker2d.yaml \
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
 --seed 4 5 6 \

