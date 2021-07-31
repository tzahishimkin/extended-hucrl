#!/bin/bash
# usage:
# run_permutations.sh <script-name> *<--field list-of-values>
# optional: -o to print the command without running

# example:
on_linuxs=true # bash run_permutations.sh ensemble_exp.py --env-config exps/mujoco/config/envs/half-cheetah.yaml --agent-config exps/mujoco/config/agents/bptt.yaml --seed 1 2 3
# bash run_permutations.sh exps/inverted_pendulum/run.py  --env-config exps/mujoco/config/envs/ant.yaml exps/mujoco/config/envs/walker2d.yaml --seed 1 2 3 -o
# bash run_permutations.sh exps/mujoco/run.py --agent MPC --agent-config exps/mujoco/config/agents/mpc.yaml --env-config rllib/examples/config/envs/cart-pole-mujoco.yaml rllib/examples/config/envs/inverted-pendulum.yaml --seed 1 2 3  --exploration optimistic expected thompson
# bash run_permutations.sh exps/mujoco/run.py --agent-config exps/mujoco/config/agents/bptt.yaml exps/mujoco/config/agents/data_augmentation.yaml --env-config rllib/examples/config/envs/cart-pole-mujoco.yaml rllib/examples/config/envs/inverted-pendulum.yaml --seed 1 2 3  --exploration optimistic expected thompson  --action-cost 0

simultanious_tasks=false
cuda_i=0
python_script="${1}"
variables="${*:2}"

if [[ "$variables" == *" -o"* ]]; then
  echo "print only:"
  variables=${variables%-o}
  only_print=true
else
  echo "print and run:"
  only_print=false
fi


if [ ${simultanious_tasks} = true ]; then
  postfix_command=' &'
else
  postfix_command=''
fi


run_simulation()
{
     func_args="${1}"
      cuda=$((cuda_i % GPU_AMOUNT))
      cuda_i=$((cuda_i + 1))
      if [ ${on_linuxs} = true ]; then
        prefix_command="CUDA_VISIBLE_DEVICES=${cuda} TS_SOCKET=/tmp/socket-cuda${cuda} tsp nohup"
      else
        prefix_command=''
      fi

      command="
          ${prefix_command}
          python -u ${python_script} ${func_args} ${postfix_command}
          > temp.txt "

      if [ ${only_print} = true ]; then
        echo $command
      else
        echo $command
        eval $command
      fi
}


if [ $HOSTNAME = 'linux2a' ]; then
  GPU_AMOUNT=2
elif [ $HOSTNAME = 'linux3' ]; then
  GPU_AMOUNT=4
elif [ $HOSTNAME = 'linux4' ]; then
  GPU_AMOUNT=2
elif [ $HOSTNAME = 'naama-server1' ]; then
  GPU_AMOUNT=4
elif [ $HOSTNAME = 'tzahi-X556UA' ]; then
  GPU_AMOUNT=1
elif [ $HOSTNAME = 'LAPTOP-1E8VE886' ]; then
  GPU_AMOUNT=1
fi

if [ ${on_linuxs} = true ]; then
  for ((i = 0; i < $GPU_AMOUNT; i += 1)); do
    TS_SOCKET=/tmp/socket-cuda$i tsp -S 1
  done
fi

isim=0
python run_permut_utils.py ${variables} | while read command_arg; do
    isim=$((isim + 1))
    run_simulation "${command_arg}"
done

if [ ${on_linuxs} = true ]; then
    echo "opened ${GPU_AMOUNT} sockets:"
    for ((i = 0; i < $GPU_AMOUNT; i += 1)); do
      echo TS_SOCKET=/tmp/socket-cuda$i tsp
    done
fi

if [ ${only_print} = true ]; then
  echo "!PRINT ONLY!"
else
  echo "!PRINT AND RUN!"
fi
echo run ${isim} simulations