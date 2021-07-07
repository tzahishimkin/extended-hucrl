command="${1}"

GPU_AMOUNT=4
for ((i = 0; i < $GPU_AMOUNT; i += 1)); do
  TS_SOCKET=/tmp/socket-cuda${i} tsp ${command}
done
