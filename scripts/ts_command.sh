command="${1}"

GPU_AMOUNT=4
for ((i = 0; i < $GPU_AMOUNT; i += 1)); do
  if [[ "$command" == *"-C"* ]]; then
    TS_SOCKET=/tmp/socket-cuda${i} tsp | grep finish >> runs/tsp_run_log.txt
  fi
  TS_SOCKET=/tmp/socket-cuda${i} tsp ${command}
done
