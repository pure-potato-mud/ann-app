PROJ_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd)
PROJ_NAME=$(basename "$PROJ_DIR")
EXEC_FILE_NAME=${PROJ_DIR}/build/${PROJ_NAME}.out

# build
echo "compiling..."
gcc --std=c99 ${PROJ_DIR}/src/main.c -o $EXEC_FILE_NAME
echo "compile completed."

# run(optional)
if [ "$1" = "--run" ]; then
  echo -e "running executable...\n"

  timer_start=$(date +%s%N)

  $EXEC_FILE_NAME
  status=$?

  timer_stop=$(date +%s%N)
  during_time=$(($timer_stop - $timer_start))
  display_during=$(awk -v during_time="$during_time" 'BEGIN { printf "%.6f\n", during_time / 1000000000 }')

  echo -e "\nprogram exit with $status during $display_during second(s)."
fi
