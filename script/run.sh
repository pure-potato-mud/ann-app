SCRIPT_FILE=${BASH_SOURCE[0]}
SOURCE_FILE=$0
PROJ_DIR=$(cd -- "$( dirname -- "$SCRIPT_FILE" )/.." &> /dev/null && pwd)
PROJ_NAME=$(basename "$PROJ_DIR")
EXEC_FILE_NAME=${PROJ_DIR}/build/${PROJ_NAME}


echo -e "running executable...\n"

timer_start=$(date +%s%N)

$EXEC_FILE_NAME
run_status=$?

timer_stop=$(date +%s%N)
during_time=$(($timer_stop - $timer_start))
display_during=$(awk -v during_time="$during_time" 'BEGIN { printf "%.6f\n", during_time / 1000000000 }')

echo -e "\nprogram exit with $run_status during $display_during second(s)."