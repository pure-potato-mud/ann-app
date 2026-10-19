SCRIPT_FILE=${BASH_SOURCE[0]}
SOURCE_FILE=$0
PROJ_DIR=$(cd -- "$( dirname -- "$SCRIPT_FILE" )/.." &> /dev/null && pwd)
PROJ_NAME=$(basename "$PROJ_DIR")
EXEC_FILE_NAME=${PROJ_DIR}/build/${PROJ_NAME}.out

# build
echo "compiling..."
gcc --std=c99 ${PROJ_DIR}/src/*.c -I ${PROJ_DIR}/src/ -lm -o $EXEC_FILE_NAME
compile_status=$?
if [ $compile_status -eq 0 ]; then
  echo "compile completed."
else
  echo "compile failed."
  if [ $SCRIPT_FILE = $SOURCE_FILE ]; then
    exit $compile_status
  else
    return $compile_status
  fi
fi


# run(optional)
if [ "$1" = "--run" ]; then
  source $PROJ_DIR/script/run.sh
fi
