SCRIPT_FILE=${BASH_SOURCE[0]}
PROJ_DIR=$(cd -- "$( dirname -- "$SCRIPT_FILE" )/.." &> /dev/null && pwd)
BUILD_DIR=${PROJ_DIR}/build

ctest --test-dir $BUILD_DIR --output-on-failure