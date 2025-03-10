#!/bin/bash

OUTPUT_DIR=${1:-sample-configurations}

GIT_DIFF=$(git diff HEAD -- $OUTPUT_DIR)
echo "git diff of configurations with HEAD:"
echo "$GIT_DIFF"

function echoWarning() {
  LIGHT_YELLOW='\033[1;33m'
  NC='\033[0m' # No Color
  echo -e "${LIGHT_YELLOW}${1}${NC}"
}

if [ ! -z "$GIT_DIFF" ]; then
    echoWarning "At least one of the configs in the plugins should have changed."
    echoWarning "Please run 'tox -e gen-configs' to ensure that the sample-configurations are correctly generated!"
    echoWarning "After that commit the generated sample-configurations to remove this error."
    exit 1
fi

echo "sample configurations up to date with configs in plugin directories"
