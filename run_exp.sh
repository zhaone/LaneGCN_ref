#!/bin/bash
set -e

function main() {
  # export CUDA_VISIBLE_DEVICES="2"
  export PYTHONPATH=/home/lgcn/argoverse-api:$PYTHONPATH
  echo "Using PYTHONPATH:$PYTHONPATH"
  if [[ $# -eq 0 ]]; then
    echo "please give model_name and exp_name"
    echo "==================="
    echo "Available model set:"
    echo commands/*
    exit 1
  fi

  local model=$1
  shift

  local exp=$1

  if [[ -f "commands/${model}.sh" ]] ; then
    source commands/${model}.sh
    if [ -n "$(LC_ALL=C type -t ${exp})" ] && [ "$(LC_ALL=C type -t ${exp})" = function ]; then
      eval ${exp}
    else
      echo "exp name ${exp} is NOT set";
      echo "=================="
      echo "Available experiments:"
      typeset -f | awk '/ \(\) $/ && !/^main / {print $1}'
    fi
  else
    echo "No scripts for ${model}"
    echo "==================="
    echo "Available model set:"
    echo commands/*
    exit 1
  fi

}

main $@
