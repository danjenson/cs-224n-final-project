#!/bin/bash
./model.py train -c "$1.yaml" \
  && ./model.py predict -c "$1.yaml" \
  && ./model.py score -c "$1.yaml"
