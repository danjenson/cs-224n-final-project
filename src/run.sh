#!/bin/bash
NAME=$1
./model.py dataset
./model.py train -c $NAME.yaml
./model.py predict -c $NAME.yaml
./model.py score -c $NAME.yaml
