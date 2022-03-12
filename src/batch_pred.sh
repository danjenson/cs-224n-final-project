#!/bin/bash
MODEL=bart
TASK=predict
for i in {1..10}
do
sed -i "s/^output_path:.*$/output_path: \".\/results\/$MODEL\/epoch$i\"/" $MODEL.yaml
./simple_run.py $TASK -c $MODEL.yaml
done