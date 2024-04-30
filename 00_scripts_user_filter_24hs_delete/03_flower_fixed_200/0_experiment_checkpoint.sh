#!/bin/bash

echo "Starting 1"

#python flower_experiment_script_complete_unbalanced_serial_check_point.py checkpoint_iteration=100 iterations=200 cycle=0 > log_learning_checkpoint_100_C0_01.log #
#sleep 10 
#python flower_experiment_script_complete_unbalanced_serial_check_point.py checkpoint_iteration=200 iterations=200 cycle=0 > log_learning_checkpoint_200_C0_02.log #

sleep 3  # Sleep for 3s to give to the GC

echo "human cycle"

python flower_human_inference_experiment_unbalanced.py iterations=200 cycle=0 > log_human_checkpoint_200_01.log #

sleep 3  # Sleep for 3s to give to the GC

echo "Starting 2"

python flower_experiment_script_complete_unbalanced_serial_check_point.py checkpoint_iteration=100 iterations=200 cycle=1 > log_learning_checkpoint_100_C1_01.log #
sleep 10 
python flower_experiment_script_complete_unbalanced_serial_check_point.py checkpoint_iteration=200 iterations=200 cycle=1 > log_learning_checkpoint_200_C1_02.log #

sleep 3  # Sleep for 3s to give to the GC

echo "human cycle"

python flower_human_inference_experiment_unbalanced.py iterations=200 cycle=1 > log_human_checkpoint_200_02.log #


#python flower_experiment_script_complete_unbalanced_serial_check_point.py iterations=200 cycle=1 > log_checkpoint_200_C1.log  #

#sleep 3  # Sleep for 3s to give to the GC

#echo "human cycle"

#python flower_human_inference_experiment_unbalanced.py iterations=200 cycle=0 > log_human_checkpoint_200_01.log #

