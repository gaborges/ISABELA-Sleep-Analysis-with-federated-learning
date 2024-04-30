#!/bin/bash

echo "Starting 1"
python 01_LSTM_BI_unbalanced_script_4_4_time_steps.py > log_LSTM_BI_4_4.log #

echo "Starting 2"
python 01_LSTM_BI_unbalanced_script_30_10_time_steps.py > log_LSTM_BI_30_10.log #

echo "Starting 3"
python 01_LSTM_BI_unbalanced_script_60_15_time_step.py  > log_LSTM_BI_60_15.log #

echo "Starting 4"
python 01_LSTM_BI_unbalanced_script_60_30_time_steps.py > log_LSTM_BI_60_30.log #

echo "Starting 5"
python 01_LSTM_BI_unbalanced_script_120_15_time_step.py > log_LSTM_BI_120_15.log #
