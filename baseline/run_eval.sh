#!/bin/sh

cd evaluation
python get_log.py
python squadv2_eval.py
cd ..
