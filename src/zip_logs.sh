#!/usr/bin/env bash

mkdir ../logs

cp ../tf_logs/ ../logs
cp ../train_logs/ ../logs
cp ../test_logs/ ../logs
cp ../checkpoints/ ../logs

zip ../logs.zip ../logs

rm -rf ../logs