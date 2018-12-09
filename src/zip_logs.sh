#!/usr/bin/env bash

mkdir ../logs

cp -r ../tf_logs/ ../logs
cp -r ../train_logs/ ../logs
cp -r ../test_logs/ ../logs
cp -r ../checkpoints/ ../logs

zip -r ../logs.zip ../logs

rm -rf ../logs