#!/bin/bash

BERT_BASE_DIR=/Users/saybot/Projects/github/bert/chinese_L-12_H-768_A-12

python run_classifier.py \
  --task_name=Category \
  --do_train=True \
  --do_eval=True \
  --do_export=False \
  --data_dir=Category \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=13.0 \
  --output_dir=category_output/


python run_classifier.py \
  --task_name=Category \
  --do_train=False \
  --do_eval=False \
  --do_export=True \
  --data_dir=Category \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=category_output/ \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=13.0 \
  --output_dir=category_output/
