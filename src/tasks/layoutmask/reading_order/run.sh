CUDA_VISIBLE_DEVICES=0 python ./src/tasks/layoutmask/reading_order/model_training_for_ro.py \
   --do_train true \
   --data_dir ./data/reading_bank \
   --train_dataset_name data.train.jsonl \
   --valid_dataset_name data.valid.jsonl \
   --test_dataset_name data.test.jsonl \
   --pretrained_model_path /path/to/layoutmask-mpm-english-base \
   --save_model_dir /path/to/savedir \
   --rate_shuffle_segments 0 \
   --use_last_layer_position_embedding_residual true \
   --use_multi_dropout true \
   --shuffle_when_evaluation False \
   --max_length 512 \
   --batch_size 16 \
   --max_epochs 100 \
   --learning_rate 5e-5 \
   --patience 100 \
   --log_every_n_steps 10 \
   --shuffle true \
   --seed 2023 \
   --gpus 1