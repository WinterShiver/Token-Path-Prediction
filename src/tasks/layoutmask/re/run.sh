CUDA_VISIBLE_DEVICES=0 python ./src/tasks/layoutmask/re/model_training_for_ner_tpp_group.py \
  --do_train true \
  --grouping true \
  --data_dir /path/to/FUNSD_group \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.val.txt \
  --test_dataset_name data.test.txt \
  --label_file_name labels.txt \
  --pretrained_model_path /path/to/layoutmask-mpm-english-base \
  --save_model_dir /path/to/savedir \
  --max_length 512 \
  --batch_size 16 \
  --max_epochs 500 \
  --learning_rate 8e-5 \
  --patience 500 \
  --log_every_n_steps 1 \
  --shuffle true \
  --enable_aug false \
  --use_multi_dropout false \
  --seed 2023 \
  --gpus 1 \
  --eval_set 0

CUDA_VISIBLE_DEVICES=2 python ./src/tasks/layoutmask/re/model_training_for_ner_tpp_group.py \
  --do_train true \
  --grouping true \
  --data_dir /path/to/FUNSD_group \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.val.txt \
  --test_dataset_name data.test.txt \
  --label_file_name labels.txt \
  --pretrained_model_path /path/to/layoutmask-mpm-english-base \
  --save_model_dir /path/to/savedir \
  --max_length 512 \
  --batch_size 16 \
  --max_epochs 500 \
  --learning_rate 8e-5 \
  --patience 500 \
  --log_every_n_steps 1 \
  --shuffle true \
  --enable_aug false \
  --use_multi_dropout true \
  --seed 2023 \
  --gpus 1 \
  --eval_set 0


CUDA_VISIBLE_DEVICES=3 python ./src/tasks/layoutmask/re/model_training_for_ner_tpp_group.py \
  --do_train true \
  --grouping true \
  --data_dir /path/to/FUNSD_group \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.val.txt \
  --test_dataset_name data.test.txt \
  --label_file_name labels.txt \
  --pretrained_model_path /path/to/layoutmask-mpm-english-base \
  --save_model_dir /path/to/savedir \
  --max_length 512 \
  --batch_size 16 \
  --max_epochs 500 \
  --learning_rate 1e-4 \
  --patience 500 \
  --log_every_n_steps 2 \
  --shuffle true \
  --enable_aug false \
  --use_multi_dropout false \
  --seed 2023 \
  --gpus 1 \
  --eval_set 0

CUDA_VISIBLE_DEVICES=0 python ./src/tasks/layoutmask/re/model_training_for_ner_tpp_group.py \
  --do_train true \
  --grouping true \
  --data_dir /path/to/FUNSD_group \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.val.txt \
  --test_dataset_name data.test.txt \
  --label_file_name labels.txt \
  --pretrained_model_path /path/to/layoutmask-mpm-english-base \
  --save_model_dir /path/to/savedir \
  --max_length 512 \
  --batch_size 16 \
  --max_epochs 500 \
  --learning_rate 5e-5 \
  --patience 500 \
  --log_every_n_steps 1 \
  --shuffle true \
  --enable_aug false \
  --use_multi_dropout true \
  --seed 2023 \
  --gpus 1 \
  --eval_set 0

# eval_set  = 0 label未知
# eval_set  = 1 label已知，同类实体不会link


CUDA_VISIBLE_DEVICES=0 python ./src/tasks/layoutmask/re/model_training_for_ner_tpp_group.py \
  --do_train true \
  --grouping true \
  --data_dir /path/to/FUNSD_group \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.val.txt \
  --test_dataset_name data.test.txt \
  --label_file_name labels.txt \
  --pretrained_model_path /path/to/layoutmask-mpm-english-base \
  --save_model_dir /path/to/savedir \
  --max_length 512 \
  --batch_size 16 \
  --max_epochs 500 \
  --learning_rate 8e-5 \
  --patience 500 \
  --log_every_n_steps 1 \
  --shuffle true \
  --enable_aug false \
  --use_multi_dropout false \
  --seed 2023 \
  --gpus 1 \
  --eval_set 1

CUDA_VISIBLE_DEVICES=1 python ./src/tasks/layoutmask/re/model_training_for_ner_tpp_group.py \
  --do_train true \
  --grouping true \
  --data_dir /path/to/FUNSD_group \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.val.txt \
  --test_dataset_name data.test.txt \
  --label_file_name labels.txt \
  --pretrained_model_path /path/to/layoutmask-mpm-english-base \
  --save_model_dir /path/to/savedir \
  --max_length 512 \
  --batch_size 16 \
  --max_epochs 500 \
  --learning_rate 8e-5 \
  --patience 500 \
  --log_every_n_steps 1 \
  --shuffle true \
  --enable_aug false \
  --use_multi_dropout true \
  --seed 2023 \
  --gpus 1 \
  --eval_set 1

CUDA_VISIBLE_DEVICES=0 python ./src/tasks/layoutmask/re/model_training_for_ner_tpp_group.py \
  --do_train true \
  --grouping true \
  --data_dir /path/to/FUNSD_group \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.val.txt \
  --test_dataset_name data.test.txt \
  --label_file_name labels.txt \
  --pretrained_model_path /path/to/layoutmask-mpm-english-base \
  --save_model_dir /path/to/savedir \
  --max_length 512 \
  --batch_size 16 \
  --max_epochs 500 \
  --learning_rate 5e-5 \
  --patience 500 \
  --log_every_n_steps 1 \
  --shuffle true \
  --enable_aug false \
  --use_multi_dropout true \
  --seed 2023 \
  --gpus 1 \
  --eval_set 1

## Test
# CUDA_VISIBLE_DEVICES=2 python -m src.tasks.layoutmask.re.model_training_for_ner_tpp_group \
#   --do_train false \
#   --do_test true \
#   --grouping true \
#   --data_dir /path/to/FUNSD_group \
#   --train_dataset_name data.train.txt \
#   --valid_dataset_name data.val.txt \
#   --test_dataset_name data.val.txt \
#   --label_file_name labels.txt \
#   --pretrained_model_path /path/to/layoutmask-mpm-english-base \
#   --ckpt_path /path/to/ckpt \
#   --save_model_dir /path/to/savedir \
#   --max_length 512 \
#   --batch_size 16 \
#   --max_epochs 500 \
#   --learning_rate 8e-5 \
#   --patience 500 \
#   --log_every_n_steps 1 \
#   --shuffle true \
#   --enable_aug false \
#   --use_multi_dropout false \
#   --seed 2023 \
#   --gpus 1