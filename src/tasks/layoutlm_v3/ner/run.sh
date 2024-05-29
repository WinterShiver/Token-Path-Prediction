export PYTHONPATH=/opt/conda/lib/python3.8/site-packages/aistudio_notebook/public:/opt/conda/lib/python3.8/site-packages/aistudio_notebook/public:/opt/conda/lib/python3.8/site-packages/aistudio_notebook/public:/opt/conda/lib/python3.8/site-packages/aistudio_notebook/public:/opt/conda/lib/python3.8/site-packages/aistudio_notebook/public

## BIO

### Train

# CUDA_VISIBLE_DEVICES=3 python ./src/tasks/layoutlm_v3/ner/model_training_for_ner_bio.py \
#   --do_train true \
#   --data_dir /path/to/FUNSD/ \
#   --train_dataset_name data.train.txt \
#   --valid_dataset_name data.val.txt \
#   --test_dataset_name data.test.txt \
#   --label_file_name labels_bio.txt \
#   --pretrained_model_path /path/to/layoutlmv3-base \
#   --save_model_dir /path/to/savedir \
#   --enable_position_1d true \
#   --box_level segment \
#   --max_length 512 \
#   --batch_size 16 \
#   --max_epochs 500 \
#   --learning_rate 5e-5 \
#   --log_every_n_steps 1 \
#   --keep_checkpoint_max 2 \
#   --patience 500 \
#   --shuffle true \
#   --seed 2023 \
#   --gpus 1

# CUDA_VISIBLE_DEVICES=3 python ./src/tasks/layoutlm_v3/ner/model_training_for_ner_bio.py \
#   --do_train true \
#   --data_dir /path/to/CORD/ \
#   --train_dataset_name data.train.txt \
#   --valid_dataset_name data.valid.txt \
#   --test_dataset_name data.test.txt \
#   --label_file_name labels_bio.txt \
#   --pretrained_model_path /path/to/layoutlmv3-base \
#   --save_model_dir /path/to/savedir \
#   --enable_position_1d true \
#   --box_level segment \
#   --max_length 512 \
#   --batch_size 16 \
#   --max_epochs 500 \
#   --learning_rate 5e-5 \
#   --log_every_n_steps 1 \
#   --keep_checkpoint_max 2 \
#   --patience 500 \
#   --shuffle true \
#   --seed 2023 \
#   --gpus 1

### Test

# CUDA_VISIBLE_DEVICES=7 python ./src/tasks/layoutlm_v3/ner/model_training_for_ner_bio.py \
#   --do_train false \
#   --do_test true \
#   --do_predict false \
#   --data_dir /path/to/FUNSD/ \
#   --train_dataset_name data.train.txt \
#   --valid_dataset_name data.val.txt \
#   --test_dataset_name data.test.txt \
#   --label_file_name labels_bio.txt \
#   --pretrained_model_path /path/to/layoutlmv3-base \
#   --save_model_dir /path/to/savedir \
#   --ckpt /path/to/ckpt \
#   --predict_result_file /path/to/predict_result_file/res.txt \
#   --enable_position_1d false \
#   --box_level word \
#   --max_length 512 \
#   --batch_size 16 \
#   --log_every_n_steps 1 \
#   --shuffle false \
#   --seed 2023 \
#   --gpus 1

# CUDA_VISIBLE_DEVICES=1 python ./src/tasks/layoutlm_v3/ner/model_training_for_ner_bio.py \
#   --do_train false \
#   --do_test true \
#   --do_predict false \
#   --data_dir /path/to/CORD/ \
#   --train_dataset_name data.train.txt \
#   --valid_dataset_name data.valid.txt \
#   --test_dataset_name data.test.txt \
#   --label_file_name labels_bio.txt \
#   --pretrained_model_path /path/to/layoutlmv3-base \
#   --save_model_dir /path/to/savedir \
#   --ckpt /path/to/ckpt \
#   --enable_position_1d true \
#   --box_level segment \
#   --max_length 512 \
#   --batch_size 16 \
#   --log_every_n_steps 1 \
#   --shuffle false \
#   --seed 2023 \
#   --gpus 1

## TPP

### Train

CUDA_VISIBLE_DEVICES=0 python ./src/tasks/layoutlm_v3/ner/model_training_for_ner_tpp.py \
  --do_train true \
  --data_dir /path/to/FUNSD-r \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.val.txt \
  --test_dataset_name data.test.txt \
  --label_file_name labels.txt \
  --pretrained_model_path /path/to/layoutlmv3-base \
  --save_model_dir /path/to/savedir \
  --enable_position_1d true \
  --use_last_layer_position_embedding_residual false \
  --use_multi_dropout false \
  --box_level segment \
  --max_entities 100 \
  --max_length 512 \
  --batch_size 16 \
  --max_epochs 500 \
  --learning_rate 5e-5 \
  --log_every_n_steps 1 \
  --keep_checkpoint_max 2 \
  --patience 500 \
  --shuffle true \
  --seed 0 \
  --gpus 1

# CUDA_VISIBLE_DEVICES=1 python ./src/tasks/layoutlm_v3/ner/model_training_for_ner_tpp.py \
#   --do_train true \
#   --data_dir /path/to/CORD-r \
#   --train_dataset_name data.train.txt \
#   --valid_dataset_name data.valid.txt \
#   --test_dataset_name data.test.txt \
#   --label_file_name labels.txt \
#   --pretrained_model_path /path/to/layoutlmv3-base \
#   --save_model_dir /path/to/savedir \
#   --enable_position_1d true \
#   --use_last_layer_position_embedding_residual true \
#   --use_multi_dropout true \
#   --box_level segment \
#   --max_entities 100 \
#   --max_length 512 \
#   --batch_size 16 \
#   --max_epochs 200 \
#   --learning_rate 5e-5 \
#   --log_every_n_steps 1 \
#   --keep_checkpoint_max 2 \
#   --patience 200 \
#   --shuffle true \
#   --seed 2023 \
#   --gpus 1

### Test

# CUDA_VISIBLE_DEVICES=0 python ./src/tasks/layoutlm_v3/ner/model_training_for_ner_tpp.py \
#   --do_train false \
#   --do_test true \
#   --do_predict false \
#   --data_dir /path/to/FUNSD/ \
#   --train_dataset_name data.train.txt \
#   --valid_dataset_name data.val.txt \
#   --test_dataset_name data.test.txt \
#   --label_file_name labels.txt \
#   --pretrained_model_path /path/to/layoutlmv3-base \
#   --ckpt /path/to/ckpt \
#   --save_model_dir /path/to/savedir \
#   --predict_result_file /path/to/predict_result_file/res.txt \
#   --enable_position_1d true \
#   --use_last_layer_position_embedding_residual true \
#   --use_multi_dropout false \
#   --box_level segment \
#   --max_entities 100 \
#   --max_length 512 \
#   --batch_size 16 \
#   --log_every_n_steps 1 \
#   --shuffle false \
#   --seed 2023 \
#   --gpus 1

# CUDA_VISIBLE_DEVICES=0 python ./src/tasks/layoutlm_v3/ner/model_training_for_ner_tpp.py \
#   --do_train false \
#   --do_test true \
#   --data_dir /path/to/CORD-r \
#   --train_dataset_name data.train.txt \
#   --valid_dataset_name data.valid.txt \
#   --test_dataset_name data.test.txt \
#   --label_file_name labels.txt \
#   --pretrained_model_path /path/to/layoutlmv3-base \
#   --ckpt /path/to/ckpt \
#   --save_model_dir /path/to/savedir \
#   --enable_position_1d true \
#   --use_last_layer_position_embedding_residual false \
#   --use_multi_dropout false \
#   --box_level segment \
#   --max_entities 100 \
#   --max_length 512 \
#   --batch_size 16 \
#   --log_every_n_steps 1 \
#   --shuffle false \
#   --seed 2023 \
#   --gpus 1