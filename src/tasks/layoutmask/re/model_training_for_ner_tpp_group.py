import json
import math
import os
import torch
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from distutils.util import strtobool
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from src.dataset.augment_processor.aug_for_perspective_trans import AugForPerspectiveTrans

from src.dataset.code_doc_data_module import CodeDocDataModule  # OK, same
from src.dataset.code_doc_dataset import CodeDocLocalDataset  # OK, same
from datasets import ClassLabel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from pytorch_lightning.utilities.argparse import add_argparse_args
from transformers import get_linear_schedule_with_warmup

from src.dataset.feature_processor.layoutmask.ner_tpp_group_processor import CodeDocForTPPGP2NerProcessorV2
from src.model.layoutmask.configuration_layoutmask import LayoutMaskConfig
from src.model.layoutmask.modeling_layoutmask import LayoutMaskModelForTPPGrouping
from src.metric.tpp_gb_entity_metric import TPPGB2NERMetric
from src.utils.log_utils import create_logger_v2


class CodeDocNERGBModelModule(pl.LightningModule):
    def __init__(self,
                 ner_labels: ClassLabel,
                 num_samples: int,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-6,
                 warmup_ratio: float = 0.05,
                 **kargs):
        '''
        task_type: NER、NER_CLS、CLS、RE
        '''
        super().__init__()
        self.save_hyperparameters(ignore=['collate_fn', 'tokenizer'])

        # 加载自定义模型类
        self.config = LayoutMaskConfig.from_pretrained(self.hparams.pretrained_model_path, output_hidden_states=True)

        ##  使用num_hidden_layers层Encoder
        if self.hparams.num_hidden_layers is not None and self.hparams.num_hidden_layers > 0:
            self.config.num_hidden_layers = self.hparams.num_hidden_layers

        self.ner_labels = ner_labels

        self.config.num_labels = ner_labels.num_classes
        self.config.classifier_dropout = self.hparams.dropout
        self.config.grouping = self.hparams.grouping
        self.config.use_multi_dropout = self.hparams.use_multi_dropout

        self.config.use_last_layer_position_embedding_residual = True
        self.model = LayoutMaskModelForTPPGrouping.from_pretrained(config=self.config,
                                                                   pretrained_model_name_or_path=self.hparams.pretrained_model_path,
                                                                   ignore_mismatched_sizes=True)

        # 设置metric
        # self.valid_metric = TPPGB2NERMetric()
        if self.config.grouping:
            self.grouping_valid_metric = TPPGB2NERMetric()
            # self.implicit_grouping_valid_metric = TPPGB2NERMetric()
        if self.global_rank == 0:
            self.local_logger = create_logger_v2(log_dir=self.hparams.save_model_dir)
            self.local_logger.info(self.hparams)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):

        outputs = self(**batch)
        loss = outputs.loss

        steps = self.global_step

        # 这里在训练时打日志，由 log_every_n_steps 控制频率
        if self.global_rank == 0 and self.local_rank == 0 and (steps + 1) % self.trainer.log_every_n_steps == 0:
            lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
            self.local_logger.info(
                f"Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"Steps: {steps}, "
                f"Learning Rate {lr_scheduler.get_last_lr()[-1]:.7f}, "
                f"Train Loss: {loss:.5f},"
                # f"Label Loss: {outputs.label_loss:.5f},"
                f"Group Loss: {outputs.group_loss:.5f}"
            )

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs.loss, outputs.logits

        entities_scores = logits.detach().cpu().numpy()

        if self.config.grouping:
            all_true_grouping_labels = [v.split(' ') if v != '' else [] for v in batch['group_label_texts']]
            all_entities = [json.loads(v) for v in batch['entities']]
            entity_type_maps = [json.loads(v) for v in batch['entity_type_map']]

            pred_grouping_labels = [[] for _ in range(len(all_true_grouping_labels))]
            true_grouping_labels = [[] for _ in range(len(all_true_grouping_labels))]

            b_idx = 0

            for grouping_label, grouping_scores, all_entity, entity_type_map in zip(all_true_grouping_labels,
                                                                                    entities_scores,
                                                                                    all_entities,
                                                                                    entity_type_maps):
                grouping_scores = grouping_scores[0]

                entities = [tuple(v) for v in all_entity]

                for qas in grouping_label:
                    q, a = qas.split(':')

                    true_grouping_labels[b_idx].append('{}:{}'.format(q, a))
                    true_grouping_labels[b_idx].append('{}:{}'.format(a, q))

                qa_threshold = 0.0

                for entity_0 in entities:
                    for entity_1 in entities:
                        if entity_0 == entity_1:
                            continue
                        entity_0 = list(entity_0)
                        entity_1 = list(entity_1)

                        entity_0_k = '-'.join([str(v) for v in entity_0])
                        entity_1_k = '-'.join([str(v) for v in entity_1])

                        if self.hparams.eval_set == 1:
                            if entity_type_map[entity_0_k] == entity_type_map[entity_1_k]:
                                continue

                        qa_scores = []
                        for q_token_id in entity_0:
                            for a_token_id in entity_1:
                                score = (grouping_scores[q_token_id][a_token_id] + grouping_scores[a_token_id][
                                    q_token_id]) / 2.0
                                qa_scores.append(score)

                        qa_score = np.mean(qa_scores)

                        if qa_score > qa_threshold:
                            pred_grouping_labels[b_idx].append(
                                entity_0_k + ':' + entity_1_k
                            )
                b_idx += 1

            self.grouping_valid_metric(pred_grouping_labels, true_grouping_labels)

        return val_loss

    def validation_epoch_end(self, validation_step_outputs):

        if self.config.grouping:
            implicit_results = self.grouping_valid_metric.compute()

            val_implicit_grouping_f1 = implicit_results['f1']
            val_implicit_grouping_precision = implicit_results['precision']
            val_implicit_grouping_recall = implicit_results['recall']

            self.log("val_grouping_f1", val_implicit_grouping_f1, prog_bar=True, on_epoch=True)
            self.log("val_grouping_precision", val_implicit_grouping_precision, prog_bar=True, on_epoch=True)
            self.log("val_grouping_recall", val_implicit_grouping_recall, prog_bar=True, on_epoch=True)

        if self.global_rank == 0 and self.local_rank == 0:

            if self.config.grouping:
                self.local_logger.info(
                    f"val_grouping_f1: {val_implicit_grouping_f1:.5f}, "
                    f"val_grouping_precision: {val_implicit_grouping_precision:.5f}, "
                    f"val_grouping_recall: {val_implicit_grouping_recall:.5f}, "
                )
        # *** 这个一定需要，不然会重复累积 *** #
        # self.valid_metric.reset()
        if self.config.grouping:
            self.grouping_valid_metric.reset()
            # self.implicit_grouping_valid_metric.reset()

    def test_step(self, batch, batch_idx):

        outputs = self(**batch)
        val_loss, logits = outputs.loss, outputs.logits

        entities_scores = logits.detach().cpu().numpy()

        if self.config.grouping:
            all_true_grouping_labels = [v.split(' ') if v != '' else [] for v in batch['group_label_texts']]
            all_entities = [json.loads(v) for v in batch['entities']]
            entity_type_maps = [json.loads(v) for v in batch['entity_type_map']]

            pred_grouping_labels = [[] for _ in range(len(all_true_grouping_labels))]
            true_grouping_labels = [[] for _ in range(len(all_true_grouping_labels))]

            b_idx = 0

            for grouping_label, grouping_scores, all_entity, entity_type_map in zip(all_true_grouping_labels,
                                                                                    entities_scores,
                                                                                    all_entities,
                                                                                    entity_type_maps):
                grouping_scores = grouping_scores[0]

                entities = [tuple(v) for v in all_entity]

                for qas in grouping_label:
                    q, a = qas.split(':')

                    true_grouping_labels[b_idx].append('{}:{}'.format(q, a))
                    true_grouping_labels[b_idx].append('{}:{}'.format(a, q))

                qa_threshold = 0.0

                for entity_0 in entities:
                    for entity_1 in entities:
                        if entity_0 == entity_1:
                            continue
                        entity_0 = list(entity_0)
                        entity_1 = list(entity_1)

                        entity_0_k = '-'.join([str(v) for v in entity_0])
                        entity_1_k = '-'.join([str(v) for v in entity_1])

                        if self.hparams.eval_set == 1:
                            if entity_type_map[entity_0_k] == entity_type_map[entity_1_k]:
                                continue

                        qa_scores = []
                        for q_token_id in entity_0:
                            for a_token_id in entity_1:
                                score = (grouping_scores[q_token_id][a_token_id] + grouping_scores[a_token_id][
                                    q_token_id]) / 2.0
                                qa_scores.append(score)

                        qa_score = np.mean(qa_scores)

                        if qa_score > qa_threshold:
                            pred_grouping_labels[b_idx].append(
                                entity_0_k + ':' + entity_1_k
                            )
                b_idx += 1

            self.grouping_valid_metric(pred_grouping_labels, true_grouping_labels)

        return val_loss

    def test_epoch_end(self, outputs):

        if self.config.grouping:
            implicit_results = self.grouping_valid_metric.compute()

            val_implicit_grouping_f1 = implicit_results['f1']
            val_implicit_grouping_precision = implicit_results['precision']
            val_implicit_grouping_recall = implicit_results['recall']

            self.log("val_grouping_f1", val_implicit_grouping_f1, prog_bar=True, on_epoch=True)
            self.log("val_grouping_precision", val_implicit_grouping_precision, prog_bar=True, on_epoch=True)
            self.log("val_grouping_recall", val_implicit_grouping_recall, prog_bar=True, on_epoch=True)

        if self.global_rank == 0 and self.local_rank == 0:

            if self.config.grouping:
                self.local_logger.info(
                    f"val_grouping_f1: {val_implicit_grouping_f1:.5f}, "
                    f"val_grouping_precision: {val_implicit_grouping_precision:.5f}, "
                    f"val_grouping_recall: {val_implicit_grouping_recall:.5f}, "
                )
        # *** 这个一定需要，不然会重复累积 *** #
        # self.valid_metric.reset()
        if self.config.grouping:
            self.grouping_valid_metric.reset()
            # self.implicit_grouping_valid_metric.reset()

    # 用于模型测试
    def predict_step(self, batch, batch_idx):

        return None

    def configure_callbacks(self):
        call_backs = []

        call_backs.append(LearningRateMonitor(logging_interval='step'))
        call_backs.append(EarlyStopping(monitor="val_grouping_f1", mode="max", patience=self.hparams.patience))
        call_backs.append(
            ModelCheckpoint(monitor="val_grouping_f1",
                            mode="max",
                            every_n_epochs=self.hparams.every_n_epochs,
                            filename='codedoc-{epoch}-{step}-{val_grouping_f1:.5f}'
                            )
        )

        if self.hparams.to_onnx:
            from callbacks.to_onnx import ToOnnxCallback

            onnx_input_names = ['input_ids', 'attention_mask', 'position_1d', 'position_2d']
            onnx_output_names = ['logits']
            onnx_dynamic_axes = ['0:batch_size,1:max_length',
                                 '0:batch_size,1:max_length',
                                 '0:batch_size,1:max_length',
                                 '0:batch_size,1:max_length',
                                 '0:batch_size']

            call_backs.append(ToOnnxCallback(onnx_input_names=onnx_input_names,
                                             onnx_output_names=onnx_output_names,
                                             onnx_dynamic_axes=onnx_dynamic_axes,
                                             to_onnx_expected_decimal=self.hparams.to_onnx_expected_decimal,
                                             fp16_expected_decimal=self.hparams.fp16_expected_decimal))
        return call_backs

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.hparams.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        # ]

        param_optimizer = list(model.named_parameters())
        # pretrain model param
        param_pre = [(n, p) for n, p in param_optimizer if 'global_pointer' not in n]
        # downstream model param
        param_downstream = [(n, p) for n, p in param_optimizer if 'global_pointer' in n]
        # no_decay = ['bias', 'LayerNorm', 'layer_norm']
        optimizer_grouped_parameters = [
            # pretrain model param
            {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.learning_rate
             },
            {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.hparams.learning_rate
             },
            # downstream model
            {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.learning_rate * 10
             },
            {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.hparams.learning_rate * 10
             }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                      eps=self.hparams.adam_epsilon)
        num_warmup_steps = int(self.total_steps * self.hparams.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def setup(self, stage=None):
        if stage != "fit":
            return
        num_samples = self.hparams.num_samples
        batch_size = self.hparams.batch_size

        steps = math.ceil(num_samples / batch_size / max(1, self.trainer.devices))
        # Calculate total steps
        ab_steps = int(steps / self.trainer.accumulate_grad_batches)
        # self.total_steps = int(records_count // ab_size * self.trainer.max_epochs)
        self.total_steps = int(ab_steps * self.trainer.max_epochs)
        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(f"- num_samples is: {num_samples}")
            self.local_logger.info(f"- max_epochs is: {self.trainer.max_epochs}")
            self.local_logger.info(f"- total_steps is: {self.total_steps}")
            self.local_logger.info(f"- batch size (1 gpu) is: {batch_size}")
            self.local_logger.info(f"- devices(gpu) num is: {max(1, self.trainer.devices)}")
            self.local_logger.info(f"- accumulate_grad_batches is: {self.trainer.accumulate_grad_batches}")

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        return add_argparse_args(cls, parent_parser, **kwargs)


class LightningRunner:
    def __init__(self, args):
        self.args = args

    def run(self):
        # mp.set_start_method('spawn')
        # 设置随机种子
        pl.seed_everything(self.args.seed)

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.args.pretrained_model_path)

        ner_labels = ClassLabel(names_file=os.path.join(self.args.data_dir, self.args.label_file_name))
        '''
        这里不同的任务使用不同的Processor
        '''
        augment_processors = []
        collate_fn_train = None
        if self.args.enable_aug:
            # augment_processors.append(AugForBillEntityReplace(prob=0.2))
            # augment_processors.append(AugForSegmentSplit(prob=0.05))
            # augment_processors.append(AugForRndMove(prob=0.5))
            augment_processors.append(AugForPerspectiveTrans(prob=0.1))

        data_processor_train = CodeDocForTPPGP2NerProcessorV2(tokenizer=tokenizer,
                                                              ner_labels=ner_labels,
                                                              bbox_level=self.args.bbox_level,
                                                              max_text_length=self.args.max_length,
                                                              norm_bbox_height=self.args.norm_bbox_height,
                                                              norm_bbox_width=self.args.norm_bbox_width,
                                                              augment_processors=augment_processors,
                                                              )

        data_processor = CodeDocForTPPGP2NerProcessorV2(tokenizer=tokenizer,
                                                        ner_labels=ner_labels,
                                                        bbox_level=self.args.bbox_level,
                                                        max_text_length=self.args.max_length,
                                                        norm_bbox_height=self.args.norm_bbox_height,
                                                        norm_bbox_width=self.args.norm_bbox_width,
                                                        )

        # 定义数据
        train_dataset, valid_dataset, test_dataset = None, None, None
        num_samples = 1
        if self.args.train_dataset_name and self.args.do_train:
            train_dataset = CodeDocLocalDataset(data_dir=self.args.data_dir,
                                                data_processor=data_processor_train,
                                                dataset_name=self.args.train_dataset_name,
                                                shuffle=self.args.shuffle
                                                )
            num_samples = len(train_dataset)
        if self.args.valid_dataset_name and self.args.do_train:
            valid_dataset = CodeDocLocalDataset(data_dir=self.args.data_dir,
                                                data_processor=data_processor,
                                                dataset_name=self.args.valid_dataset_name
                                                )
        if self.args.test_dataset_name and (self.args.do_test or self.args.do_predict):
            test_dataset = CodeDocLocalDataset(data_dir=self.args.data_dir,
                                               data_processor=data_processor,
                                               dataset_name=self.args.test_dataset_name,
                                               is_test=True
                                               )
        data_module = CodeDocDataModule(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            collate_fn_train=collate_fn_train,
            **vars(self.args)
        )

        # 定义模型
        model = CodeDocNERGBModelModule(
            ner_labels=ner_labels,
            num_samples=num_samples,
            **vars(self.args)
        )

        # 本地tensorboard
        logger = TensorBoardLogger(save_dir=self.args.save_model_dir, name='')
        # 定义trainer
        trainer = Trainer.from_argparse_args(self.args,
                                             weights_save_path=args.save_model_dir,
                                             logger=logger,
                                             enable_progress_bar=False,
                                             plugins=[LightningEnvironment()])

        if self.args.do_train:
            if self.args.ckpt_path is not None:
                model.load_state_dict(torch.load(self.args.ckpt_path)['state_dict'])
            trainer.fit(model, data_module)

        if self.args.do_test:
            trainer.test(model, data_module, ckpt_path='best' if self.args.ckpt_path is None else self.args.ckpt_path)

        if self.args.do_predict:
            predictions = trainer.predict(model, data_module,
                                          ckpt_path='best' if self.args.ckpt_path is None else self.args.ckpt_path)

            fw = open(self.args.predict_result_file, 'w')
            for prediction_batch in predictions:
                for uid, prediction_list, group_token_labels, group_word_labels, implicit_group_word_labels in prediction_batch:
                    fw.write('\t'.join([uid, json.dumps(prediction_list, ensure_ascii=False)]))
                    fw.write(f"\t{group_token_labels}")
                    fw.write(f"\t{group_word_labels}")
                    fw.write(f"\t{implicit_group_word_labels}")
                    fw.write('\n')
            fw.close()


def main(args):
    gpus = args.gpus
    if gpus > 1 and args.strategy is None:
        args.strategy = 'ddp'
    print(args)
    runner = LightningRunner(args)
    runner.run()


if __name__ == '__main__':
    # 添加conflict_handler，防止和trainer参数冲突
    parser = ArgumentParser(conflict_handler='resolve')
    parser = Trainer.add_argparse_args(parser)
    parser = CodeDocDataModule.add_argparse_args(parser)

    # Data Hyperparameters
    parser.add_argument('--data_dir', default='../../../data/examples/', type=str)
    parser.add_argument('--train_dataset_name', default='data.train.txt', type=str)
    parser.add_argument('--valid_dataset_name', default='data.val.txt', type=str)
    parser.add_argument('--test_dataset_name', default='data.test.txt', type=str)
    parser.add_argument('--label_file_name', default='labels.txt', type=str)
    parser.add_argument('--bbox_level', default='segment', type=str, help='word or segment')

    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--norm_bbox_height', default=512, type=int)
    parser.add_argument('--norm_bbox_width', default=256, type=int)

    parser.add_argument('--eval_set', default=0, type=int)

    parser.add_argument('--shuffle', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='是否shuffle',
                        default=False)
    parser.add_argument('--use_global_1d', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='use_global_1d', default=False)
    parser.add_argument('--use_image', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='use_image',
                        default=False)

    # Model Hyperparameters
    parser.add_argument('--pretrained_model_path', default='../../../data/pretrained_models/code-layout-chinese-base',
                        type=str)
    parser.add_argument('--num_hidden_layers', default=-1, type=int, help='默认使用12层Bert')
    parser.add_argument('--dropout', default=0.1, type=float)

    # Basic Training Control
    parser.add_argument('--to_onnx', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        default=False, help='是否导出ONNX')
    parser.add_argument('--to_onnx_expected_decimal', type=int, default=3, help='torch模型导出onnx后两者精度差距，越大越严格')
    parser.add_argument('--fp16_expected_decimal', type=int, default=2, help='onnx模型进行FP16优化后，两者精度差距，越大越严格')

    parser.add_argument('--do_train', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do train',
                        default=True)
    parser.add_argument('--do_test', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do test',
                        default=False)
    parser.add_argument('--do_predict', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do test',
                        default=False)
    parser.add_argument('--predict_result_file', default=None, type=str)
    parser.add_argument('--enable_aug', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='enable aug',
                        default=False)
    parser.add_argument('--grouping', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='标签分组',
                        default=True)
    parser.add_argument('--use_multi_dropout', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='',
                        default=True)
    parser.add_argument('--precision', default=32, type=int, )
    parser.add_argument('--num_nodes', default=1, type=int, )
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--strategy', default=None, type=str)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--val_test_batch_size', default=-1, type=int)
    parser.add_argument('--preprocess_workers', default=4, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--patience', default=50, type=int)

    parser.add_argument('--save_model_dir', default='lightning_logs', type=str)
    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--log_every_n_steps', default=1, type=int)
    parser.add_argument('--val_check_interval', default=1.0, type=float)  # int时多少个steps跑验证集,float 按照比例算
    parser.add_argument('--every_n_epochs', default=1, type=int)
    parser.add_argument('--keep_checkpoint_max', default=1, type=int)
    parser.add_argument('--deploy_path', default='', type=str)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--detect_anomaly', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='是否开启detect',
                        default=False)

    args = parser.parse_args()

    main(args)
