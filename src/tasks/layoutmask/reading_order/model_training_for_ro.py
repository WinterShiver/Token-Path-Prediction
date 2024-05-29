
import json
import math
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from distutils.util import strtobool
from pytorch_lightning.plugins.environments import LightningEnvironment

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from src.dataset.augment_processor.aug_for_perspective_trans import AugForPerspectiveTrans
from src.dataset.augment_processor.aug_for_ro_segment_split import AugForROSegmentSplit
from src.dataset.code_doc_data_module import CodeDocDataModule
from src.dataset.code_doc_dataset import CodeDocReadingOrderDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from pytorch_lightning.utilities.argparse import add_argparse_args
from transformers import get_linear_schedule_with_warmup

from src.dataset.feature_processor.reading_order.ro_tl_tpp_processor import \
    CodeDocForTPPReadingOrderTLProcessor
from src.model.code_layout.configuration_code_layout import CodeLayoutConfig
from src.model.code_layout.modeling_code_layout import CodeLayoutForGlobalPointer
from src.modules.doc_utils import nnw_ro_tl_decode_beam
from src.metric.tpp_gb_entity_metric import TPPGB2NERMetric
from src.metric.reading_order_metric import ReadingOrderBLEUMetric, ReadingOrderAvgRelDistMetric
from src.utils.log_utils import create_logger_v2


class CodeDocLayoutROTLModelModule(pl.LightningModule):
    def __init__(self,
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
        self.config = CodeLayoutConfig.from_pretrained(self.hparams.pretrained_model_path, output_hidden_states=True)
        # self.config.enable_position_1d = True # 用来屏蔽1d位置编码

        ##  使用num_hidden_layers层Encoder
        if self.hparams.num_hidden_layers is not None and self.hparams.num_hidden_layers > 0:
            self.config.num_hidden_layers = self.hparams.num_hidden_layers

        self.config.num_labels = 1
        self.config.classifier_dropout = self.hparams.dropout
        self.config.use_last_layer_position_embedding_residual = self.hparams.use_last_layer_position_embedding_residual
        self.config.use_multi_dropout = self.hparams.use_multi_dropout

        self.model = CodeLayoutForGlobalPointer.from_pretrained(config=self.config,
                                                                pretrained_model_name_or_path=self.hparams.pretrained_model_path,
                                                                ignore_mismatched_sizes=True)

        # 设置metric
        self.valid_metric = TPPGB2NERMetric()
        self.bleu_metric = ReadingOrderBLEUMetric()
        self.ard_metric = ReadingOrderAvgRelDistMetric()
        if self.global_rank == 0:
            self.local_logger = create_logger_v2(log_dir=self.hparams.save_model_dir)
            self.local_logger.info(self.hparams)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):

        outputs = self(**batch)
        loss = outputs.loss

        steps = batch_idx

        # 这里在训练时打日志，由 log_every_n_steps 控制频率
        if self.global_rank == 0 and self.local_rank == 0 and (steps + 1) % self.trainer.log_every_n_steps == 0:
            lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
            self.local_logger.info(
                f"Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"Steps: {steps}, "
                f"Learning Rate {lr_scheduler.get_last_lr()[-1]:.7f}, "
                f"Train Loss: {loss:.5f}"
            )

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs.loss, outputs.logits

        entities_scores = torch.sigmoid(logits)
        entities_scores = torch.log(entities_scores)
        entities_scores = entities_scores.squeeze(1).detach().cpu().numpy()

        seg_se_maps = batch['seg_se_map']

        # seg paths，0开头,0结尾
        decode_results = nnw_ro_tl_decode_beam(entities_scores, seg_se_maps, beam_size=self.hparams.beam_size)

        true_order_list = [v.split('-') for v in batch['path_pair']]

        pred_order_list = []
        for decode_result in decode_results:
            p = []
            for i in range(0, len(decode_result) - 1):
                p.append('{}_{}'.format(decode_result[i], decode_result[i + 1]))
            pred_order_list.append(p)

        segment_word_ids = batch['segment_word_ids']
        self.valid_metric(pred_order_list, true_order_list)
        self.bleu_metric(pred_order_list, true_order_list, segment_word_ids)
        self.ard_metric(pred_order_list, true_order_list, segment_word_ids)

        return val_loss

    def validation_epoch_end(self, validation_step_outputs):

        metric_results = self.valid_metric.compute()
        bleu_results = self.bleu_metric.compute()
        ard_results = self.ard_metric.compute()

        val_loss = torch.stack(validation_step_outputs).mean()
        val_f1 = metric_results['f1']
        val_precision = metric_results['precision']
        val_recall = metric_results['recall']
        val_samples = metric_results['samples']
        val_bleu_seg = bleu_results['avg_bleu_seg']
        val_bleu_word = bleu_results['avg_bleu_word']
        val_ard_seg = ard_results['avg_dist_seg']
        val_ard_word = ard_results['avg_dist_word']

        self.log("val_f1", val_f1, prog_bar=True, on_epoch=True)
        self.log("val_precision", val_precision, prog_bar=True, on_epoch=True)
        self.log("val_recall", val_recall, prog_bar=True, on_epoch=True)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        self.log("val_bleu_seg", val_bleu_seg, prog_bar=True, on_epoch=True)
        self.log("val_bleu_word", val_bleu_word, prog_bar=True, on_epoch=True)
        self.log("val_ard_seg", val_ard_seg, prog_bar=True, on_epoch=True)
        self.log("val_ard_word", val_ard_word, prog_bar=True, on_epoch=True)

        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(
                f"**Validation** , Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"GlobalSteps: {self.global_step}, "
                f"val_loss: {val_loss:.5f}, "
                f"val_f1: {val_f1:.5f}, "
                f"val_precision: {val_precision:.5f}, "
                f"val_recall: {val_recall:.5f}, "
                f"val_bleu_seg: {val_bleu_seg:.5f}, "
                f"val_bleu_word: {val_bleu_word:.5f}, "
                f"val_ard_seg: {val_ard_seg:.5f}, "
                f"val_ard_word: {val_ard_word:.5f}, "
                f"val_samples: {val_samples}"
            )
        # *** 这个一定需要，不然会重复累积 *** #
        self.valid_metric.reset()
        self.bleu_metric.reset()
        self.ard_metric.reset()

    def test_step(self, batch, batch_idx):

        outputs = self(**batch)
        val_loss, logits = outputs.loss, outputs.logits

        entities_scores = torch.sigmoid(logits)
        entities_scores = torch.log(entities_scores)
        entities_scores = entities_scores.squeeze(1).detach().cpu().numpy()

        seg_se_maps = batch['seg_se_map']

        # seg paths，0开头,0结尾
        decode_results = nnw_ro_tl_decode_beam(entities_scores, seg_se_maps, beam_size=self.hparams.beam_size)

        true_order_list = [v.split('-') for v in batch['path_pair']]

        pred_order_list = []
        for decode_result in decode_results:
            p = []
            for i in range(0, len(decode_result) - 1):
                p.append('{}_{}'.format(decode_result[i], decode_result[i + 1]))
            pred_order_list.append(p)
        segment_word_ids = batch['segment_word_ids']
        self.valid_metric(pred_order_list, true_order_list)
        self.bleu_metric(pred_order_list, true_order_list, segment_word_ids)
        self.ard_metric(pred_order_list, true_order_list, segment_word_ids)

        return val_loss

    def test_epoch_end(self, outputs):

        metric_results = self.valid_metric.compute()
        bleu_results = self.bleu_metric.compute()
        ard_results = self.ard_metric.compute()

        val_loss = torch.stack(outputs).mean()
        val_f1 = metric_results['f1']
        val_precision = metric_results['precision']
        val_recall = metric_results['recall']
        val_samples = metric_results['samples']
        val_bleu_seg = bleu_results['avg_bleu_seg']
        val_bleu_word = bleu_results['avg_bleu_word']
        val_ard_seg = ard_results['avg_dist_seg']
        val_ard_word = ard_results['avg_dist_word']


        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(
                f"**Test** , "
                f"test_loss: {val_loss:.5f}, "
                f"test_f1: {val_f1:.5f}, "
                f"test_precision: {val_precision:.5f}, "
                f"test_recall: {val_recall:.5f}, "
                f"test_bleu_seg: {val_bleu_seg:.5f}, "
                f"test_bleu_word: {val_bleu_word:.5f}, "
                f"test_ard_seg: {val_ard_seg:.5f}, "
                f"test_ard_word: {val_ard_word:.5f}, "
                f"test_samples: {val_samples}"
            )

        # *** 这个一定需要，不然会重复累积 *** #
        self.valid_metric.reset()
        self.bleu_metric.reset()
        self.ard_metric.reset()

    # 用于模型测试
    def predict_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs.loss, outputs.logits

        uids = batch['uid']

        entities_scores = torch.sigmoid(logits)
        entities_scores = torch.log(entities_scores)
        entities_scores = entities_scores.squeeze(1).detach().cpu().numpy()

        seg_se_maps = batch['seg_se_map']
        # seg paths，0开头,0结尾
        decode_results = nnw_ro_tl_decode_beam(entities_scores, seg_se_maps, beam_size=self.hparams.beam_size)

        final_results = []
        for uid, decode_result in zip(uids, decode_results):
            # 去除首尾0；index-1，还原为真实segment id
            decode_result = [v - 1 for v in decode_result[1:-1]]
            final_results.append([uid, decode_result])

        return final_results

    def configure_callbacks(self):
        call_backs = []

        call_backs.append(LearningRateMonitor(logging_interval='step'))
        call_backs.append(EarlyStopping(monitor="val_bleu_seg", mode="max", patience=self.hparams.patience))
        call_backs.append(
            ModelCheckpoint(monitor="val_bleu_seg",
                            mode="max",
                            every_n_epochs=self.hparams.every_n_epochs,
                            filename='codedoc-{epoch}-{step}-{val_bleu_seg:.5f}',
                            save_top_k=self.hparams.keep_checkpoint_max
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
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
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

        # ner_labels = ClassLabel(names_file=os.path.join(self.args.data_dir, self.args.label_file_name))
        '''
        这里不同的任务使用不同的Processor
        '''
        augment_processors = []
        collate_fn_train = None
        if self.args.enable_aug:
            if self.args.seg_split_ratio > 0.0:
                augment_processors.append(AugForROSegmentSplit(prob=self.args.seg_split_ratio))
            if self.args.per_trains_ratio > 0.0:
                augment_processors.append(AugForPerspectiveTrans(prob=self.args.per_trains_ratio))

        data_processor_train = CodeDocForTPPReadingOrderTLProcessor(
            tokenizer=tokenizer,
            max_text_length=self.args.max_length,
            norm_bbox_height=self.args.norm_bbox_height,
            norm_bbox_width=self.args.norm_bbox_width,
            augment_processors=augment_processors,
            is_train=True,
            rate_shuffle_segments=self.args.rate_shuffle_segments,
            shuffle_when_evaluation=self.args.shuffle_when_evaluation,
            mask_prob=self.args.mask_prob
        )

        data_processor = CodeDocForTPPReadingOrderTLProcessor(
            tokenizer=tokenizer,
            max_text_length=self.args.max_length,
            norm_bbox_height=self.args.norm_bbox_height,
            norm_bbox_width=self.args.norm_bbox_width,
            is_train=False,
            rate_shuffle_segments=self.args.rate_shuffle_segments,
            shuffle_when_evaluation=self.args.shuffle_when_evaluation,
        )

        # 定义数据
        train_dataset, valid_dataset, test_dataset = None, None, None
        num_samples = 1
        if self.args.train_dataset_name and self.args.do_train:
            train_dataset = CodeDocReadingOrderDataset(data_dir=self.args.data_dir,
                                                       data_processor=data_processor_train,
                                                       dataset_name=self.args.train_dataset_name
                                                       )
            num_samples = len(train_dataset)
        if self.args.valid_dataset_name and self.args.do_train:
            valid_dataset = CodeDocReadingOrderDataset(data_dir=self.args.data_dir,
                                                       data_processor=data_processor,
                                                       dataset_name=self.args.valid_dataset_name
                                                       )
        if self.args.test_dataset_name and (self.args.do_test or self.args.do_predict):
            test_dataset = CodeDocReadingOrderDataset(data_dir=self.args.data_dir,
                                                      data_processor=data_processor,
                                                      dataset_name=self.args.test_dataset_name
                                                      )
        data_module = CodeDocDataModule(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            collate_fn_train=collate_fn_train,
            **vars(self.args)
        )

        # 定义模型
        model = CodeDocLayoutROTLModelModule(
            num_samples=num_samples,
            **vars(self.args)
        )
        # 定义日志
        # 本地tensorboard
        logger = TensorBoardLogger(save_dir=self.args.save_model_dir, name='')
        # 定义trainer
        trainer = Trainer.from_argparse_args(self.args,
                                             weights_save_path=args.save_model_dir,
                                             logger=logger,
                                             enable_progress_bar=False,
                                             plugins=[LightningEnvironment()])

        if self.args.do_train:
            trainer.fit(model, data_module, ckpt_path=None if self.args.ckpt_path is None else self.args.ckpt_path)

        if self.args.do_test:
            trainer.test(model, data_module, ckpt_path='best' if self.args.ckpt_path is None else self.args.ckpt_path)

        if self.args.do_predict:
            predictions = trainer.predict(model, data_module,
                                          ckpt_path='best' if self.args.ckpt_path is None else self.args.ckpt_path)

            fw = open(self.args.predict_result_file, 'w')
            for prediction_batch in predictions:
                for uid, prediction_list in prediction_batch:
                    fw.write('\t'.join([uid, json.dumps(prediction_list, ensure_ascii=False)]))
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
    parser.add_argument('--data_dir', default='../../../data/examples/reading_order', type=str)
    parser.add_argument('--train_dataset_name', default='data.train.txt', type=str)
    parser.add_argument('--valid_dataset_name', default='data.val.txt', type=str)
    parser.add_argument('--test_dataset_name', default='data.test.txt', type=str)

    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--norm_bbox_height', default=512, type=int)
    parser.add_argument('--norm_bbox_width', default=256, type=int)
    parser.add_argument('--mask_prob', default=0.3, type=float)
    parser.add_argument('--seg_split_ratio', default=0.3, type=float)
    parser.add_argument('--per_trains_ratio', default=0.3, type=float)
    parser.add_argument('--beam_size', default=4, type=int)

    parser.add_argument('--use_last_layer_position_embedding_residual', type=lambda x: bool(strtobool(x)), 
                        nargs='?', const=True, help='是否使用最后一层残差链接', default=True)
    parser.add_argument('--use_multi_dropout', type=lambda x: bool(strtobool(x)), 
                        nargs='?', const=True, help='是否使用multi_dropout', default=False)

    parser.add_argument('--shuffle', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='是否shuffle',
                        default=False)
    parser.add_argument('--use_global_1d', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='use_global_1d', default=False)
    parser.add_argument('--use_image', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='use_image',
                        default=False)
    parser.add_argument('--rate_shuffle_segments', default=0., type=float)
    parser.add_argument('--shuffle_when_evaluation', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='shuffle_when_evaluation', default=False)
                        

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
