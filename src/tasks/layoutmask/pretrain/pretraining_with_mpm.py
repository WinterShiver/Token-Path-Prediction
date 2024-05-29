
import torch
import torch.multiprocessing as mp
import pytorch_lightning as pl
from argparse import ArgumentParser
from distutils.util import strtobool
from pytorch_lightning.plugins.environments import LightningEnvironment

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.argparse import add_argparse_args
from transformers import get_linear_schedule_with_warmup

from src.utils.log_utils import create_logger_v2
#数据和预处理
from src.dataset.code_doc_data_module import CodeDocDataModule
from src.dataset.code_doc_dataset import CodeDocLocalDataset
from src.dataset.code_doc_collator import DocumentWwmMlmCollator
from src.dataset.feature_processor.layoutmask.pretrain_processor import LayoutmaskPretrainProcessor
# LayoutMask模型结构
from src.model.layoutmask.configuration_layoutmask import LayoutMaskConfig
from src.model.layoutmask.modeling_layoutmask import LayoutMaskModelForPretrainingMPM


class LayoutMaskPretrainModule(pl.LightningModule):
    def __init__(self,
                 num_samples: int = 100,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-6,
                 warmup_ratio: float = 0.05,
                 **kargs):

        super().__init__()
        self.save_hyperparameters(ignore=['collate_fn', 'tokenizer']) #参数都存入self.hparams

        self.config = LayoutMaskConfig().from_pretrained(self.hparams.pretrained_model_path, output_hidden_states=True)
        
        self.model = LayoutMaskModelForPretrainingMPM.from_pretrained(config=self.config,
                                                                     pretrained_model_name_or_path=self.hparams.pretrained_model_path,
                                                                     ignore_mismatched_sizes=True)

        if self.global_rank == 0:
            self.local_logger = create_logger_v2(log_dir=self.hparams.save_model_dir)
            self.local_logger.info(self.hparams)

    def forward(self, **inputs):

        model_output = self.model(**inputs)
        return model_output

    def training_step(self, batch, batch_idx):

        outputs = self(**batch)
        mlm_loss = outputs['mlm_loss']
        mpm_loss = outputs['mpm_loss']

        loss = mlm_loss * self.hparams.mlm_task_weight + mpm_loss * self.hparams.mpm_task_weight

        steps = self.global_step

        # 这里在训练时打日志，由 log_every_n_steps 控制频率
        if self.global_rank == 0 and self.local_rank == 0 and (steps + 1) % self.trainer.log_every_n_steps == 0:
            # 本地日志输入
            lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1]
            self.local_logger.info(
                f"GlobalSteps: {self.global_step + 1}/{self.trainer.max_steps}, "
                f"Learning Rate {lr:.7f}, "
                f"Train Loss: {loss.detach().item():.5f},  "
                f"MLM Loss: {mlm_loss.detach().item():.5f},  "
                f"MPM Loss: {mpm_loss.detach().item():.5f}."
            )
        if (steps + 1) % self.trainer.log_every_n_steps == 0:
            self.log("steps", steps + 1, logger=False)
            # tensorboard logger
            #  name、y-value、x-value
            self.logger.experiment.add_scalar("MLM_Loss/Train", mlm_loss.detach().item(), int(steps + 1))
            self.logger.experiment.add_scalar("MPM_Loss/Train", mpm_loss.detach().item(), int(steps + 1))

        return loss

    def configure_callbacks(self):
        call_backs = []

        call_backs.append(LearningRateMonitor(logging_interval='step'))
        call_backs.append(ModelCheckpoint(
            save_top_k=self.hparams.keep_checkpoint_max,
            monitor="steps",
            mode="max",
            filename="codedoc-pretrained-{steps}",
            every_n_train_steps=self.hparams.save_every_n_train_steps
        ))

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
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        num_samples = self.hparams.num_samples
        batch_size = self.hparams.batch_size
        tb_size = batch_size * max(1, self.trainer.devices)
        self.total_steps = self.hparams.max_steps
        if self.global_rank == 0:
            self.local_logger.info(f"num_samples is: {num_samples}")
            self.local_logger.info(f"total_steps is: {self.total_steps}")
            self.local_logger.info(f"batch_size is: {batch_size}*{max(1, self.trainer.devices)}={tb_size}")

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        return add_argparse_args(cls, parent_parser, **kwargs)


class LightningRunner:
    def __init__(self, args):
        self.args = args

    def run(self):
        # 设置随机种子
        '''
        mp.set_start_method('spawn')
        - ddp模式不加这个会oom，原因未知，见：https://github.com/pytorch/pytorch/issues/13246#issuecomment-1088215408
        - spawn模式preprocess_workers数不能太多，内存占用会多，但可避免mem leak
        '''
        mp.set_start_method('spawn')

        pl.seed_everything(self.args.seed)

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.args.pretrained_model_path)

        collate_fn_train = DocumentWwmMlmCollator(
            tokenizer=tokenizer,
            mlm_prob=self.args.mlm_prob,
            use_chs_wwm=self.args.use_chs_wwm
        )
        
        augment_processors = []
        #if self.args.enable_aug:
        #    augment_processors.append(AugForSegmentSplit(prob=0.2))
        #    augment_processors.append(AugForRndMove(prob=0.95))
        data_processor_train = LayoutmaskPretrainProcessor(tokenizer,
                                                           max_text_length=self.args.max_length,
                                                           norm_bbox_height=self.args.norm_bbox_height,
                                                           norm_bbox_width=self.args.norm_bbox_width,
                                                           augment_processors=augment_processors,
                                                           mpm_prob= self.args.mpm_prob,
                                                           )
        train_dataset = CodeDocLocalDataset(data_dir=self.args.data_dir,
                                            data_processor=data_processor_train,
                                            dataset_name=self.args.train_dataset_name,
                                            shuffle=self.args.shuffle
                                            )

        data_module = CodeDocDataModule(
            train_dataset=train_dataset,
            valid_dataset=None,
            test_dataset=None,
            collate_fn_train=collate_fn_train,
            **vars(self.args)
        )
        train_loader = data_module.train_dataloader()

        num_samples = len(train_dataset)

        model = LayoutMaskPretrainModule(
            num_samples=num_samples,
            **vars(self.args)
        )
         
        # 本地tensorboard
        logger = TensorBoardLogger(save_dir=self.args.save_model_dir, name='')

        trainer = Trainer.from_argparse_args(self.args,
                                             weights_save_path=args.save_model_dir,
                                             logger=logger,
                                             enable_progress_bar=False,
                                             plugins=[LightningEnvironment()])

        if self.args.do_train:
            trainer.fit(model, data_module)

def main(args):
    gpus = args.gpus
    if gpus > 1 and args.strategy is None:
        args.strategy = 'ddp'
    print(args)
    runner = LightningRunner(args)
    runner.run()


if __name__ == '__main__':
    '''
    DistributedDataParallel(DDP) 工作方式:
        每个节点的每个 GPU 拥有自己独立的进程
        每个 GPU 只能看到整体数据集的一个子集，并且一直只能看到那个子集
        每个进程都会初始化模型
        每个进程执行完整的 forward 和 backward 过程
        梯度会在所有进程里做同步和取平均
        每个进程都会更新自己的 optimizer
    '''
    # 添加conflict_handler，防止和trainer参数冲突
    parser = ArgumentParser(conflict_handler='resolve')
    parser = Trainer.add_argparse_args(parser)
    parser = CodeDocDataModule.add_argparse_args(parser)

    # Data Hyperparameters
    parser.add_argument('--data_dir', default='../../data/cdip_sample', type=str, help='数据集文件夹')
    parser.add_argument('--train_dataset_name', default='data.train.txt', type=str, help='数据集')
    parser.add_argument('--bbox_level', default='segment', type=str, help='word or segment')
    parser.add_argument('--norm_bbox_height', default=512, type=int)
    parser.add_argument('--norm_bbox_width', default=256, type=int)

    parser.add_argument('--shuffle', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='是否shuffle',
                        default=False)

    # Model Hyperparameters
    parser.add_argument('--pretrained_model_path', default='/path/to/xlm-roberta-base.bin',
                        type=str)
    parser.add_argument('--tokenizer_path', default='/models/tokenizers/chinese')

    #模型参数
    parser.add_argument('--use_large_model', action="store_true", default=False)
    parser.add_argument('--use_aug', action="store_true", default=False)
    #mlm
    parser.add_argument('--mlm_task_weight', default=1, type=float)
    parser.add_argument('--mlm_prob', type=float, default=0.15, metavar='mp', help='mpm prob')
    parser.add_argument('--max_length', default=512, type=int)
    #mpm
    parser.add_argument('--mpm_task_weight', default=0.0, type=float)
    parser.add_argument('--mpm_prob', type=float, default=0.0, help='mpm prob')

    parser.add_argument('--segment_split_prob', default=0.0, type=float)
    parser.add_argument('--use_chs_wwm', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='use wwm for chinese',
                        default=False)

    # Basic Training Control
    parser.add_argument('--do_train', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do train',
                        default=True)
    parser.add_argument('--precision', default=32, type=int, )

    parser.add_argument('--num_nodes', default=1, type=int, )
    parser.add_argument('--strategy', default=None, type=str)  # 多卡 ddp
    parser.add_argument('--gpus', default=0, type=int)

    parser.add_argument('--max_steps', default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)

    parser.add_argument('--preprocess_workers', default=4, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)

    parser.add_argument('--save_model_dir', default='lightning_logs', type=str)

    parser.add_argument('--log_every_n_steps', default=1, type=int)
    parser.add_argument('--save_every_n_train_steps', default=10, type=int)
    parser.add_argument('--keep_checkpoint_max', default=5, type=int)

    parser.add_argument('--seed', default=2022, type=int)


    args = parser.parse_args()
    
    main(args)


