
from datasets import ClassLabel
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl

import logging

logger = logging.getLogger(__name__)


class CodeDocDataModule(pl.LightningDataModule):
    r"""

    """

    def __init__(self,

                 data_dir,

                 train_dataset,
                 valid_dataset,
                 test_dataset,

                 ner_label_file: str = None,
                 cls_label_file: str = None,

                 batch_size: int = None,
                 val_test_batch_size: int = None,

                 collate_fn=None,
                 collate_fn_train=None,
                 preprocess_workers: int = 4,
                 shuffle: bool = True,
                 **kwargs,
                 ):
        super().__init__()
        # Set all input args as attributes
        self.__dict__.update(locals())
        
        self.ner_label_map = None
        self.cls_label_map = None

        if val_test_batch_size is None or val_test_batch_size < 1:
            self.val_test_batch_size = batch_size

        # 加载label
        if ner_label_file and len(ner_label_file) > 0:
            self.ner_label_map = ClassLabel(names_file=ner_label_file)

        if cls_label_file and len(cls_label_file) > 0:
            self.cls_label_map = ClassLabel(names_file=cls_label_file)

    # 下方代码可以不用改
    def train_dataloader(self):
        shuffle = self.shuffle
        if isinstance(self.train_dataset, IterableDataset):
            shuffle = False
        data_loader = DataLoader(dataset=self.train_dataset,
                                 batch_size=self.batch_size,
                                 collate_fn=self.collate_fn_train,
                                 pin_memory=False,
                                 persistent_workers=True,
                                 shuffle=shuffle,
                                 num_workers=self.preprocess_workers)

        return data_loader

    def val_dataloader(self):
        if self.valid_dataset is not None:
            data_loader = DataLoader(dataset=self.valid_dataset,
                                     batch_size=self.val_test_batch_size,
                                     collate_fn=self.collate_fn,
                                     pin_memory=False,
                                     persistent_workers=True,
                                     shuffle=False,
                                     num_workers=self.preprocess_workers)
            return data_loader
        else:
            return None

    def test_dataloader(self):
        if self.test_dataset is not None:
            data_loader = DataLoader(dataset=self.test_dataset,
                                     batch_size=self.val_test_batch_size,
                                     collate_fn=self.collate_fn,
                                     pin_memory=False,
                                     persistent_workers=True,
                                     shuffle=False,
                                     num_workers=self.preprocess_workers)
            return data_loader
        else:
            return None

    # 为了方便测试，请定义好predict_dataloader
    def predict_dataloader(self):
        if self.test_dataset is not None:
            data_loader = DataLoader(dataset=self.test_dataset,
                                     batch_size=self.val_test_batch_size,
                                     collate_fn=self.collate_fn,
                                     shuffle=False,
                                     num_workers=self.preprocess_workers)
            return data_loader
        else:
            return None
