import itertools
import json
from typing import List, Union, Any, Dict
import jieba
import pandas as pd
from datasets import ClassLabel

from torch.utils.data.dataset import T_co, IterableDataset
from transformers import BatchEncoding, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import logging
from src.utils.dist_utils import get_total_world_size, get_world_rank
from transformers.data.data_collator import _torch_collate_batch, tolist
from src.modules.utils import *


jieba.setLogLevel(logging.INFO)
logger = logging.getLogger(__name__)


class DocumentDirectoryDataset(Dataset):

    def __init__(self, data_dir, data_processor, cache_dir=".", dataset_name="pretrain", index_file="data.txt",
                 images_dir="images"):
        super(DocumentDirectoryDataset).__init__()

        self.data_processor = data_processor
        self.dataset_name = dataset_name
        self.index_file = index_file
        self.images_dir = images_dir
        self.samples = self.load_data(data_dir, cache_dir)

    def load_data(self, data_dir, cache_dir):
        cache_file = f"{self.__class__.__name__}_{self.dataset_name}.pkl"
        cache_path = osp.join(cache_dir, cache_file)

        if osp.exists(cache_path):
            data = torch.load(cache_path)
            logger.info(f"Loaded dataset from cache: {cache_path}, total: {len(data)}")
            return data

        self.samples = []
        for dir_name in os.listdir(data_dir):
            cur_dir = osp.join(data_dir, dir_name)
            if osp.isdir(cur_dir):
                logger.info(f"Loading dataset {dir_name} from {data_dir}")
                if osp.exists(osp.join(cur_dir, self.index_file)):
                    with open(osp.join(cur_dir, self.index_file), "r", encoding="utf-8") as f:
                        for line in f.readlines():
                            items = line.strip().split("\t")
                            data_sample = {"json_file": osp.join(cur_dir, items[1]), "source": dir_name,
                                           "image": osp.join(cur_dir, items[0])}
                            self.samples.append(data_sample)
        torch.save(self.samples, cache_path)
        logger.info(f"Loaded dataset from disk: {data_dir}, total: {len(self.samples)}")
        return self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> T_co:
        self.samples[index]['json'] = json.load(open(self.samples[index]['json_file'], "r", encoding="utf-8"))
        data_feature = self.data_processor.process(self.samples[index])
        data_feature["data_index"] = index
        return data_feature





class DocumentLocalDataset(Dataset):
    def __init__(self,
                 data_processor,
                 data_path):
        self._data_df = pd.read_csv(data_path)
        self._data = self._data_df.to_dict(orient='records')
        self._fields = self._data_df.columns
        self.data_processor = data_processor

    def __iter__(self):
        self.data_iter = iter(self._data)
        return self

    def __next__(self):
        item = next(self.data_iter)
        return self.data_processor.process(item)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        item = self._data.iloc[index]
        return self.data_processor.process(item)


class CodeDocPretrainDataset(IterableDataset):
    '''
    CodeDoc预训练数据集专用Dataset

    使用前请去Nas的multimodal/datasets/chinese目录使用 python z_pretrain_data_process.py --task info 生成预训练数据
    '''

    def __init__(self,
                 data_dir,
                 data_processor,
                 dataset_name="pretrain",
                 shuffle=False
                 ):

        super(DocumentDirectoryDataset).__init__()

        self.data_processor = data_processor
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.samples = self.load_data(data_dir)

    def load_data(self, data_dir):
        '''
        手动切片支持 单机多卡；每个卡对应的主进程，主进程init时读取数据分片
        多机多卡没验证过 TODO
        '''
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        world_rank = torch.distributed.get_rank() if torch.cuda.is_available() else 0
        self.samples = []
        data_sum = 0
        with open(osp.join(data_dir, self.dataset_name), "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data_sum += 1
                if idx % world_size != world_rank:
                    continue
                img_path, json_path = line.strip().split('\t')
                data_sample = {
                    "json_file": osp.join(data_dir, json_path)
                    , "source": json_path
                    , "image": osp.join(data_dir, img_path)
                }
                self.samples.append(data_sample)
        if self.shuffle:
            random.shuffle(self.samples)

        logger.info(
            f"Loaded dataset from disk: {data_dir}, DeviceCount: {world_size}, DeviceId: {world_rank}, DataSize: {len(self.samples)} / {data_sum}")

        return self.samples

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        '''
        这里是dataLoader设置的num_worker多进程读取数据，每卡都对应自己的数据处理workers
        '''
        worker_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id  # 从0开始

        # 不同的worker读取数据分片，注意这个是在以上load data基础上再次划分
        this_iter = itertools.islice(self.samples, worker_id, None, worker_num)

        self.data_iter = this_iter

        return self

    def __next__(self):

        sample = next(self.data_iter)
        obj = json.load(open(sample['json_file'], "r", encoding="utf-8"))
        data_feature = self.data_processor.process({'json': obj})

        return data_feature


class CodeDocLocalIterableDataset(IterableDataset):
    r"""

    """

    def __init__(self,
                 data_dir: str,
                 data_processor,
                 dataset_name="data.train.txt",
                 shuffle=False
                 ):

        super(CodeDocLocalIterableDataset).__init__()

        self.data_processor = data_processor
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.samples = self.load_data(data_dir)

        # Set all input args as attributes
        self.__dict__.update(locals())

    def load_data(self, data_dir):
        '''
        手动切片支持 单机多卡；每个卡对应的主进程，主进程init时读取数据分片
        多机多卡没验证过 TODO
        '''

        world_size = get_total_world_size()
        world_rank = get_world_rank()
        datas = []
        data_sum = 0
        file_name = osp.join(data_dir, self.dataset_name)
        with open(file_name, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data_sum += 1
                # if idx % world_size != world_rank:
                #     continue
                img_path, json_path = line.strip().split('\t')
                data_sample = {
                    "json_path": osp.join(data_dir, json_path)
                    # , "source": json_path
                    , "image_path": osp.join(data_dir, img_path)
                }
                datas.append(data_sample)

        # 先shuffle，再选择；TODO 需要测试下 多卡是否每个线程保持 互斥
        if self.shuffle:
            random.shuffle(datas)

        for i in range(min(len(datas), 3)):
            d = datas[i]
            logger.info(f"DeviceCount: {world_size}, DeviceId: {world_rank}, {d['json_path']}, {self.shuffle}")

        '''
        然后选择本卡需要处理的数据
        '''
        samples = [v for i, v in enumerate(datas) if i % world_size == world_rank]

        logger.info(
            f"Loaded dataset from disk: {file_name}, DeviceCount: {world_size}, DeviceId: {world_rank}, DataSize: {len(samples)} / {data_sum}")

        return samples

    def __len__(self):
        return len(self.samples)

    def count(self):
        # 增加这个函数是为了和ODPS表数据的count函数保持一致
        print(f"count {len(self.samples)}")
        return len(self.samples)

    def __iter__(self):
        '''
        这里是dataLoader设置的num_worker多进程读取数据，每卡都对应自己的数据处理workers
        '''
        worker_num = torch.utils.data.get_worker_info().num_workers if torch.utils.data.get_worker_info() else 1
        worker_id = torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0

        # 不同的worker读取数据分片，注意这个是在以上load data基础上再次划分
        this_iter = itertools.islice(self.samples, worker_id, None, worker_num)

        self.data_iter = this_iter

        return self

    def __next__(self):

        sample = next(self.data_iter)
        try:
            data_feature = self.data_processor.process({
                'json': sample['json_path'],
                'image': sample['image_path']
            })
        except Exception as e:
            logger.error('Error, {}'.format(json.dumps(sample, ensure_ascii=False)))
            logger.error('Error, {}'.format(e))

        return data_feature


class CodeDocLocalDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 data_processor,
                 dataset_name="data.train.txt",
                 shuffle=False,
                 is_test=False,
                 ):
        super(CodeDocLocalDataset).__init__()

        self.data_processor = data_processor
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.samples = self.load_data(data_dir)
        self.is_test = is_test

    def load_data(self, data_dir):
        '''
        单卡
        '''
        datas = []
        file_name = osp.join(data_dir, self.dataset_name)

        with open(file_name, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                line = line.strip()
                try:
                    tmp = line.split('\t')
                    img_path, json_path = tmp[0], tmp[1]
                except Exception as e:
                    tmp = line.split(' ')
                    img_path, json_path = tmp[0], tmp[1]
                img_path = img_path.strip()
                json_path = json_path.strip()
                data_sample = {
                    "json": osp.join(data_dir, json_path), 
                    "image": osp.join(data_dir, img_path),
                }
                datas.append(data_sample)

        logger.info(
            f"Loaded dataset from disk: {file_name}, DataSize: {len(datas)}")

        datas = pd.DataFrame(data=datas)
        return datas

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples.iloc[index]

        results = self.data_processor.process({
            'json': item['json'],
            'image': item['image']
        })

        return results


class CodeDocReadingOrderDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 data_processor,
                 dataset_name="data.train.txt",
                 ):
        super(CodeDocLocalDataset).__init__()

        self.data_processor = data_processor
        self.dataset_name = dataset_name
        self.samples = self.load_data(data_dir)

    def load_data(self, data_dir):
        '''
        单卡
        '''
        datas = []
        file_name = osp.join(data_dir, self.dataset_name)

        with open(file_name, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                # obj = json.loads(line)
                # data_sample = {
                #     "json": line.strip()
                # }
                datas.append(line.strip())
                if idx % 10000 == 0:
                    logger.info(
                        f"Loaded dataset from disk: {file_name}, Idx: {idx}")

        logger.info(
            f"Loaded dataset from disk: {file_name}, DataSize: {len(datas)}")

        return datas

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_str = self.samples[index]
        try:
            results = self.data_processor.process({
                'json': sample_str
            })
        except Exception as e:
            logger.error(sample_str)
            logger.error(e)

        return results
