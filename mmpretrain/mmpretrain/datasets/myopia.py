# Copyright (c) OpenMMLab. All rights reserved.
import xml.etree.ElementTree as ET
from typing import List, Optional, Union

from mmengine import get_file_backend, list_from_file
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .base_dataset import expanduser
from .categories import SA_CATEGORIES, P_CATEGORIES
from .multi_label import MultiLabelDataset
from .base_dataset import BaseDataset
import pandas as pd


@DATASETS.register_module()
class MYOPIA(MultiLabelDataset):
    """MYOPIA Dataset"""

    METAINFO = {'classes': SA_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 split: str = 'trainval',
                 image_set_path: str = '',
                 data_prefix: Union[str, dict] = dict(
                     img_path='JPEGImages', ann_path='Deprecated'),
                 test_mode: bool = False,
                 metainfo: Optional[dict] = None,
                 **kwargs):

        self.backend = get_file_backend(data_root, enable_singleton=True)

        if split:
            splits = ['train', 'val', 'trainval', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"
            self.split = split

            if not data_prefix:
                data_prefix = dict(
                    img_path='JPEGImages', ann_path='Deprecated')
            if not image_set_path:
                current = ''
                image_set_path = self.backend.join_path(
                    'ImageSets', 'Main', f'{current}', f'{split}.csv')

        # To handle the BC-breaking
        if (split == 'train' or split == 'trainval') and test_mode:
            logger = MMLogger.get_current_instance()
            logger.warning(f'split="{split}" but test_mode=True. '
                           f'The {split} set will be used.')

        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))
        assert isinstance(data_prefix, dict) and 'img_path' in data_prefix, \
            '`data_prefix` must be a dict with key img_path'

        if (split and split not in ['val', 'test']) or not test_mode:
            assert 'ann_path' in data_prefix and data_prefix[
                'ann_path'] is not None, \
                '"ann_path" must be set in `data_prefix`' \
                'when validation or test set is used.'

        self.data_root = data_root
        self.image_set_path = self.backend.join_path(data_root, image_set_path)

        super().__init__(
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        df = pd.read_csv(self.image_set_path)
        lesion_cols = {'S0': 0, 'S1': 1, 'A0': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6}
        data_list = []
        for i in range(0, len(df)):
            row = df.iloc[i]
            img_path = self.backend.join_path(self.img_prefix, row['filename'])
            labels = set()
            for lesion, index in lesion_cols.items():
                if row[lesion] != 0:
                    labels.add(index)

            info = dict(img_path=img_path, gt_label=list(labels), )
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Prefix of dataset: \t{self.data_root}',
            f'Path of image set: \t{self.image_set_path}',
            f'Prefix of images: \t{self.img_prefix}',
        ]

        return body


@DATASETS.register_module()
class MYOPIA_P(MultiLabelDataset):
    """MYOPIA_P Dataset"""

    METAINFO = {'classes': P_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 split: str = 'trainval',
                 image_set_path: str = '',
                 data_prefix: Union[str, dict] = dict(
                     img_path='Deprecated', ann_path='Deprecated'),
                 test_mode: bool = False,
                 metainfo: Optional[dict] = None,
                 **kwargs):

        self.backend = get_file_backend(data_root, enable_singleton=True)

        if split:
            splits = ['train', 'val', 'trainval', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"
            self.split = split

            if not data_prefix:
                data_prefix = dict(
                    img_path='Deprecated', ann_path='Deprecated')
            if not image_set_path:
                current = ''
                image_set_path = self.backend.join_path(
                    'ImageSets', 'Main', f'{current}', f'{split}.csv')

        # To handle the BC-breaking
        if (split == 'train' or split == 'trainval') and test_mode:
            logger = MMLogger.get_current_instance()
            logger.warning(f'split="{split}" but test_mode=True. '
                           f'The {split} set will be used.')

        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))
        assert isinstance(data_prefix, dict) and 'img_path' in data_prefix, \
            '`data_prefix` must be a dict with key img_path'

        if (split and split not in ['val', 'test']) or not test_mode:
            assert 'ann_path' in data_prefix and data_prefix[
                'ann_path'] is not None, \
                '"ann_path" must be set in `data_prefix`' \
                'when validation or test set is used.'

        self.data_root = data_root
        self.image_set_path = self.backend.join_path(data_root, image_set_path)

        super().__init__(
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        df = pd.read_csv(self.image_set_path)
        lesion_cols = {'P0': 0, 'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4}
        data_list = []
        for i in range(0, len(df)):
            row = df.iloc[i]
            img_path = self.backend.join_path(self.img_prefix, row['filename'])
            labels = set()
            for lesion, index in lesion_cols.items():
                if row[lesion] != 0:
                    labels.add(index)

            info = dict(img_path=img_path, gt_label=list(labels), )
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Prefix of dataset: \t{self.data_root}',
            f'Path of image set: \t{self.image_set_path}',
            f'Prefix of images: \t{self.img_prefix}',
        ]

        return body