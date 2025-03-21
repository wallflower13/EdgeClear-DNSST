import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os, sys
import random
from PIL import Image
from torchvision.utils import make_grid

random.seed(2)
np.random.seed(2)


class RainDS_Dataset(data.Dataset):
    def __init__(self, path, train, crop=False, size=240, format='.png', dataset_type='all'):
        super(RainDS_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.crop = crop
        self.format = format
        self.gt_path = os.path.join(path, 'Clear')
        self.gt_list = []
        self.rain_list = []

        drop_path = os.path.join(path, 'Drop')

        # 获取drop文件夹中的所有子文件夹
        drop_folders = [f for f in os.listdir(drop_path) if os.path.isdir(os.path.join(drop_path, f))]

        if train:
            # 使用90%的文件夹作为训练集
            selected_folders = drop_folders[:int(len(drop_folders) * 0.9)]
        else:
            # 使用10%的文件夹作为验证集
            selected_folders = drop_folders[int(len(drop_folders) * 0.9):]

        # 遍历选定的文件夹
        for folder in selected_folders:
            drop_folder_path = os.path.join(drop_path, folder)
            clear_folder_path = os.path.join(self.gt_path, folder)

            # 确保对应的清晰图像文件夹存在
            if not os.path.exists(clear_folder_path):
                continue

            # 获取文件夹中的所有图片
            drop_images = [os.path.join(drop_folder_path, f) for f in os.listdir(drop_folder_path)
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
            clear_images = [os.path.join(clear_folder_path, f) for f in os.listdir(clear_folder_path)
                            if f.endswith(('.png', '.jpg', '.jpeg'))]

            # 确保文件数量匹配
            min_count = min(len(drop_images), len(clear_images))
            drop_images = sorted(drop_images)[:min_count]
            clear_images = sorted(clear_images)[:min_count]

            self.rain_list.extend(drop_images)
            self.gt_list.extend(clear_images)

        # 确保数据集不为空
        assert len(self.rain_list) > 0, f"在{drop_path}中未找到图像"

        print(f"{'训练' if train else '验证'}数据集已加载: {len(self.rain_list)}对图像")

    def __getitem__(self, index):
        rain = Image.open(self.rain_list[index])
        clear_path = self.gt_list[index]
        clear = Image.open(clear_path)
        name = self.rain_list[index].split('/')[-1].split(".")[0]
        if not isinstance(self.size, str) and self.crop:
            i, j, h, w = tfs.RandomCrop.get_params(clear, output_size=(self.size, self.size))
            clear = FF.crop(clear, i, j, h, w)
            rain = FF.crop(rain, i, j, h, w)

        if self.train:
            rain, clear = self.augData(rain.convert("RGB"), clear.convert("RGB"))
        else:
            rain = tfs.ToTensor()(rain.convert("RGB"))
            clear = tfs.ToTensor()(clear.convert("RGB"))
        return rain, clear, name

    def augData(self, data, target):
        rand_hor = random.randint(0, 1)
        rand_rot = random.randint(0, 3)
        data = tfs.RandomHorizontalFlip(rand_hor)(data)
        target = tfs.RandomHorizontalFlip(rand_hor)(target)
        if rand_rot:
            data = FF.rotate(data, 90 * rand_rot)
            target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.rain_list)

