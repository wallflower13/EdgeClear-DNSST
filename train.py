import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, EarlyStopping, LearningRateMonitor
from pytorch_lightning.trainer.connectors import *
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import *
from metrics import *
from psnr_ssim import *
from copy import deepcopy
import tensorboardX

from loss.CL1 import L1_Charbonnier_loss, PSNRLoss
from loss.perceptual import PerceptualLoss2

from EdgeClear_DNSST import *
import argparse
import yaml

import lpips
from psnr_ssim import PSNR_LOSS, SSIM_LOSS

# Set seed
seed = 42
seed_everything(seed)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

logger = TensorBoardLogger(r'tb_logs', name='ECDNSST')
lr_monitor = LearningRateMonitor(logging_interval='step')


# 添加 RGB 转 Y 通道函数
def rgb_to_ycbcr(image):
    """将RGB图像转换为YCbCr格式，并返回Y通道"""
    if image.shape[1] != 3:
        raise ValueError("输入图像必须是RGB格式")

    # RGB到YCbCr的转换矩阵
    transform = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.169, -0.331, 0.500],
        [0.500, -0.419, -0.081]
    ], device=image.device).float()

    transform = transform.view(3, 3, 1, 1)

    # 应用转换
    result = F.conv2d(image, transform)

    # 添加偏移量
    result[:, 1:] += 0.5

    # 只返回Y通道
    return result[:, 0:1]


class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""

    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CoolSystem(pl.LightningModule):
    def __init__(self,
                 train_datasets,
                 train_bs,
                 test_datasets,
                 test_bs,
                 val_datasets,
                 val_bs,
                 dataset_type,
                 initlr,
                 weight_decay,
                 crop_size,
                 crop_size_test,
                 num_workers,
                 img_width=None,
                 img_height=None,
                 ):
        super(CoolSystem, self).__init__()

        # 训练/验证/测试数据集
        self.train_datasets = train_datasets
        self.train_batchsize = train_bs
        self.test_datasets = test_datasets
        self.test_batchsize = test_bs
        self.validation_datasets = val_datasets
        self.val_batchsize = val_bs
        self.dataset_type = dataset_type

        # 训练设置
        self.initlr = initlr
        self.weight_decay = weight_decay
        self.num_workers = num_workers

        self.crop_size = crop_size
        self.crop_size_test = crop_size_test

        # 模型定义
        self.model = Transformer(img_size=(crop_size, crop_size))

        # 损失函数
        self.loss_f = PSNRLoss()
        self.loss_l1 = nn.L1Loss()
        self.loss_per = PerceptualLoss2()

        # LPIPS损失函数
        self.lpips_loss = lpips.LPIPS(net='alex')

    def forward(self, x, y=None):

        y_list = self.model(x)

        var_list = [None] * len(y_list)

        diff_maps = []

        if y is not None:
            # 自行计算差异矩阵用于训练
            diff_map = torch.abs(y_list[0] - y)  # 简单计算Y和Ŷ之间的绝对差异
            loss_matrix = torch.mean(diff_map, dim=1, keepdim=True)  # 生成损失权重矩阵
            diff_matrices = (diff_map, loss_matrix)
        else:
            diff_matrices = (None, None)

        return y_list, var_list, diff_maps, diff_matrices

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initlr, betas=[0.9, 0.999])

        # 余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )

        # 返回优化器和学习率调度器配置
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # 监控指标
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        # 创建下采样的目标图像
        y1 = nn.functional.interpolate(y, scale_factor=0.5, mode='bicubic')
        y2 = nn.functional.interpolate(y, scale_factor=0.25, mode='bicubic')
        y3 = nn.functional.interpolate(y, scale_factor=0.125, mode='bicubic')

        # 模型前向传播 - 传入目标图像用于生成差异矩阵
        y_list, var_list, diff_maps, (diff_map, loss_matrix) = self.forward(x, y)

        # 将RGB图像转换为Y通道用于计算PSNR和SSIM
        y_y = rgb_to_ycbcr(y)
        y0_y = rgb_to_ycbcr(y_list[0])

        # 计算原有损失函数: PSNR(Y) + 10*SSIM(Y) - 5*LPIPS
        psnr_value = PSNR(y0_y, y_y)  # Y通道的PSNR
        ssim_value = SSIM(y0_y, y_y)  # Y通道的SSIM
        lpips_value = self.lpips_loss(y_list[0], y).mean()  # LPIPS

        # 计算差异矩阵损失 - 最小化高差异区域
        if loss_matrix is not None:
            # 使用权重损失 - 差异越大的区域权重越高
            weighted_l1 = torch.mean(loss_matrix * torch.abs(y_list[0] - y))
            diff_loss = weighted_l1 * 2.0  # 可调节的权重系数
        else:
            diff_loss = 0.0

        if batch_idx % 100 == 0:
            if diff_map is not None:
                try:
                    dm = diff_map[0].detach().cpu()

                    # 标准化为图像格式
                    if dm.dim() == 3 and dm.shape[0] == 1:
                        # 如果是[1,1,N]形状，将其重构为方形图像
                        if dm.shape[1] == 1:
                            data = dm.flatten()
                            side_len = int(math.ceil(math.sqrt(data.numel())))
                            padded = torch.zeros(side_len * side_len)
                            padded[:data.numel()] = data
                            dm = padded.reshape(1, side_len, side_len)

                        # 扩展到3通道
                        dm = dm.repeat(3, 1, 1)

                    # 确保值在有效范围内
                    dm = torch.clamp(dm, 0, 1)

                    # 添加到tensorboard(确保最终形状是[C,H,W])
                    if dm.dim() == 3 and dm.shape[0] in [1, 3] and dm.shape[1] > 1 and dm.shape[2] > 1:
                        self.logger.experiment.add_image('difference_map', dm, self.global_step)
                    else:
                        print(f"跳过差异图可视化: 形状不适合: {dm.shape}")
                except Exception as e:
                    print(f"差异图可视化失败: {e}")

                if len(diff_maps) > 0:
                    try:
                        # 选择第一个差异图
                        feature_diff_map = diff_maps[0]

                        if isinstance(feature_diff_map, torch.Tensor):
                            # 首先处理批次维度
                            if feature_diff_map.dim() == 4:
                                # 批次数据[B,C,H,W]或[B,H,W,C]，取第一个样本
                                fdm = feature_diff_map[0].detach().cpu()
                            else:
                                fdm = feature_diff_map.detach().cpu()

                            # 处理非标准尺寸
                            if fdm.dim() == 3 and fdm.shape[0] > 3:
                                # 如果第一个维度大于3，假定是[H,W,C]格式，需要转置
                                fdm = fdm.permute(2, 0, 1)
                                # 如果通道数大于3，取前3个通道或平均
                                if fdm.shape[0] > 3:
                                    fdm = fdm[:3]
                            elif fdm.dim() == 2:
                                # 2D张量转为单通道图像
                                fdm = fdm.unsqueeze(0)
                            elif fdm.dim() == 1 or (fdm.dim() == 3 and fdm.shape[1] == 1 and fdm.shape[2] <= 16):
                                # 处理向量或小型3D张量，重构为2D图像
                                data = fdm.flatten()
                                side_len = int(math.ceil(math.sqrt(data.numel())))
                                padded = torch.zeros(side_len * side_len)
                                padded[:data.numel()] = data
                                fdm = padded.reshape(1, side_len, side_len)

                            # 确保是3通道或单通道
                            if fdm.dim() == 3:
                                if fdm.shape[0] != 1 and fdm.shape[0] != 3:
                                    # 非1或3通道，转为单通道
                                    fdm = fdm.mean(dim=0, keepdim=True)

                                # 如果是单通道，复制为3通道以便彩色显示
                                if fdm.shape[0] == 1:
                                    fdm = fdm.repeat(3, 1, 1)

                            # 确保值在0-1范围内
                            fdm = torch.clamp(fdm, 0, 1)

                            # 添加到tensorboard(确保最终形状是[C,H,W]且C=3或1)
                            if fdm.dim() == 3 and fdm.shape[0] in [1, 3] and fdm.shape[1] > 1 and fdm.shape[2] > 1:
                                self.logger.experiment.add_image('feature_diff_map', fdm, self.global_step)
                            else:
                                print(f"跳过可视化: 形状无法转为标准图像格式: {fdm.shape}")
                        else:
                            print(f"差异图不是张量类型: {type(feature_diff_map)}")

                    except Exception as e:
                        print(
                            f"差异图可视化失败: {e}, 张量形状: {feature_diff_map.shape if hasattr(feature_diff_map, 'shape') else 'unknown'}")

        # 组合损失函数 (保持原有损失 + 差异矩阵损失)
        main_loss = -psnr_value - 10 * ssim_value + 5 * lpips_value + diff_loss

        # 记录指标
        self.log('train_loss', main_loss)
        self.log('psnr_y', psnr_value)
        self.log('ssim_y', ssim_value)
        self.log('lpips', lpips_value)
        self.log('diff_loss', diff_loss)

        return {'loss': main_loss}

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        b, c, h, w = x.size()

        # 滑动窗口处理
        tile = min(self.crop_size_test, h, w)
        tile_overlap = 32
        sf = 1

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E1 = torch.zeros(b, c, h * sf, w * sf).type_as(x)
        W1 = torch.zeros_like(E1)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = x[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch_list = self.model(in_patch)
                out_patch1 = out_patch_list[0]  # 获取主输出
                out_patch_mask1 = torch.ones_like(out_patch1)
                E1[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch1)
                W1[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask1)

        # 得到完整的预测图像
        y_hat = E1.div_(W1)
        y_hat = torch.clamp(y_hat, 0, 1)

        # 转换为Y通道进行评估
        y_y = rgb_to_ycbcr(y)
        y_hat_y = rgb_to_ycbcr(y_hat)

        # 计算三个指标
        psnr_value = PSNR(y_hat_y, y_y)  # Y通道的PSNR
        ssim_value = SSIM(y_hat_y, y_y)  # Y通道的SSIM
        lpips_value = self.lpips_loss(y_hat, y).mean()  # LPIPS

        # 打印验证时的PSNR和SSIM值
        print(f"\nValidation Batch {batch_idx}")
        print(f"PSNR(Y): {psnr_value:.4f}")
        print(f"SSIM(Y): {ssim_value:.4f}")

        val_loss = -psnr_value - 10 * ssim_value + 5 * lpips_value

        self.log('val_loss', val_loss)
        self.log('psnr', psnr_value)
        self.log('ssim', ssim_value)
        self.log('lpips', lpips_value)

        return {'val_loss': val_loss, 'psnr': psnr_value, 'ssim': ssim_value, 'lpips': lpips_value}

    def train_dataloader(self):
        # 加载白天和晚上的数据集
        day_dataset = RainDS_Dataset("D:\\RaindropClarity-main\\datasets\\DayRainDrop_Train",
                                     train=True, crop=True, size=self.crop_size)
        night_dataset = RainDS_Dataset("D:\\RaindropClarity-main\\datasets\\NightRainDrop_Train",
                                       train=True, crop=True, size=self.crop_size)

        full_dataset = torch.utils.data.ConcatDataset([day_dataset, night_dataset])
        print(f"合并后的总数据量: {len(full_dataset)}对图像")

        val_size = int(0.1 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        print(f"训练集大小: {train_size}对图像")
        print(f"验证集大小: {val_size}对图像")
        # 随机分割数据集
        train_dataset, _ = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batchsize,
            shuffle=True,
            num_workers=self.num_workers
        )
        return train_loader

    def val_dataloader(self):
        # 加载白天和晚上的数据集
        day_dataset = RainDS_Dataset("D:\\RaindropClarity-main\\datasets\\DayRainDrop_Train",
                                     train=True, crop=True, size=self.crop_size)
        night_dataset = RainDS_Dataset("D:\\RaindropClarity-main\\datasets\\NightRainDrop_Train",                                           train=True, crop=True, size=self.crop_size)

        full_dataset = torch.utils.data.ConcatDataset([day_dataset, night_dataset])

        val_size = int(0.1 * len(full_dataset))
        train_size = len(full_dataset) - val_size

        _, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.val_batchsize,
            shuffle=False,
            num_workers=self.num_workers
        )
        return val_loader

            
def cli_main():
    checkpoint_callback = ModelCheckpoint(
        monitor='psnr',
        filename='RainDrop-Base-epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}',
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=6,
        mode="max",
        save_last=True
    )

    # 创建早停回调
    early_stop_callback = EarlyStopping(
        monitor='psnr',
        patience=20,  # 如果20个epoch内PSNR没有提高，则停止训练
        mode='max',
        verbose=True
    )

    # 手动读取配置文件
    config_path = "config_raindrop.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if 'trainer' not in config:
        config['trainer'] = {}

    if 'gradient_clip_val' not in config['trainer']:
        config['trainer']['gradient_clip_val'] = 1.0  # 设置梯度范数上限为1.0

    if 'gradient_clip_algorithm' not in config['trainer']:
        config['trainer']['gradient_clip_algorithm'] = 'norm'  # 使用L2范数进行剪枝

    # 创建trainer
    trainer = pl.Trainer(**config['trainer'])
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(lr_monitor)
    trainer.callbacks.append(early_stop_callback)  # 添加早停回调
    trainer.logger = logger

    # 创建模型
    model = CoolSystem(**config['model'])

    # 开始训练
    trainer.fit(model)


if __name__ == '__main__':
    cli_main()
