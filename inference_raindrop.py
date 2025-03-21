import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import yaml
from EdgeClear_DNSST import Transformer
import time
import cv2
import torch.nn as nn 
# 命令行参数
parser = argparse.ArgumentParser(description='雨滴去除推理')
parser.add_argument('--checkpoint', type=str, default='best_model.ckpt', help='模型检查点路径')
parser.add_argument('--input_dir', type=str, default='D:/RaindropClarity-main/datasets/RainDrop', help='输入图片目录')
parser.add_argument('--output_dir', type=str, default='results', help='结果输出目录')
parser.add_argument('--tile', type=int, default=128, help='滑动窗口大小')
parser.add_argument('--tile_overlap', type=int, default=64, help='重叠区域大小')
parser.add_argument('--device', type=str, default='cuda:0', help='使用的设备')
parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
parser.add_argument('--save_comparison', action='store_true', help='是否保存对比图')
args = parser.parse_args()

# 确保输出目录存在
os.makedirs(args.output_dir, exist_ok=True)

# 设置设备
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


def reduce_raindrop_module_strength(module, factor=0.5):
    """降低RaindropFeatureModulation模块的增强强度"""
    for i in range(len(module.channel_attn)):
        if isinstance(module.channel_attn[i], nn.Conv2d):
            # 初始化为较小的值
            with torch.no_grad():
                module.channel_attn[i].weight.data.mul_(factor)
    
    for i in range(len(module.spatial_attn)):
        if isinstance(module.spatial_attn[i], nn.Conv2d):
            # 初始化为较小的值
            with torch.no_grad():
                module.spatial_attn[i].weight.data.mul_(factor)
def load_model(checkpoint_path):
    """加载训练模型"""
    print(f"从 {checkpoint_path} 加载模型...")
    
    try:
        # 创建模型实例
        model = Transformer(img_size=(args.tile, args.tile))
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 检查是否是Lightning模型权重
        if 'state_dict' in checkpoint:
            # 提取实际模型权重
            state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.'):
                    state_dict[key[6:]] = value
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # 激活边缘增强和多尺度模块
        print("启用边缘增强和多尺度注意力模块...")
        # 启用所有边缘增强器
        model.edge_enhancer0.enabled = False
        model.edge_enhancer1.enabled = False
        model.edge_enhancer2.enabled = False
        model.edge_enhancer3.enabled = False
        
        # 设置gamma参数 - 控制边缘增强强度
        model.edge_enhancer0.gamma.data.fill_(0.1)
        model.edge_enhancer1.gamma.data.fill_(0.1)
        model.edge_enhancer2.gamma.data.fill_(0.15)
        model.edge_enhancer3.gamma.data.fill_(0.2)
        
        # 启用所有多尺度注意力模块
        model.pyramid_attn0.enabled = True
        model.pyramid_attn1.enabled = True
        model.pyramid_attn2.enabled = True
        model.pyramid_attn3.enabled = True
        print("边缘增强和多尺度注意力模块已成功配置！")
        
    except Exception as e:
        print(f"加载模型出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    model = model.to(device)
    model.eval()
    return model

def process_image(model, img_path):
    """使用滑动窗口处理单张图像"""
    # 加载图像
    img = Image.open(img_path).convert('RGB')
    # 记录原始大小
    original_size = img.size
    
    # 转换为张量
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度
    
    # 获取图像尺寸
    _, c, h, w = img_tensor.size()
    
    # 设置滑动窗口参数
    tile = args.tile
    tile_overlap = args.tile_overlap
    sf = 1
    
    # 计算步长
    stride = tile - tile_overlap
    
    # 使用反射填充增加边界区域，避免边缘处理问题
    pad_h = tile_overlap
    pad_w = tile_overlap
    img_tensor_padded = F.pad(img_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    _, _, padded_h, padded_w = img_tensor_padded.size()
    
    # 计算滑动窗口的索引
    h_idx_list = list(range(0, padded_h-tile, stride)) + [padded_h-tile]
    w_idx_list = list(range(0, padded_w-tile, stride)) + [padded_w-tile]
    
    # 存储预测和权重
    E = torch.zeros(1, c, padded_h, padded_w).to(device)
    W = torch.zeros_like(E)
    
    # 使用滑动窗口预测
    with torch.no_grad():
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                # 提取patch
                in_patch = img_tensor_padded[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                
                # 预测
                try:
                    outputs = model(in_patch)
                    
                    # 根据返回类型处理
                    if isinstance(outputs, list):
                        out_patch = outputs[0]
                    elif isinstance(outputs, tuple):
                        if len(outputs) == 4:
                            y_list, _, _, _ = outputs
                            out_patch = y_list[0]
                        else:
                            out_patches, _, _ = outputs
                            out_patch = out_patches[0]
                    else:
                        out_patch = outputs
                        
                except Exception as e:
                    print(f"处理模型输出时出错: {e}")
                    raise
                
                # 确保预测在[0,1]范围内
                out_patch = torch.clamp(out_patch, 0, 1)
                
                # 创建高斯权重mask
                weight = torch.ones_like(out_patch)
                
                center = tile // 2
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(tile, device=device),
                    torch.arange(tile, device=device)
                )
                dist_sq = ((y_grid - center) / center) ** 2 + ((x_grid - center) / center) ** 2
                gaussian = torch.exp(-6 * dist_sq)
                
                # 应用高斯权重到所有通道
                for c_idx in range(weight.shape[1]):
                    weight[:, c_idx, :, :] *= gaussian
                
                # 将预测加入结果
                E[..., h_idx:h_idx+tile, w_idx:w_idx+tile].add_(out_patch * weight)
                W[..., h_idx:h_idx+tile, w_idx:w_idx+tile].add_(weight)
 
    
    # 合并预测结果
    output = E.div_(W)
    output = torch.clamp(output, 0, 1)
    
    # 裁剪回原始大小
    output = output[..., pad_h:pad_h+h, pad_w:pad_w+w]
    
    # 转换为PIL图像
    output_tensor = output.cpu().squeeze(0)
    output_img = transforms.ToPILImage()(output_tensor)
    
    # 确保输出图像与输入图像大小一致
    if output_img.size != original_size:
        output_img = output_img.resize(original_size)
    
    return output_img, img

def create_comparison(input_img, output_img, filename):
    """创建输入和输出对比图"""
    # 确保两张图像大小一致
    width, height = input_img.size
    
    # 创建拼接图像
    comparison = Image.new('RGB', (width*2, height))
    comparison.paste(input_img, (0, 0))
    comparison.paste(output_img, (width, 0))
    
    # 添加描述文本
    comparison_np = np.array(comparison)
    cv2.putText(comparison_np, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(comparison_np, "Output", (width+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    comparison = Image.fromarray(comparison_np)
    comparison.save(os.path.join(args.output_dir, f"comparison_{filename}"))

def main():
    # 加载模型
    model = load_model(args.checkpoint)
    if model is None:
        print("模型加载失败，退出程序。")
        return
    
    # 获取测试图像列表
    image_files = sorted([f for f in os.listdir(args.input_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    print(f"找到 {len(image_files)} 张测试图像")
    
    # 处理每张图像
    total_time = 0
    for idx, file in enumerate(tqdm(image_files, desc="处理图像")):
        img_path = os.path.join(args.input_dir, file)
        
        # 推理计时
        start_time = time.time()
        output_img, input_img = process_image(model, img_path)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        output_filename = f"{idx+1:05d}.png"
        
        # 保存结果
        output_img.save(os.path.join(args.output_dir, output_filename))
        
        if args.save_comparison:
            create_comparison(input_img, output_img, output_filename)
    
    # 输出性能信息
    avg_time = total_time / len(image_files)
    print(f"处理完成！平均每张图像处理时间: {avg_time:.4f} 秒")
    print(f"结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
