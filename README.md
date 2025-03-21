# EdgeClear-DNSST: Edge-preserving Day-Night Sparse-Sampling Transformer

This project is designed for the **NTIRE 2025 The First Challenge on Day and Night Raindrop Removal for Dual-Focused Images**. We propose **EdgeClear-DNSST**, a deep learning-based raindrop removal network that utilizes edge enhancement and multi-scale attention mechanisms to effectively remove raindrops from images while preserving image quality.

## Project Structure

```plaintxt
├── base_net_snow.py          # Base network model
├── condconv.py               # Conditional Convolution implementation
├── config_raindrop.yaml      # Raindrop removal configuration file
├── dataloader.py             # Data loader
├── EdgeClear_DNSST.py        # Core model implementation
├── inference_raindrop.py     # Raindrop removal inference script
├── matlab_function.py        # MATLAB functions implemented in Python
├── metrics.py                # Evaluation metrics
├── psnr_ssim.py              # PSNR and SSIM calculation
├── train.py                  # Training script
└── loss/                     # Loss functions
    ├── CL1.py                # CL1 loss function
    └── perceptual.py         # Perceptual loss function
```
## Requirements

- Python 3.8
- PyTorch 1.10.0
- torchvision
- numpy
- pillow
- opencv-python
- tqdm
- pyyaml

## Training the Model

### Prepare the Dataset

Organize the dataset with the following structure:
```plaintxt
datasets/
└── RainDrop/
    ├── train/
    │   ├── input/       # Raindrop-corrupted images
    │   └── target/      # Clean ground truth images
    └── test/
        ├── input/       # Test images with raindrops
        └── target/      # Test ground truth images
```
## Configure Training Parameters

Adjust parameters in `config_raindrop.yaml` according to your requirements
## Start Training
Execute the training command:

To start training the model, run the following command in your terminal:

```bash
python train.py --config config_raindrop.yaml
```
Make sure to update the dataset path inside train.py accordingly.

The training process will automatically save the best model as `best_model.ckpt`.
## Inference

Use the trained model to process images containing raindrops. The trained model is available on **Baidu Netdisk**:

**Download link**: [Baidu Netdisk](https://pan.baidu.com/s/1x9QhdWGP4sU9ezdZ9vZhyQ?pwd=6x4i) (Password: 6x4i)

Run the following command:

```bash
python inference_raindrop.py \
  --checkpoint best_model.ckpt \
  --input_dir <input_images_directory> \
  --output_dir results
```
## Arguments

- `--checkpoint`: Path to the trained model checkpoint (`best_model.ckpt`).  
- `--input_dir`: Directory containing input images with raindrops.  
- `--output_dir`: Directory to save the processed images.  

## Features

- **Automatic Edge Enhancement Module**: Preserves sharp edges while removing raindrops.  
- **Configurable Multi-scale Attention Blocks**: Adapts to raindrops of varying sizes.  
- **Day/Night Image Processing**: Handles both daytime and nighttime scenarios effectively.  

## Contact

For technical inquiries or support, please contact:  
[2351666@tongji.edu.cn](mailto:2351666@tongji.edu.cn)
