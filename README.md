# BLH-Seg
Official PyTorch implementation of BLH-Seg: A Semantic Segmentation Framework with Boundary Enhancement and Logit Adaptation under Hyperbolic Auxiliary Supervision.

## Model Checkpoints

The trained segmentation model checkpoints are provided in the `segmentation/Checkpoints/` directory.

## Pretrained Weights

The pretrained weights used for model initialization are the official ImageNet-1K pretrained weights provided by [SegMAN](https://github.com/yunxiangfu2001/SegMAN).

- **Download:** [SegMAN ImageNet-1K Pretrained Weights](https://drive.google.com/drive/folders/1QYU7nhpe0ddH7bPxI7VH4drc__07uEHs?usp=sharing)
- **Usage:** Download the appropriate pretrained weights and place them in the `pretrained/` directory, following the same setup and configuration as the original SegMAN implementation.

Please refer to the [official SegMAN repository](https://github.com/yunxiangfu2001/SegMAN) for details on pretrained weight configuration and loading.

## Training and Testing

The following examples use the ADE20K dataset. Run the commands from the `segmentation/` directory.

### Training

**Single-GPU training:**

```bash
cd segmentation

python tools/train.py \
    local_configs/BLHSeg/tiny/BLHSeg_t_ade.py \
    --work-dir outputs/EXP_NAME
```

**Multi-GPU training:**

```bash
cd segmentation

bash tools/dist_train.sh \
    local_configs/BLHSeg/tiny/BLHSeg_t_ade.py \
    GPU_NUM \
    --work-dir outputs/EXP_NAME
```

Replace `GPU_NUM` with the number of GPUs to use, and `EXP_NAME` with your desired experiment name.

### Testing

**Single-GPU testing:**

```bash
cd segmentation

python tools/test.py \
    local_configs/BLHSeg/tiny/BLHSeg_t_ade.py \
    /path/to/checkpoint_file
```

**Multi-GPU testing:**

```bash
cd segmentation

bash tools/dist_test.sh \
    local_configs/BLHSeg/tiny/BLHSeg_t_ade.py \
    /path/to/checkpoint_file \
    GPU_NUM
```

Replace `/path/to/checkpoint_file` with the path to the trained model checkpoint you want to evaluate, and replace `GPU_NUM` with the number of GPUs to use.
