# Diagnosing pathologic myopia by identifying morphologic patterns using ultra widefield images with deep learning

![overview](assets/overview.png)

## News

[2025/7] The training and inference code is released.

[2025/7] The paper is published in [npj Digital Medicine](10.1038/s41746-025-01849-y) (IF=15.1).

## Environment

Create the environment with conda:

```shell
cd WORK_DIR/
conda create -n RealMNet python=3.10 -y
conda activate RealMNet
```

Install PyTorch with the tested version 1.13.0:

```shell
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Build customized mmengine with the tested version 0.8.4:

```shell
mmenginecd mmengine
pip install -e . -v
```

Build mmcv with the tested version 2.0.1:

```shell
cd mmcv
pip install -r requirements/optional.txt
nvcc --version
gcc --version
pip install -e . -v
```

Build customized mmpretrain with the tested version 1.0.2:

```shell
cd mmpretrain
pip install -r requirements.txt
pip install -v -e .
```

Verify the installation:

```shell
# verify the installation of PyTorch
python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'
python -c 'import torch;print(torch.cuda.is_available())'
# verify the installation of mmengine
python -c 'from mmengine.utils.dl_utils import collect_env;print(collect_env())'
python -c 'import mmengine;print(mmengine.__version__)'
# verify the installation of mmcv
python .dev_scripts/check_installation.py
```

## Training

```shell
tools/dist_train.sh configs/myopia_models/tinyvit-21m-distill-384_MYOPIA.py 4
```

## Inference

```shell
export run_dir=work_dirs/tinyvit-21m-distill-384_MYOPIA; \
export run_time=RUN_TIME_PLACEHOLDER; \
export epoch=50; \
export out_item=metrics; \
tools/dist_test.sh \
$run_dir/$run_time/vis_data/config.py \
$run_dir/$run_time/epoch_$epoch.pth \
4 \
--out $run_dir/$run_time/vis_data/test.json \
--out-item $out_item
```

## Citation

If you find this repository useful, please consider citing this paper:

```
TBD
```

