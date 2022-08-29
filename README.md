# TransVG
<!-- <p align="center"> <img src='docs/framework.jpg' align="center" height="540px"> </p>

This is the official implementation of [**TransVG: End-to-End Visual Grounding with Transformers**](https://arxiv.org/abs/2104.08541). This paper has been accepted by ICCV 2021.

    @article{deng2021transvg,
      title={TransVG: End-to-End Visual Grounding with Transformers},
      author={Deng, Jiajun and Yang, Zhengyuan and Chen, Tianlang and Zhou, Wengang and Li, Houqiang},
      journal={arXiv preprint arXiv:2104.08541},
      year={2021}
} -->

### Installation
1.  Clone this repository.
    ```
    git clone 
    ```

2.  Prerequisites. 
    * Python 3.8.5
    * pytorch 1.11.0 and torchvision 0.12.0
    * mmdetection3D 1.0.0rc4
    * Others ([Pytorch-Bert](https://pypi.org/project/pytorch-pretrained-bert/), etc.) Check requirements.txt for reference.

<!-- ### Getting Started

Please refer to [GETTING_STARGTED.md](docs/GETTING_STARTED.md) to learn how to prepare the datasets and pretrained checkpoints. -->

<!-- ### Model Zoo -->

<!-- The models with ResNet-50 backbone and ResNet-101 backbone are available in [[Gdrive]](https://drive.google.com/drive/folders/17CVnc5XOyqqDlg1veXRE9hY9r123Nvqx?usp=sharing) -->



### Training and Evaluation

<!-- 1.  Training
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50 --epochs 90 --lr_drop 60
    ```

    We recommend to set --max_query_len 40 for RefCOCOg, and --max_query_len 20 for other datasets. 
    
    We recommend to set --epochs 180 (--lr_drop 120 acoordingly) for RefCOCO+, and --epochs 90 (--lr_drop 60 acoordingly) for other datasets. 

2.  Evaluation
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset referit --max_query_len 20 --eval_set test --eval_model ./outputs/referit_r50/best_checkpoint.pth --output_dir ./outputs/referit_r50
    ``` -->

<!-- ### Acknowledge
This codebase is partially based on [ReSC](https://github.com/zyang-ur/ReSC) and [DETR](https://github.com/facebookresearch/detr). -->
