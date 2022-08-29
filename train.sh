export CUDA_VISIBLE_DEVICES=4,5,6,7

# sunrefer2d
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py configs/sunrefer2d.yaml --output_dir outputs/sunrefer2d

# sunrefer2d
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py configs/sunrefer3d.yaml --output_dir outputs/sunrefer3d
python train.py configs/sunrefer3d.yaml --output_dir outputs/sunrefer3d
