config = dict(
    type='PointNet2SASSG',
        in_channels=6,  # [xyz, rgb], should be modified with dataset
        num_points=(2048, 1024, 1024, 512),
        radius=(0.1, 0.2, 0.4, 0.8),
        num_samples=(32, 32, 32, 32),
        sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512)),
        fp_channels=(),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)
)