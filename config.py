cifar10_cfg = {
    "resolution": 32,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": (1,2,2,2),
    "num_res_blocks": 2,
    "attn_resolutions": (16,),
    "dropout": 0.1,
}

lsun_cfg = {
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": (1,1,2,2,4,4),
    "num_res_blocks": 2,
    "attn_resolutions": (16,),
    "dropout": 0.0,
}

celeba64_cfg = {
    "resolution": 64,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": (1,2,2,2,4),
    "num_res_blocks": 2,
    "attn_resolutions": (16,),
    "dropout": 0.1,
}

model_config_map = {
    "cifar10": cifar10_cfg,
    "lsun_bedroom": lsun_cfg,
    "lsun_cat": lsun_cfg,
    "lsun_church": lsun_cfg,
    "celeba64": celeba64_cfg
}

diffusion_config = {
    "beta_0": 0.0001,
    "beta_T": 0.02,
    "T": 1000,
}

model_var_type_map = {
    "cifar10": "fixedlarge",
    "lsun_bedroom": "fixedsmall",
    "lsun_cat": "fixedsmall",
    "lsun_church": "fixedsmall",
}
