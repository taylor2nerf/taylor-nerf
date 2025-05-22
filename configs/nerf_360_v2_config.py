max_steps = 20000
init_batch_size = 4096
weight_decay = 0

unbounded = True
near_plane = 0.2  
far_plane = 1e3

train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
test_dataset_kwargs = {"factor": 4}

num_samples = 48
num_samples_per_prop = [256, 96]
opaque_bkgd = True
test_chunk_size = 8192