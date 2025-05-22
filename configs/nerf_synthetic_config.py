max_steps = 20000

init_batch_size = 1024
target_sample_batch_size = 1 << 18

aabb = [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]
near_plane = 2
far_plane = 1.0e10
train_dataset_kwargs = {}
test_dataset_kwargs = {}
grid_resolution = 128
grid_nlvl = 1
render_step_size = 5e-3
alpha_thre = 0.0
cone_angle = 0.0
test_chunk_size = 16384