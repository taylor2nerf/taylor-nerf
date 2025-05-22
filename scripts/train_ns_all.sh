DATA_ROOT="data/nerf_synthetic"

for scene in `ls $DATA_ROOT` 
do
    python runner/train_taylor_occ.py --scene $scene --data_root $DATA_ROOT
done