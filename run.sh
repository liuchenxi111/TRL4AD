#!/bin/bash


echo "Running pretraining and training phase..."
python train.py --pretrain_epochs 5 --epochs 5 --task train 

echo "Running: distance=2, fraction=0.3, observed_ratio=1.0"
python train.py --distance 2 --fraction 0.3 --obeserved_ratio 1.0 --task test

echo "Running: distance=2, fraction=0.3, observed_ratio=0.7"
python train.py --distance 2 --fraction 0.3 --obeserved_ratio 0.7 --task test

echo "Running: distance=2, fraction=0.3, observed_ratio=0.5"
python train.py --distance 2 --fraction 0.3 --obeserved_ratio 0.5 --task test

echo "Running: distance=2, fraction=0.2, observed_ratio=1.0"
python train.py --distance 2 --fraction 0.2 --obeserved_ratio 1.0 --task test

echo "Running: distance=2, fraction=0.2, observed_ratio=0.7"
python train.py --distance 2 --fraction 0.2 --obeserved_ratio 0.7 --task test

echo "Running: distance=2, fraction=0.2, observed_ratio=0.5"
python train.py --distance 2 --fraction 0.2 --obeserved_ratio 0.5 --task test

echo "Running: distance=1, fraction=0.3, observed_ratio=1.0"
python train.py --distance 3 --fraction 0.1 --obeserved_ratio 1.0 --task test

echo "Running: distance=1, fraction=0.3, observed_ratio=0.7"
python train.py --distance 3 --fraction 0.1 --obeserved_ratio 0.7 --task test

echo "Running: distance=1, fraction=0.3, observed_ratio=0.5"
python train.py --distance 3 --fraction 0.1 --obeserved_ratio 0.5 --task test

echo "All jobs finished."

# CD
echo "Running pretraining and training phase..."
python train.py --pretrain_epochs 5 --epochs 5 --task train --dataset cd --mask_token 25718 

echo "Running: distance=2, fraction=0.3, observed_ratio=1.0"
python train.py --distance 2 --fraction 0.3 --obeserved_ratio 1.0 --task test  --dataset cd --mask_token 25718 

echo "Running: distance=2, fraction=0.3, observed_ratio=0.7"
python train.py --distance 2 --fraction 0.3 --obeserved_ratio 0.7 --task test --dataset cd --mask_token 25718 

echo "Running: distance=2, fraction=0.3, observed_ratio=0.5"
python train.py --distance 2 --fraction 0.3 --obeserved_ratio 0.5 --task test --dataset cd --mask_token 25718 

echo "Running: distance=2, fraction=0.2, observed_ratio=1.0"
python train.py --distance 2 --fraction 0.2 --obeserved_ratio 1.0 --task test --dataset cd --mask_token 25718 

echo "Running: distance=2, fraction=0.2, observed_ratio=0.7"
python train.py --distance 2 --fraction 0.2 --obeserved_ratio 0.7 --task test --dataset cd --mask_token 25718 

echo "Running: distance=2, fraction=0.2, observed_ratio=0.5"
python train.py --distance 2 --fraction 0.2 --obeserved_ratio 0.5 --task test --dataset cd --mask_token 25718 

echo "Running: distance=1, fraction=0.3, observed_ratio=1.0"
python train.py --distance 3 --fraction 0.1 --obeserved_ratio 1.0 --task test --dataset cd --mask_token 25718 

echo "Running: distance=1, fraction=0.3, observed_ratio=0.7"
python train.py --distance 3 --fraction 0.1 --obeserved_ratio 0.7 --task test --dataset cd --mask_token 25718 

echo "Running: distance=1, fraction=0.3, observed_ratio=0.5"
python train.py --distance 3 --fraction 0.1 --obeserved_ratio 0.5 --task test --dataset cd --mask_token 25718 

echo "All jobs finished."