# TRL4AD
Code for "TRL4AD: Self-Supervised Dual-View Representation Learning for Anomalous Route Detection in Trajectory Databases"
### Preprocessing
- Step1: Download data (<tt>train.csv.zip</tt>) from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data.
- Step2: Put the data file in <tt>../datasets/porto/</tt>, and unzip it as <tt>porto.csv</tt>.
- Step3: Run preprocessing by
```
mkdir -p data/porto
cd preprocess
python preprocess_porto_gps.py
cd ..
```
### Generating ground truth
```
mkdir logs models
python generate_outliers_gps.py --distance 2 --fraction 0.2 --obeserved_ratio 1.0 --dataset porto
```
distance is used to control the moving distance of outliers, fraction is the fraction of continuous outlier
### Training and testing
./run.sh
Example on the Porto dataset:
```
python train.py --task train --dataset porto --mask_token 6069 
python train.py --task test --distance 2 --fraction 0.2 --obeserved_ratio 1.0 --dataset porto
```

