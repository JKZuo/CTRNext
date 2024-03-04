## CTRNext
A Collaborative Trajectory Representation model (CTRNext) to enhance the Next POI Recommendation.

## Requirements
The code has been tested running under Python 3.9.12.

The required packages are as follows: 

Python == 3.9.12 

PyTorch == 1.11.0

NumPy == 1.18.2

## Running
```shell
nohup python train.py > train.log 2>&1 &
```

You will see on the screen the result: 

  0%|          | 0/100 [00:00<?, ?it/s] 
  
  1%|          | 1/100 [00:01<01:59,  1.21s/it]
  
  2%|▏         | 2/100 [00:02<01:37,  1.01it/s]
  
  3%|▎         | 3/100 [00:02<01:32,  1.05it/s]
  
  4%|▍         | 4/100 [00:03<01:28,  1.08it/s]
  
  5%|▌         | 5/100 [00:04<01:26,  1.10it/s]
  
  ...
  
  100%|██████████| 100/100 [01:26<00:00,  1.16it/s]
  
epoch:1, time:86, valid_acc:[0.35 0.48 0.57 0.62], MRR:0.426, mAP20:0.419, NDCG5:0.424, NDCG10:0.452, NDCG20:0.465

epoch:1, time:86, test_acc:[0.30 0.43 0.51 0.59], MRR:0.367, mAP20:0.361, NDCG5:0.366, NDCG10:0.392, NDCG20:0.412
  
