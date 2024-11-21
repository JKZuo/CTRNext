# CTRNext
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)
![Code](https://img.shields.io/badge/Code-python-purple)

Collaborative Trajectory Representation for Enhanced Next POI Recommendation (CTRNext)

- Motivation: Although existing studies have taken factors such as spatial, social relations, and temporal elements influencing the effectiveness of POI recommendations into account, their focus tends to lean toward fitting individual users’ historical check-in behaviors. In fact, predicting the next visit of a specific user not only relies on that user’s individual historical check-in POI (i.e., POI-POI Interaction) but also correlates with the trajectories of other users who share similar habitual preferences (i.e., User–User Interaction), as shown in **Fig. 1**.
<p align="center">
<img align="middle" src="https://github.com/JKZuo/CTRNext/blob/main/Figures/fig0.png" width="600"/>
</p>
<p align = "center">
<b>Figure 1. An illustration of the similar-minded users. </b> 
</p>

The overall framework of our proposed CTRNext model is illustrated in **Fig. 2**, comprising primarily of two embedding layers that learn the dense representations from user historical check-in trajectories, and three learning modules that refine and update these representations to generate recommendations tailored to user preferences.
<p align="center">
<img align="middle" src="https://github.com/JKZuo/CTRNext/blob/main/Figures/fig1.png" width="800"/>
</p>
<p align = "center">
<b>Figure 2. The framework of the proposed Collaborative Trajectory Representation (CTRNext) model. </b> 
</p>

## Requirements
The code has been tested running under Python 3.9.12.

The required packages are as follows: 
- Python == 3.9.12 
- PyTorch == 1.11.0
- NumPy == 1.18.2

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
  
epoch:1, time:86, valid_acc:[0.35 0.48 0.57 0.62],MRR:0.426, mAP20:0.419, NDCG5:0.424, NDCG10:0.452, NDCG20:0.465

epoch:1, time:86, test_acc:[0.30 0.43 0.51 0.59], MRR:0.367, mAP20:0.361, NDCG5:0.366, NDCG10:0.392, NDCG20:0.412

We ultimately chose the results from the testing set corresponding to the best performance on the validation set as the output.


## Data
We conduct our experiments on four widely used cities for the POI location recommendation task, namely New York City (NYC), Tokyo (TKY), Singapore (SIN), California and Nevada (CA), all collected from
the two most popular location-based social networks (LBSNs) service providers, i.e., Foursquare and Gowalla.  The basic statistical information of the four datasets is shown in **Table 2**.
<p align="center">
<img align="middle" src="https://github.com/JKZuo/CTRNext/blob/main/Figures/table2.png"/>
</p>
<p align = "center">
<b>Table 2. Basic dataset statistics of four datasets. </b> 
</p>

Due to the large dataset, you can download it through this link:

https://pan.baidu.com/s/19NG8Vn3u4fhsUK1P_kEr0Q?pwd=poi1

Each record consists of the user ID, POI ID, POI’s GPS-specific coordinates, POI category, and check-in time. For some locations without POI categories, we have supplemented them using the OpenStreetMap-based Nominatim tool. 
The top ten most check-in location categories among the four cities are shown in **Fig. 3**.
<p align="center">
<img align="middle" src="https://github.com/JKZuo/CTRNext/blob/main/Figures/fig2.png" width="600"/>
</p>
<p align = "center">
<b>Figure 3. The most frequently visited POI categories. </b> 
</p>

## Result
**Fig. 4** and **Fig. 5** depicts the final results.
<p align="center">
<img align="middle" src="https://github.com/JKZuo/CTRNext/blob/main/Figures/fig3.png"/>
</p>
<p align = "center">
<b>Figure 4. Performance of CTRNext under different values of the embedding size and time decay rate. </b> 
</p>

<p align="center">
<img align="middle" src="https://github.com/JKZuo/CTRNext/blob/main/Figures/fig4.png"/>
</p>
<p align = "center">
<b>Figure 5. Performance of CTRNext under different values of the hidden state size and attention channels. </b> 
</p>


## Cite
If you feel that this work has been helpful for your research, please cite it as: 

```tex
@ARTICLE{CTRNext,
author = {Jiankai Zuo and Yaying Zhang},
title = {Collaborative trajectory representation for enhanced next POI recommendation},
journal = {Expert Systems with Applications},
volume = {256},
pages = {124884},
year = {2024},
ISSN = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.124884},
URL = {https://www.sciencedirect.com/science/article/pii/S0957417424017512},
keywords = {Next POI recommendation, Trajectory similarity, Attention mechanism, Representation learning},
}
```

- This code is a simplified version of the CTRNext model, and some of the core code will also be needed for my future research. If you want to know more, please send me a private message.
