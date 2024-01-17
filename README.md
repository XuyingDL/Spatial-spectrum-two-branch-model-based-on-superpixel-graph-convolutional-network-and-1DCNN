# System-specific notes
Pytorch1.9 version, python3.7
# How to use it?
## Data preparation
Firstly, you should prapare the data and label as ".mat" format for the whole study area and put them in the file "*/data/name"   
Then you should prapare the rgb image of the study area for superpixel segmentation and put it in the file "*/data/rgb".
## Code 
Firstly, run trainTestSplit.py to split train and verify data  
Then run train_hubei.py for model training.  
Finally run predict.py for obtaining the prediction.
# REFERENCE
[Jia S., Jiang S., Zhang S., Xu M., Jia X., 2024. Graph-in-Graph Convolutional Network for Hyperspectral Image Classification. IEEE Transactions on Neural Networks and Learning Systems, 35(1): 1157-1171.](https://ieeexplore.ieee.org/abstract/document/9801664)  
If you have any question, you can contact me via email xvying1021@gmail.com
