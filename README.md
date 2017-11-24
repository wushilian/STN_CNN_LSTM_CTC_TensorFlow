# STN_CNN_LSTM_CTC_TensorFlow
use STN+CNN+BLSTM+CTC to do OCR
## Attention
It's hard to converge use STN,so you can delete the STN in model ,and it's easy for you.If you delete the STN,the number of val image don't need equal to 256

## How to use this model
1.put your train iamge to 'train' dir,and image name should be like index_label_.jpg(1_abc_.jpg)

2.put your val image to 'val256' dir,and the number of val image should be equal to batch_size(default 256)

3.run train.py

## Use STN in mnist
### left is rotated image，right is transformed image（STN's output）
![image](https://github.com/wushilian/STN_CNN_LSTM_CTC_TensorFlow/blob/master/result/stn.JPG?raw=true)
