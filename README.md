# STN_CNN_CTC_TensorFlow
use STN+CNN+BLSTM+CTC to do OCR
## How to use this model
1.put your train iamge to 'train' dir,and image name should be like index_label_.jpg(1_abc_.jpg)

2.put your val image to 'val256' dir,and the number of val image should be equal to batch_size(default 256)

3.run train.py

## this work based on @ilovin lstm_ctc_ocr project https://github.com/ilovin
