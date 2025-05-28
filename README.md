## HGFormer

Despite considerable progress in colon polyp segmentation,current models remain inadequate in differentiating polyp boundaries from high-frequency noise components,resulting in compromised boundary detection accuracy. To address this challenge, we propose a High-Frequency Guided Transformer (HGFormer) for accurate polyp segmentation, which contains two key innovations: high-frequency boundary mining (HBM) and high-frequency boundary-guided fusion (HBF) modules. The HBM employs bidirectional gradient magnitude computation and dynamic modulation to selectively enhance diagnostically relevant boundary features while suppressing noise. The HBF leverages high-frequency edge features to guide multi-scale feature fusion, reducing false positive and false negative predictions. Extensive experiments across five benchmark datasets validate HGFormer’s state-of-the-art (SOTA) performance. HGFormer achieves leading segmentation accuracy on Kvasir-
SEG and ClinicDB, while demonstrating notable improvements on the more challenging ETIS dataset, with a 1.0% increase in mDice and a 1.6% gain in mIoU over existing methods. 


### Proposed Baseline
#### Training/Testing
The training and testing experiments are conducted using PyTorch with a single NVIDIA 3090 with 24 GB Memory.<br>
>Note that our model also supports low memory GPU, which means you can lower the batch size.<br>
  * downloading testing dataset and move it into `./data/TestDataset/`, which can be found in this [download link (327.2MB)](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view). It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).<br>
  * downloading training dataset and move it into `./data/TrainDataset/`, which can be found in this [download link (399.5MB)](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view). It contains two sub-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples).
  * downloading pretrained weights and move it into ./checkpoint/, which can be found in this [download link(134.9MB)](https://pan.baidu.com/s/13Pv8xEGNAM3KxJgT-mLwZw?pwd=u5rd).<br>
  * downloading PVT-V2 weights and and move it into ./lib/, which can be found in this [download link(96.8MB)](https://pan.baidu.com/s/12CoPRhzwKOfnjfg6saEZ0w?pwd=7i9u).
#### Training Configuration:
Assigning your costumed path, like  `--save_model` and `--train_path` in `train.py`.<br>
After you download all the pre-trained model and testing dataset, just run `test.py` to generate the final prediction map: replace your trained model directory (`--pth_path`).
### Evaluating your trained model
Matlab: One-key evaluation is written in MATLAB code ([link](https://drive.google.com/file/d/1_h4_CjD5GKEf7B1MRuzye97H0MXf2GE9/view)), please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results.

### Pre-computed maps
The results of FreqNet can be found in download [link](https://pan.baidu.com/s/1NfemtSo1-MlleH7mGsq4Mw?pwd=haii).
