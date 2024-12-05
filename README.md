# HMDP

Hi, this repository contains the code and the data for the paper "Tackling Real-world Complexity: Hierarchical Modeling and Dynamic Prompting for Multimodal Long Document Classification"

To download the dataset, pls go to this URL: [https://drive.google.com/open?id=1qGmyEVD19ruvLLz9J0QGV7rsZPFEz2Az](https://drive.google.com/drive/folders/1759nBGt7J0ZkUK8-jSbAys9_SANQVC-E?dmr=1&ec=wgc-drive-hero-goto)

Please organize the dataset structure as follows:
```
dataset
├── MMaterials
├── MAAPD
├── Review
├── Food101
```

Any questions can be sent to tfliu@emails.bjut.edu.cn

Requirements:

    Pytorch
    Transformer (pytorch): https://github.com/huggingface/transformers
    GPU

##Training
```
python train_aapd.py --max_seq_length 256 
```
