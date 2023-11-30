This research work is based on the code from https://github.com/yenchenlin/nerf-pytorch and the [Paper](https://arxiv.org/abs/2003.08934)

Installation

```
git clone https://github.com/Joeyu8863/NeRF-security-search.git
cd nerf-pytorch
pip install -r requirements.txt
```

Download data for two example datasets: `lego` and `fern`
```
bash download_example_data.sh
```

Download more data [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Place the downloaded dataset according to the following directory structure:
```
├── configs                                                                                                       
│   ├── ...                                                                                     
│                                                                                               
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern                                                                                                                             
│   │   └── flower  # downloaded llff dataset                                                                                  
│   │   └── horns   # downloaded llff dataset
|   |   └── ...
|   ├── nerf_synthetic
|   |   └── lego
|   |   └── ship    # downloaded synthetic dataset
|   |   └── ...
```
Or using following commend

```
mkdir -p data
cd data
gdown 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
unzip nerf_synthetic.zip 
```

Quick Start
I provide the trained model in origin.tar and quan.tar. If you don't want to train your own model, you can download that and place them in /data/nerf_synthetic/lego, then rename them to 200000.tar. After that you can directly go to last part. 


Train orignal NeRF model

```
python3 run_nerf.py --config configs/lego.txt
```

Before you train the quantized model, you need replace all nn.Linear to bilinear in NeRF class in run_nerf_helpers. Then excute the same command.
After the previous steps complete, run the following part to convert the model to quantized form and excute BFA. rounds meanshow many bits you want to attack in total, rd mean do you want to generate the imageset and video or not.(1 means yes, 0 means no)
All train process takes about 13-15 hours with NVIDIA A100 GPU Server.
```
python3 run_nerf.py --config configs/lego.txt --convert=1 --rounds=10 --rd=1
```
