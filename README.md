# Parameter Space Representation Learning on Mixed-type Data







## Setup

```bash
pip install accelerate matplotlib omegaconf rich neptune
```



## Folder Organization

```tex
datasets/
└── MNIST/ # root folder for MNITS, FashionMNIST, CelebA, CiFAR10, FFHQ128
CacheDisFID/
└── fashion/
    ├── gen/
    └── true/
CacheConFID/
└── Celeba/
    ├── gen/
    └── true/
ParamReL/
├── README.md
├── ConCkpts/
│   ├── Celeba/
├── DisCkpts/
│   ├── mnist/
├── DisCkpts/
│   ├── mnist/
├── Configs/
│   ├── Celeba/
│       ├── celeba.yaml
```



## Train



###  For Discrete Data

```shell
BFN_MNIST.ipynb
```

### For Continous Data

```shell
# train infoBFN based model on CelebA
accelerate launch trainInfoBFN.py config_file=configs/celeba/celeba.yaml
```



## Representation Learning Test
### Sample and CleanFID for Discrete Data

```shell
# sample
BFN_MNIST.ipynb
# exreact data
python Extract_MNIST.py
# FID
## Specify raw data folder and generated data folder
python FID.py
```



### Sample and CleanFID for Continuous Data



```bash
# extract raw data
accelerate launch extract.py config_file=infoBFNconfigs/extract.yaml
# sampling from trained model # must specify the ckpt in gen.py
accelerate launch gen.py config_file=./infoBFNconfigs/celeba_continuous_Nobin.yaml
# cleanfid
python fid.py 
```

### latent quality

```bash
# extract raw data
accelerate launch extract.py config_file=infoBFNconfigs/extract.yaml
# sampling from trained model # must specify the ckpt in gen.py
accelerate launch gen.py config_file=./infoBFNconfigs/celeba_continuous_Nobin.yaml
# cleanfid
python fid.py 
```

### Disentanglement

```bash
python dis.py 
```

### Classification

```bash
python dis.py 
```

### Time-var ying representation learning 
```bash
python time.py 
```

### Latent Traversal

```bash
python inter.py 
```
