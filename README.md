# Perceptual HLE-VQVAE-VC
Based on paper: [Non-parallel Voice Conversion based on Hierarchical Latent Embedding Vector Quantized Variational Autoencoder](https://www.isca-speech.org/archive_v0/VCC_BC_2020/abstracts/VCC2020_paper_21.html), by Tuan Vu Ho and Masato Akagi

## Pre-requisites
1. Python 3.7+
2. ```
   pip install -r requirements.txt
   ```
3. Specify your cuda gpu device under `config/global_params` by modifying the entry `CUDA_VISIBLE_DEVICES`.
   
## Training

The repo is powered by [PyTorch Lighning](https://lightning.ai/docs/pytorch/latest/)'s functionality.

Accordingly, adjust the respective model, data and trainer config files under
 - `config/model/hle-vqvae-vc.yml`
 - `config/data/vctk20/vctk20mel-24kHZ.yml`
 - `config/trainer/config-train.yml`

to your needs. Then the model can be trained using the following command: 
```
python main.py fit -c config/model/hle-vqvae-vc.yml -c config/data/vctk20/vctk20mel-24kHZ.yml -c config/trainer/config-train.yml
```

## Inference

Refer to [notebooks/1.0-results.ipynb](notebooks/1.0-results.ipynb) for details. 
