# Improving Voice Quality in Speech Anonymization With Just Perception-Informed Losses

### 🎉 Accepted at NeurIPS 2024 Audio Imagination Workshop 🎉

### 🧠 Links
- Paper: [arxiv](https://arxiv.org/abs/2410.15499)
- Poster at NeurIPS: [here](https://drive.google.com/file/d/1gv2IgZtfGaEm9E8X1qRiTJfkbakKn0_l/view?usp=share_link)
- 5 min Presentation: t.b.a.
- Audio Samples: [here](https://drive.google.com/drive/folders/16XM85nEJAOKiZ4MJYOWY3r-BRklTwDnz?usp=share_link)

### 🚀 Overview
The increasing use of cloud-based speech assistants has heightened the need for effective speech anonymization, which aims to obscure a speaker's identity while retaining critical information for subsequent tasks. One approach to achieving this is through voice conversion. While existing methods often emphasize complex architectures and training techniques, our research underscores the importance of loss functions inspired by the human auditory system. Our proposed loss functions are model-agnostic, incorporating handcrafted and deep learning-based features to effectively capture quality representations [see figure below]. Through objective and subjective evaluations, we demonstrate that a VQVAE-based model, enhanced with our perception-driven losses, surpasses the vanilla model in terms of naturalness, intelligibility, and prosody while maintaining speaker anonymity. These improvements are consistently observed across various datasets, languages, target speakers, and genders.

<p align="center">
  <img src="documentation/VC-PL-Framework.png" alt="VC-PL-Framework" width="600">
</p>


### 💻 Code
#### Pre-requisites
1. Python 3.7+
2. ```
   pip install -r requirements.txt
   ```
3. Specify your cuda gpu device under `config/global_params` by modifying the entry `CUDA_VISIBLE_DEVICES`.

#### Datasets
We trained the model on a subset of the [VCTK dataset](https://datashare.ed.ac.uk/handle/10283/2651) and tested it on another subset of VCTK, 
additionally on the full 'clean' and 'other' datasets of [LibriSpeech English](https://www.openslr.org/12) 
and [Multilingual LibriSpeech German](https://www.openslr.org/94/). These need to be manually downloaded. For using the same splits, look at the config dir.

#### Perception-Informed Loss Functions
We provide the models of our proposed losses also, located under "src/external/".
You can download the model checkpoints for them [here](https://drive.google.com/drive/folders/1eGpqZUriDj2siNWqkMGFZ9cnh0RbOnjP?usp=share_link).

Note: The formant model was trained on [VTR](http://www.seas.ucla.edu/spapl/VTRFormants.html) dataset 
that uses a subset of [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) dataset.

#### Training
The repo is powered by [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)'s functionality.

Our baseline VQVAE is based on the paper [Non-parallel Voice Conversion based on Hierarchical Latent Embedding Vector Quantized Variational Autoencoder](https://www.isca-speech.org/archive_v0/VCC_BC_2020/abstracts/VCC2020_paper_21.html), by Tuan Vu Ho and Masato Akagi.

Accordingly, to reproduce our results, adjust the respective model, data and trainer config files under
 - `config/model/hle-vqvae-vc.yml`
 - `config/data/vctk20/vctk20-16kHZ.yml`
 - `config/trainer/config-train.yml`

to your needs. Then the model can be trained using the following command:
```
python main.py fit -c config/model/hle-vqvae-vc.yml -c config/data/vctk20/vctk20mel-24kHZ.yml -c config/trainer/config-train.yml
```

#### Inference
You can download the [model](https://drive.google.com/drive/folders/1RQmtmak4KihylkqZ6YaFKwHifUUzZtao?usp=share_link) checkpoint for our VC-PL-VQVAE architecture.
Refer to [notebooks/results.ipynb](notebooks/results.ipynb) for testing.