# Improving Voice Quality in Speech Anonymization With Just Perception-Informed Losses

### 🎉 Accepted at NeurIPS 2024 Audio Imagination Workshop 🎉

### 🧠 Links
- Paper: https://arxiv.org/abs/2410.15499
- Poster at NeurIPS: https://drive.google.com/file/d/1gv2IgZtfGaEm9E8X1qRiTJfkbakKn0_l/view?usp=share_link
- 5 min Presentation: t.b.a.
- Audio Samples: t.b.a.

### 🚀 Overview
The increasing use of cloud-based speech assistants has heightened the need for effective speech anonymization, which aims to obscure a speaker's identity while retaining critical information for subsequent tasks. One approach to achieving this is through voice conversion. While existing methods often emphasize complex architectures and training techniques, our research underscores the importance of loss functions inspired by the human auditory system. Our proposed loss functions are model-agnostic, incorporating handcrafted and deep learning-based features to effectively capture quality representations [see figure below]. Through objective and subjective evaluations, we demonstrate that a VQVAE-based model, enhanced with our perception-driven losses, surpasses the vanilla model in terms of naturalness, intelligibility, and prosody while maintaining speaker anonymity. These improvements are consistently observed across various datasets, languages, target speakers, and genders.

<p align="center">
  <img src="documentation/VC-PL-Framework.png" alt="VC-PL-Framework" width="600">
</p>

### 💻 Code
To replicate our results, follow these steps: t.b.a.
