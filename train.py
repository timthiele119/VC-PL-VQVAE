import os
import torch
from torch.optim import Adam
from src.data.datasets import VCTKDataset
from src.models.vqvae_vc import VQVAEVC
from src.training.vqvae_vc_trainers import VQVAEVCTrainer
from src.losses.vqvae_losses import VQVAELoss
from src.params import Params

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":

    DEVICE = device if torch.cuda.is_available() else "cpu"
    print(f"Use {DEVICE} device")

    dataset_params = Params("config/dataset/VCTK-22.05kHZ.json")
    train_dataset = VCTKDataset(dataset_params, train=True)
    test_dataset = VCTKDataset(dataset_params, train=False)

    model_params = Params("config/model/group-vqvae-vc.json")
    model = VQVAEVC(model_params)

    print(model.parameters())

    optimizer = Adam(params=model.parameters())

    trainer = VQVAEVCTrainer(model=model, train_dataset=train_dataset, test_dataset=test_dataset, batch_size=64,
                             loss_fn=VQVAELoss(), optimizer=optimizer, epochs=10, device=device)
    trainer.train()


