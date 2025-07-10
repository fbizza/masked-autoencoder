import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from src.model.mae import MaskedAutoencoder

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def denormalize(tensor):
    return tensor * 0.5 + 0.5


def get_val_dataset():
    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    return torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)


def load_model(model_path, mask_ratio=0.75, device='cuda'):
    model = MaskedAutoencoder(mask_ratio=mask_ratio)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def inference_on_batch(model, dataset, start_index, batch_size=16, device='cuda'):
    dataset_size = len(dataset)
    end_index = min(start_index + batch_size, dataset_size)
    actual_batch_size = end_index - start_index

    imgs = torch.stack([dataset[i][0] for i in range(start_index, end_index)]).to(device)  # [B, 3, 32, 32]

    with torch.no_grad():
        predicted_imgs, masks = model(imgs)
        reconstructed_imgs = predicted_imgs * masks + imgs * (1 - masks)

    originals = denormalize(imgs.cpu())
    masked_inputs = denormalize((imgs * (1 - masks)).cpu())
    reconstructions = denormalize(reconstructed_imgs.cpu())

    fig, axs = plt.subplots(3, actual_batch_size, figsize=(actual_batch_size * 2, 6))

    if actual_batch_size == 1:
        axs = axs.reshape(3, 1)

    for i in range(actual_batch_size):
        axs[0, i].imshow(originals[i].permute(1, 2, 0).clip(0, 1))
        axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_title('Original')

        axs[1, i].imshow(masked_inputs[i].permute(1, 2, 0).clip(0, 1))
        axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_title('Masked Input')

        axs[2, i].imshow(reconstructions[i].permute(1, 2, 0).clip(0, 1))
        axs[2, i].axis('off')
        if i == 0:
            axs[2, i].set_title('Reconstructed')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    start_index = 4000
    num_images = 6

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "src/data/weights/mae-self-supervised-training_epoch_160.pt"

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    val_dataset = get_val_dataset()

    inference_on_batch(model, val_dataset, start_index, batch_size=num_images, device=device)
