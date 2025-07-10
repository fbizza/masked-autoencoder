import torch
import torchvision
import math
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def denormalize(tensor):
    return tensor * 0.5 + 0.5


def get_val_dataset():
    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    return torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)


def load_model(model_path, device='cuda'):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def predict_on_batch(model, dataset, start_index, batch_size=10, device='cuda'):
    dataset_size = len(dataset)
    end_index = min(start_index + batch_size, dataset_size)

    imgs = torch.stack([dataset[i][0] for i in range(start_index, end_index)]).to(device)
    labels = torch.tensor([dataset[i][1] for i in range(start_index, end_index)]).to(device)

    with torch.no_grad():
        logits = model(imgs)
        preds = logits.argmax(dim=1)

    show_predictions(imgs.cpu(), labels.cpu(), preds.cpu(), dataset.classes)


def show_predictions(images, labels, preds, class_names):
    images = denormalize(images)
    images = images.permute(0, 2, 3, 1)

    num_images = len(images)
    cols = min(num_images, 5)  # max 5 for each row
    rows = math.ceil(num_images / cols)
    figsize = (cols * 3, rows * 3)

    plt.figure(figsize=figsize)
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].numpy().clip(0, 1))
        plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # you can change the following 3 parameters:
    start_index = 4000
    num_images = 8
    model_path = "src/data/weights/classifier-with-pretrained-encoder.pt"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(model_path, device=device)
    val_dataset = get_val_dataset()

    predict_on_batch(model, val_dataset, start_index, batch_size=num_images, device=device)
