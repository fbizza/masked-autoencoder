import os
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from src.model.mae import MaskedAutoencoderClassifier, MaskedAutoencoder
from utils import set_seed, load_config

def get_cifar10_dataloaders(train_subset=None, load_batch_size=64):
    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    train_full = torchvision.datasets.CIFAR10(
        'data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        'data', train=False, download=True, transform=transform
    )

    if train_subset:
        indices = torch.randperm(len(train_full))[:train_subset]
        train_dataset = torch.utils.data.Subset(train_full, indices)
    else:
        train_dataset = train_full

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=load_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=load_batch_size, shuffle=False
    )
    return train_loader, test_loader, test_dataset



if __name__ == '__main__':
    config = load_config(config_name="classifier-without-pretrained-encoder")
    set_seed(config.seed)

    batch_size = config.batch_size
    load_batch_size = min(config.max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0, \
        f"batch_size ({batch_size}) must be divisible by load_batch_size ({load_batch_size})"
    steps_per_update = batch_size // load_batch_size

    train_loader, test_loader, test_dataset = get_cifar10_dataloaders(config.train_subset, load_batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if hasattr(config, 'pretrained_model_path') and config.pretrained_model_path is not None:
        print("Using pretrained ViT encoder")
        model = torch.load(config.pretrained_model_path, map_location='cpu', weights_only=False)
        writer = SummaryWriter(os.path.join('data/logs', 'cifar10', 'mae-pretrained-classifier'))
    else:
        print("Using a not pretrained ViT encoder")
        model = MaskedAutoencoder()
        writer = SummaryWriter(os.path.join('data/logs', 'cifar10', 'not-pretrained-classifier'))

    model = MaskedAutoencoderClassifier(model.encoder, num_classes=10).to(device)

    accuracy_function = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    loss_function = torch.nn.CrossEntropyLoss()

    # as described in the paper
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.base_learning_rate * config.batch_size / 256,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay
    )

    lr_func = lambda epoch: min(
        (epoch + 1) / (config.warmup_epoch + 1e-8),
        0.5 * (math.cos(epoch / config.total_epoch * math.pi) + 1)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    best_test_accuracy = 0
    step_count = 0  # for gradient accumulation
    optim.zero_grad()

    for e in range(config.total_epoch):
        print(f'Epoch {e}:')
        model.train()
        train_losses = []
        train_accs = []

        for img, label in tqdm(train_loader):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_function(logits, label)
            acc = accuracy_function(logits, label)
            loss.backward()

            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()

            train_losses.append(loss.item())
            train_accs.append(acc.item())

        lr_scheduler.step()

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_accuracy = sum(train_accs) / len(train_accs)
        print(f'Train loss: {avg_train_loss:.6f}, Train accuracy: {avg_train_accuracy:.6f}')

        model.eval()
        test_losses = []
        test_accuracies = []
        with torch.no_grad():
            for img, label in tqdm(test_loader):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_function(logits, label)
                acc = accuracy_function(logits, label)
                test_losses.append(loss.item())
                test_accuracies.append(acc.item())

        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
        print(f'Test loss: {avg_test_loss:.6f}, Test accuracy: {avg_test_accuracy:.6f}')

        writer.add_scalars('cls/loss', {'train': avg_train_loss, 'test': avg_test_loss}, global_step=e)
        writer.add_scalars('cls/accuracy', {'train': avg_train_accuracy, 'test': avg_test_accuracy}, global_step=e)

        if avg_test_accuracy > best_test_accuracy:
            best_test_accuracy = avg_test_accuracy
            print(f'Saving best model with accuracy {best_test_accuracy:.6f} at epoch {e}')
            os.makedirs(config.model_path, exist_ok=True)
            torch.save(model, f"{config.model_path}/{config.config_name}_epoch_{e}.pt")
