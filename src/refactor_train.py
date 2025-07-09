import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from einops import rearrange

from src.model.mae import MaskedAutoencoder
from utils import set_seed

def get_cifar10_dataloaders(train_subset=None, load_batch_size=64):
    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    train_full = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    val = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)

    if train_subset:
        indices = torch.randperm(len(train_full))[:train_subset]
        train = torch.utils.data.Subset(train_full, indices)
    else:
        train = train_full

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=load_batch_size, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=load_batch_size, shuffle=False, num_workers=2,
        pin_memory=True, drop_last=False
    )
    return train_loader, val_loader, val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--max_device_batch_size', type=int, default=64)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=200)
    parser.add_argument('--warmup_epoch', type=int, default=10)
    parser.add_argument('--model_path', type=str, default='data/weights/vit-t-mae.pt')
    parser.add_argument('--train_subset', type=int, default=None,
                        help='Numero di immagini da usare per il training (default: tutte)')

    args = parser.parse_args()
    set_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    dataloader, val_loader, val_dataset = get_cifar10_dataloaders(args.train_subset, load_batch_size)

    writer = SummaryWriter(os.path.join('data/logs', 'cifar10', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = MaskedAutoencoder(mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(),
                              lr=args.base_learning_rate * args.batch_size / 256,
                              betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)

    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        print(f'Epoch {e}:')
        model.train()
        losses = []
        for img, _ in tqdm(dataloader):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        print(f'Train loss on masked pixels: {avg_loss:.6f}')

        ''' Visualizzazione immagini '''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)

        ''' Calcolo della loss di validazione '''
        val_losses = []
        with torch.no_grad():
            for val_img, _ in val_loader:
                val_img = val_img.to(device)
                predicted_val_img, mask = model(val_img)
                loss = torch.mean((predicted_val_img - val_img) ** 2 * mask) / args.mask_ratio
                val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)

        writer.add_scalars('loss', {
            'train': avg_loss,
            'test': avg_val_loss
        }, global_step=e)
        print(f'Val loss on masked pixels: {avg_val_loss:.6f}')

        ''' Salvataggio del modello '''
        torch.save(model, args.model_path)
