import os
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from einops import rearrange
from src.model.mae import MaskedAutoencoder
from utils import set_seed, load_config


def get_cifar10_dataloaders(train_subset=None, load_batch_size=64):
    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    train_full = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)

    if train_subset:
        indices = torch.randperm(len(train_full))[:train_subset]
        train = torch.utils.data.Subset(train_full, indices)
    else:
        train = train_full

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=load_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=load_batch_size, shuffle=False
    )
    return train_loader, test_loader, test



if __name__ == '__main__':
    config = load_config(config_name="mae-self-supervised-training-25-masking")  #TODO: change this
    set_seed(config.seed)
    batch_size = config.batch_size
    gpu_load_batch_size = min(config.max_device_batch_size, batch_size)
    assert batch_size % gpu_load_batch_size == 0, \
        f"batch_size ({batch_size}) must be divisible by gpu_load_batch_size ({gpu_load_batch_size})"
    steps_per_update = batch_size // gpu_load_batch_size
    dataloader, test_loader, test_dataset = get_cifar10_dataloaders(config.train_subset, gpu_load_batch_size)
    writer = SummaryWriter(os.path.join('data/logs', 'cifar10', f'{config.config_name}'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = MaskedAutoencoder(mask_ratio=config.mask_ratio).to(device)

    optim = torch.optim.AdamW(model.parameters(),
                              lr=config.base_learning_rate * config.batch_size / 256,
                              betas=(0.9, 0.95),
                              weight_decay=config.weight_decay)

    lr_func = lambda epoch: min(
                                (epoch + 1) / (config.warmup_epoch + 1e-8),  # warm up
                                0.5 * (math.cos(epoch / config.total_epoch * math.pi) + 1)  # cosine Decay
                                )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    step_count = 0  # for gradient accumulation
    optim.zero_grad()
    for e in range(config.total_epoch):
        print(f'Epoch {e}:')
        model.train()
        train_losses = []
        for img, _ in tqdm(dataloader):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / config.mask_ratio  # NOTE: no loss is computed on visible patches!! also note that loss is computed on normalized pixels
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            train_losses.append(loss.item())
        lr_scheduler.step()
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f'Train loss on masked pixels: {avg_train_loss:.6f}')

        # to visualize improvement on images reconstruction quality for each epoch
        model.eval()
        with torch.no_grad():
            test_img = torch.stack([test_dataset[i][0] for i in range(16)])
            test_img = test_img.to(device)
            predicted_test_img, mask = model(test_img)
            predicted_test_img = predicted_test_img * mask + test_img * (1 - mask)
            img = torch.cat([test_img * (1 - mask), predicted_test_img, test_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)

        # test loss
        test_losses = []
        with torch.no_grad():
            for test_img, _ in test_loader:
                test_img = test_img.to(device)
                predicted_test_img, mask = model(test_img)
                loss = torch.mean((predicted_test_img - test_img) ** 2 * mask) / config.mask_ratio
                test_losses.append(loss.item())
        avg_test_loss = sum(test_losses) / len(test_losses)

        writer.add_scalars('loss', {
            'train': avg_train_loss,
            'test': avg_test_loss
        }, global_step=e)
        print(f'Test loss on masked pixels: {avg_test_loss:.6f}')

        # save model every 10 epochs
        if e % 10 == 0:
            os.makedirs(config.model_path, exist_ok=True)
            torch.save(model, f"{config.model_path}/{config.config_name}_epoch_{e}.pt")

        # save checkpoint every 25 epochs
        if e % 25 == 0:
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }
            os.makedirs(config.model_path, exist_ok=True)
            torch.save(checkpoint, f"{config.model_path}/{config.config_name}_checkpoint_epoch_{e}.pt")



