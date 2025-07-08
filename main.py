import torch

def print_hi(name):
    print(f'Hi, {name}')

def check_device():
    if torch.cuda.is_available():
        print("CUDA è disponibile.")
        print(f"Numero di GPU disponibili: {torch.cuda.device_count()}")
        print(f"Nome della GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU totale: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    else:
        print("CUDA non è disponibile. Verrà usata solo la CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo corrente: {device}")


if __name__ == '__main__':
    print_hi('Masked Autoencoder')
    check_device()


