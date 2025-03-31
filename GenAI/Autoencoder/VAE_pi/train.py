from dataset import *
from model import *
import torch
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import os
import argparse

def train_model(args):
    dataset = PI_dataset(args.xs_path, args.ys_path, args.img_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device="cuda"

    model = VAE_PI(5, args.latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def VAE_loss(input, reconstruct, mu, log_var, k_weight=0.0025):
        recons_loss = F.mse_loss(reconstruct, input)
        kl_loss = torch.mean(-0.5 * torch.sum(1+log_var-mu**2-log_var.exp() ,dim = 1), dim = 0)
        loss = recons_loss + k_weight * kl_loss
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KL_loss': kl_loss.detach()}

    mu_history = []
    log_var_history = []
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        for i, d in enumerate(pbar):
            d = d.to(device)
            [reconstruct, input, mu, log_var] = model(d)
            loss = VAE_loss(input, reconstruct, mu, log_var)
            opt.zero_grad()
            loss['loss'].backward()
            opt.step()
            pbar.set_postfix(MSE=loss['loss'].item())
        avg_mu = torch.mean(torch.sum(mu, dim=1), dim=0).item()
        avg_log_var = torch.mean(torch.sum(log_var, dim=1), dim=0).item()
        print("Mu is: ", avg_mu)
        print("Log_var is: ", avg_log_var)
        mu_history.append(avg_mu)
        log_var_history.append(avg_log_var)
        torch.save(model.state_dict(), os.path.join("ckpts", "pi.pt"))

    epochs = range(len(mu_history))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mu_history, label='mu', marker='o', linestyle='-')
    plt.plot(epochs, log_var_history, label='log_var', marker='s', linestyle='--')

    # Add labels and legend
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title('Mu and Log-Var History')
    plt.legend()
    plt.grid(True)

    plt.savefig('img/mu_log_var_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, default=30)
    parse.add_argument('--latent_dim', type=int, default=2)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--device', type=str, default="cuda")
    parse.add_argument('--lr', type=float, default=0.005)
    parse.add_argument('--weight_decay', type=float, default=0.0)
    parse.add_argument('--xs_path', type=str, default="pi_xs.npy")
    parse.add_argument('--ys_path', type=str, default="pi_ys.npy")
    parse.add_argument('--img_path', type=str, default="sparse_pi_colored.jpg")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)