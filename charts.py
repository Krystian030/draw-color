import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(genloss_history, latloss_history):
    plt.figure(figsize=(10, 5))

    plt.plot(genloss_history, label='Generation Loss', color='blue')
    plt.plot(latloss_history, label='Latent Loss', color='orange')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.title('Generation and Latent Loss History')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    genloss_history = np.load('genloss_history_svhn_attn.npy')
    latloss_history = np.load('latloss_history_svhn_attn.npy')
    plot_loss_history(genloss_history, latloss_history)