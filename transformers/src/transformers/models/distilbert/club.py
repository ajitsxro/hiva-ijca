import torch
import torch.nn as nn

class CLUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        positive = - (mu - y_samples)**2 / (2. * logvar.exp())
        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1) / (2. * logvar.exp())
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def learning_loss(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 / logvar.exp() - logvar).sum(dim=1).mean()
