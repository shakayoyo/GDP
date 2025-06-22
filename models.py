import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        res = self.act(self.norm(self.linear(x)))
        return x + res

class CombinedDiffusionNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim: int,
        time_embed_dim: int,
        num_blocks: int,
        dropout: float = 0.0,
        device: str = "cuda",
        clamp_min: float = None,
        clamp_max: float = None
    ):
        super().__init__()
        # Condition encoder
        layers = [
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        for _ in range(num_blocks):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.condition_encoder = nn.Sequential(*layers)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, time_embed_dim)
        )

        # Main network
        total_dim = input_dim + hidden_dim + time_embed_dim
        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output = nn.Linear(hidden_dim, input_dim)
        self.device =  device
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x, cond, t):
        c = self.condition_encoder(cond)
        te = self.time_embed(t)
        h = F.relu(self.fc1(torch.cat([x, c, te], dim=1)))
        for block in self.res_blocks:
            h = block(h)
        out = self.output(h)
        if self.clamp_min is not None:
            out = torch.clamp(out, self.clamp_min, self.clamp_max)
        return out

class DDIMSampler:
    def __init__(self, noise_steps: int, beta_start: float, beta_end: float, device: str):
        self.noise_steps = noise_steps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, noise_steps, device=device)
        self.register_buffer = False
        self.beta = betas
        self.alpha = 1 - betas
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, batch_size: int):
        return torch.randint(1, self.noise_steps, (batch_size, 1), device=self.device)

    def noise_images(self, x, t):
        a_hat = self.alpha_hat[t]
        eps = torch.randn_like(x)
        return torch.sqrt(a_hat) * x + torch.sqrt(1 - a_hat) * eps, eps

    def sample(self, model, n, cond):
        x = torch.randn(n, 1, device=self.device)
        for t in reversed(range(1, self.noise_steps)):
            t_vec = torch.full((n, 1), t, dtype=torch.float32,device=self.device)
            pred_noise = model(x, cond, t_vec)
            a = self.alpha[t]
            a_hat = self.alpha_hat[t]
            b = self.beta[t]
            noise = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            x = 1 / a.sqrt() * (x - (1 - a) / (1 - a_hat).sqrt() * pred_noise) + b.sqrt() * noise
        return x

