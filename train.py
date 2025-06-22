import copy
import torch
import torch.nn as nn
import optuna
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils import set_seed, ToyDataset
from models import CombinedDiffusionNetwork, DDIMSampler



def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    sampler,
    lr: float,
    epochs: int,
    patience: int,
    seed:int,
):
    set_seed(seed)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(model.device), y.unsqueeze(1).to(model.device)
            t = sampler.sample_timesteps(y.size(0)).to(model.device).view(-1,1)
            y_t, noise = sampler.noise_images(y, t)
            #print(y_t,x,t)
            pred = model(y_t, x, t.float())
            loss = criterion(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(model.device), y.unsqueeze(1).to(model.device)
                t = sampler.sample_timesteps(y.size(0)).to(model.device).view(-1,1)
                y_t, noise = sampler.noise_images(y, t)
                pred = model(y_t, x, t.float())
                val_loss += criterion(pred, noise).item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}  Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_weights)
    return model, best_loss



def objective(
    trial,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    seed,
    param_ranges: dict
):
    """
    Optuna objective that builds and evaluates a diffusion model using
    hyperparameter ranges provided in param_ranges.

    param_ranges should be a dict with keys:
      - 'batch_size': list of ints
      - 'lr': (low, high) tuple for loguniform
      - 'hidden_dim': list of ints
      - 'time_embed_dim': list of ints
      - 'layers': (min_int, max_int)
      - 'noise_steps': (min_int, max_int)
      - 'beta_end': (low, high) tuple
      - 'dropout': (low, high) tuple for float
    """
    # Sample hyperparameters from provided ranges
    batch_size      = trial.suggest_categorical('batch_size', param_ranges['batch_size'])
    lr              = trial.suggest_loguniform('lr', *param_ranges['lr'])
    hidden_dim      = trial.suggest_categorical('hidden_dim', param_ranges['hidden_dim'])
    time_dim        = trial.suggest_categorical('time_embed_dim', param_ranges['time_embed_dim'])
    num_blocks      = trial.suggest_int('layers', *param_ranges['layers'])
    noise_steps     = trial.suggest_int('noise_steps', *param_ranges['noise_steps'])
    beta_end        = trial.suggest_float('beta_end', *param_ranges['beta_end'])
    dropout_p       = trial.suggest_float('dropout', *param_ranges['dropout'])

    # Prepare data loaders
    train_ds = ToyDataset(X_train, y_train)
    val_ds   = ToyDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build model
    model = CombinedDiffusionNetwork(
        input_dim=1,
        condition_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        time_embed_dim=time_dim,
        num_blocks=num_blocks,
        dropout=dropout_p,
        device=device
    ).to(device)
    sampler = DDIMSampler(
        noise_steps=noise_steps,
        beta_start=param_ranges.get('beta_start', 1e-4),
        beta_end=beta_end,
        device=device
    )

    # Train
    model, val_loss = train(
        model,
        train_loader,
        val_loader,
        sampler,
        lr,
        epochs=param_ranges.get('epochs', 100),
        patience=param_ranges.get('patience', 10),
        seed=seed
    )
    return val_loss




def run_hpo(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    seed,
    param_ranges: dict,
    n_trials: int = 20
):
    """
    Run an Optuna study to optimize the diffusion model hyperparameters.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda t: objective(
            t, X_train, y_train, X_val, y_val, device, seed, param_ranges
        ),
        n_trials=n_trials
    )
    return study.best_params
