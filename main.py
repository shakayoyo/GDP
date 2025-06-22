import argparse
import json
import os
import time

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode

from autogluon.multimodal import MultiModalPredictor
from train import train, run_hpo
from models import CombinedDiffusionNetwork, DDIMSampler
from utils import set_seed, path_expander, sample_batched, ToyDataset, extract_embeddings

from dataset import load_shopee_data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hpo_trials', type=int, default=20)
    parser.add_argument('--output_dir',    type=str, default='outputs')
    parser.add_argument('--hpo_config',    type=str, help='JSON file for HPO ranges')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_df, test_df = load_shopee_data(args.data_dir)

    # Initialize and fit AutoGluon predictor
    predictor = MultiModalPredictor(
        problem_type='classification',
        label='label',
        path=os.path.join(args.output_dir, 'ag_models'),
        eval_metric='accuracy'
    )
    
    automm_hyperparameters = {
       "model.names": ['timm_image','fusion_mlp'],
       "model.timm_image.checkpoint_name": "resnet18",
       "optim.lr": "1e-4",
       "optim.lr_decay": "1",
       "optim.max_epochs": "200",
    }
    
    predictor.fit(
        train_data=train_df,
        seed=args.seed,
        hyperparameters= automm_hyperparameters,
        holdout_frac=0.2
    )

    # Evaluate and print AutoGluon score
    ag_score = predictor.evaluate(test_df, )
    print(f"AutoGluon multimodal predictor score: {ag_score}")

    # Extract embeddings
    train_emb, test_emb = extract_embeddings(predictor, train_df, test_df,
                                             label='label')

    # Prepare targets and scaler
    scaler = MinMaxScaler(feature_range=(-1,1))
    y_train_full = scaler.fit_transform(train_df['label'].values.reshape(-1,1)).ravel()
    y_test       = test_df['label'].values

    # Reshape embeddings and split
    X = train_emb.reshape(train_emb.shape[0], -1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_train_full, test_size=0.2, random_state=args.seed
    )
    X_test = test_emb.reshape(test_emb.shape[0], -1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load or define HPO ranges
    if args.hpo_config:
        with open(args.hpo_config) as f:
            param_ranges = json.load(f)
    else:
        param_ranges = {
            'batch_size': [32,64,128],
            'lr': (1e-5,1e-4),
            'hidden_dim': [512,1024],
            'time_embed_dim': [64,128,256],
            'layers': (3,10),
            'noise_steps': (500,1000),
            'beta_start': 1e-4,
            'beta_end': (0.01,0.02),
            'dropout': (0.0,0.05),
            'epochs': 200,
            'patience': 20
        }

    # Hyperparameter optimization
    best_params = run_hpo(
        X_train, y_train, X_val, y_val,
        device, args.seed, param_ranges, n_trials=args.hpo_trials, 
    )
    #print('Best HPO params:', best_params)

    # Train final diffusion model
    model = CombinedDiffusionNetwork(
        input_dim=1,
        condition_dim=X_train.shape[1],
        hidden_dim=best_params['hidden_dim'],
        time_embed_dim=best_params['time_embed_dim'],
        num_blocks=best_params['layers'],
        dropout=best_params['dropout'],
        device = device,
    ).to(device)
    sampler = DDIMSampler(
        noise_steps=best_params['noise_steps'],
        beta_start=param_ranges['beta_start'],
        beta_end=best_params['beta_end'],
        device=device
    )
    train_loader = torch.utils.data.DataLoader(
        ToyDataset(X_train, y_train), batch_size=best_params['batch_size'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        ToyDataset(X_val, y_val),   batch_size=best_params['batch_size'], shuffle=False
    )

    trained_model, _ = train(
        model, train_loader, val_loader,
        sampler, best_params['lr'], epochs=1000, patience=100,
        seed=args.seed
    )

    # Generate samples, invert scaling, compute mode
    ms = sample_batched(sampler, trained_model, num_samples=1000, X_test=X_test,chunk_size=5)
    ms_orig = scaler.inverse_transform(ms.reshape(-1,1)).reshape(ms.shape)
    y_pred = mode(np.round(ms_orig).astype(int), axis=1).mode.flatten()

    # Metrics
    acc   = accuracy_score(y_test, y_pred)
    #kappa = cohen_kappa_score(y_test, y_pred)
    metrics = {'ag_score': ag_score, 'accuracy': acc}
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Metrics saved to', args.output_dir)

if __name__ == '__main__':
    main()