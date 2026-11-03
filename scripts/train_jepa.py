"""
Script 2: Train PAG-JEPA Adapter on RTX 4050 (VICReg Objective)
Trains the lightweight temporal pooling and orthogonal projection head (~1.5M parameters).
Training time: < 2.5 hours on 56k clips (or minutes on sample batch).
Peak VRAM: < 1.8 GB.
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from src.models.pag_jepa import PAG_JEPA
from src.losses.vicreg import VICRegLoss
from src.data.codebook import generate_synthetic_aligned_pair


def train(config_path: str = "configs/config.yaml", epochs: int = 10, batch_size: int = 64):
    print("=" * 75)
    print("TRAINING PAG-JEPA (PLATONIC ATTRIBUTE GROUNDING)")
    print("Architecture: Non-Autoregressive JEPA + Orthogonal Procrustes Projection")
    print("Objective: VICReg (Invariance + Variance Hinge + Covariance Decorrelation)")
    print("=" * 75)

    # 1. Config
    if not os.path.isabs(config_path):
        config_path = os.path.join(repo_root, config_path)
    
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    embed_dim = cfg.get("model", {}).get("embed_dim", 384)
    models_dir = os.path.join(repo_root, "outputs", "models")
    os.makedirs(models_dir, exist_ok=True)

    # 2. Model & Loss
    model = PAG_JEPA(
        embed_dim=embed_dim,
        temporal_layers=cfg.get("model", {}).get("temporal_layers", 2),
        temporal_heads=cfg.get("model", {}).get("temporal_heads", 4),
        grounding_threshold=cfg.get("model", {}).get("grounding_threshold", 0.62)
    ).to(device)

    loss_cfg = cfg.get("loss", {})
    criterion = VICRegLoss(
        lambda_inv=loss_cfg.get("lambda_inv", 25.0),
        mu_var=loss_cfg.get("mu_var", 25.0),
        nu_cov=loss_cfg.get("nu_cov", 1.0),
        gamma=loss_cfg.get("variance_gamma", 1.0)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("training", {}).get("learning_rate", 3e-4)), weight_decay=1e-5)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,} (~{trainable_params*4 / (1024*1024):.2f} MB)")

    # 3. Data setup (Synthetic / Cached feature tensors)
    print("\nPreparing training data (T=8 temporal frames, d=384)...")
    N = 1000
    Zv_raw, Za_raw = generate_synthetic_aligned_pair(num_samples=N, dim=embed_dim, condition="good_views")
    
    # Broadcast to T=8 frames with small temporal jitter
    T = 8
    X_frames = np.repeat(Zv_raw[:, np.newaxis, :], T, axis=1)
    X_frames += np.random.randn(*X_frames.shape) * 0.05
    
    dataset = TensorDataset(
        torch.tensor(X_frames, dtype=torch.float32),
        torch.tensor(Za_raw, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Training loop
    print(f"\nStarting {epochs} training epochs...")
    start_time = time.time()
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        inv_loss_acc = 0.0
        var_loss_acc = 0.0
        cov_loss_acc = 0.0

        for frames, targets in loader:
            frames, targets = frames.to(device), targets.to(device)
            optimizer.zero_grad()

            proj = model(frames)
            loss, metrics = criterion(proj, targets)
            loss.backward()
            optimizer.step()

            # Orthogonal Procrustes projection step
            model.projector.project_orthogonal()

            total_epoch_loss += loss.item()
            inv_loss_acc += metrics["loss_invariance"]
            var_loss_acc += metrics["loss_variance"]
            cov_loss_acc += metrics["loss_covariance"]

        num_batches = len(loader)
        avg_loss = total_epoch_loss / num_batches
        avg_inv = inv_loss_acc / num_batches
        avg_var = var_loss_acc / num_batches
        avg_cov = cov_loss_acc / num_batches

        if epoch % 2 == 0 or epoch == 1:
            print(f"Epoch [{epoch:>2}/{epochs:>2}] | Total Loss: {avg_loss:.4f} (Inv: {avg_inv:.4f}, Var: {avg_var:.4f}, Cov: {avg_cov:.4f})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(models_dir, "pag_jepa_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": cfg
            }, ckpt_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s!")
    print(f"Best checkpoint saved to: {os.path.join(models_dir, 'pag_jepa_best.pt')}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    train(args.config, epochs=args.epochs, batch_size=args.batch_size)
