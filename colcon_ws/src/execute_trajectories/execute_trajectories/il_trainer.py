# il_trainer.py
import json
import os
import glob
import time
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import ILPolicy, TrainConfig
from model import BlockNormalizer, BlockNormalizer24


# ---------------- Dataset ----------------
class ILDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ---------------- Utilities ----------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    mse = 0.0
    mae = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        mse += nn.functional.mse_loss(pred, yb, reduction='sum').item()
        mae += nn.functional.l1_loss(pred, yb, reduction='sum').item()
        n += yb.numel()
    return mse / max(n, 1), mae / max(n, 1)


def split_by_episode(ep, val_ratio=0.2, seed=0):
    eps = np.unique(ep)
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n_val = max(1, int(len(eps) * val_ratio))
    val_eps = set(eps[:n_val])
    train_mask = ~np.isin(ep, list(val_eps))
    val_mask = ~train_mask
    return train_mask, val_mask, np.array(sorted(list(val_eps)))


def add_noise_augmentation(X: torch.Tensor, noise_std=0.01):
    return X + torch.randn_like(X) * noise_std if noise_std and noise_std > 0 else X


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------- Data Loading ----------------
def load_episode_data(data_path, pattern="run_*.npz"):
    """Load data from a single .npz or a directory of episode .npz files."""
    if os.path.isfile(data_path):
        print(f"Loading single file: {data_path}")
        data = np.load(data_path)

        # Build X
        if 'X' in data:
            X = data['X'].astype(np.float32)
        else:
            components = []
            for key in ['pos', 'vel', 'eff', 'ft']:
                if key in data:
                    components.append(data[key].astype(np.float32))
            if not components:
                raise KeyError("Could not construct X from file: missing pos/vel/eff/ft or 'X'")
            X = np.concatenate(components, axis=1)

        # Labels Y
        if 'Y' in data:
            Y = data['Y'].astype(np.float32)
        elif 'y_cmd' in data:
            Y = data['y_cmd'].astype(np.float32)
        elif 'y_qdes' in data:
            Y = data['y_qdes'].astype(np.float32)
        elif 'qdes' in data:
            Y = data['qdes'].astype(np.float32)
        elif 'action' in data:
            Y = data['action'].astype(np.float32)
        else:
            raise KeyError("No valid target labels found. Expected one of Y, y_cmd, y_qdes, qdes, action")

        ep = data['ep'] if 'ep' in data else np.zeros((X.shape[0],), dtype=np.int32)
        return X, Y, ep

    if os.path.isdir(data_path):
        episode_files = sorted(glob.glob(os.path.join(data_path, pattern)))
        if not episode_files:
            raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {data_path}")

        print(f"Found {len(episode_files)} episode files")
        X_list, Y_list, ep_list = [], [], []

        for ep_idx, filepath in enumerate(tqdm(episode_files, desc="Loading episodes")):
            try:
                data = np.load(filepath)
                if 'X' in data:
                    X_ep = data['X'].astype(np.float32)
                else:
                    parts = []
                    for key in ['pos', 'vel', 'eff', 'ft']:
                        if key in data:
                            parts.append(data[key].astype(np.float32))
                    if not parts:
                        print(f"Warning: no state data in {filepath}")
                        continue
                    X_ep = np.concatenate(parts, axis=1)

                if 'Y' in data:
                    Y_ep = data['Y'].astype(np.float32)
                elif 'y_cmd' in data:
                    Y_ep = data['y_cmd'].astype(np.float32)
                elif 'y_qdes' in data:
                    Y_ep = data['y_qdes'].astype(np.float32)
                elif 'qdes' in data:
                    Y_ep = data['qdes'].astype(np.float32)
                elif 'action' in data:
                    Y_ep = data['action'].astype(np.float32)
                else:
                    print(f"Warning: no targets in {filepath}")
                    continue

                n = min(X_ep.shape[0], Y_ep.shape[0])
                X_ep = X_ep[:n]
                Y_ep = Y_ep[:n]
                ep_indices = np.full((n,), ep_idx, dtype=np.int32)

                X_list.append(X_ep)
                Y_list.append(Y_ep)
                ep_list.append(ep_indices)

            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

        if not X_list:
            raise ValueError("No valid episode data could be loaded")

        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        ep = np.concatenate(ep_list, axis=0)
        print(f"Total data loaded: {len(X_list)} episodes, {X.shape[0]} samples")
        return X, Y, ep

    raise FileNotFoundError(f"Data path not found: {data_path}")


def print_data_info(X, Y, ep):
    n_episodes = len(np.unique(ep))
    ep_lengths = [int(np.sum(ep == ep_id)) for ep_id in np.unique(ep)]

    print("\nDataset summary:")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Input features: {X.shape[1]}")
    print(f"  Output features: {Y.shape[1]}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Episode lengths - min: {min(ep_lengths)}, max: {max(ep_lengths)}, "
          f"mean: {np.mean(ep_lengths):.1f}, std: {np.std(ep_lengths):.1f}")


# ---------------- Trainer Object ----------------
class ILTrainer:
    """
    Reusable trainer that can be called multiple times as new data arrives.
    It keeps the same model/optimizer/scheduler between calls (incremental training).
    """

    def __init__(
        self,
        out_dir: str = "trained_models",
        model_ctor=ILPolicy,              # swap to ILPolicyV2 if desired
        model_kwargs: Optional[dict] = None,
        normalize: bool = True,
        seed: int = 0,
        device: Optional[torch.device] = None,
        amp: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or _get_device()
        self.normalize = normalize
        self.amp_enabled = amp and (self.device.type == "cuda")  # safe default

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.model_ctor = model_ctor
        self.model_kwargs = model_kwargs or {}
        self.model: Optional[nn.Module] = None
        self.opt: Optional[torch.optim.Optimizer] = None
        self.sched: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)
        self.norm = None
        self.best_val = float("inf")
        self.cfg: Optional[TrainConfig] = None

        self.ckpt_path = self.out_dir / "best.pt"
        self.normalizer_path = self.out_dir / "normalizer.json"
        self.config_path = self.out_dir / "config.json"

        # ---- metrics files ----
        self.metrics_jsonl = self.out_dir / "metrics.jsonl"
        self.metrics_csv = self.out_dir / "metrics.csv"
        self.history: List[Dict[str, Any]] = []
        self._resume_metrics()  # load existing metrics if present

        # Try to resume if a checkpoint already exists
        if self.ckpt_path.exists():
            self._load_checkpoint(self.ckpt_path)

        # store base hparams
        self._base_lr = float(lr)
        self._base_wd = float(weight_decay)

    # ---------- Public API ----------
    def fit(
        self,
        data_path: str,
        pattern: str = "run_*.npz",
        hidden: int = 64,
        layers: int = 4,
        dropout: float = 0.1,
        batch_size: int = 512,
        epochs: int = 100,
        val_ratio: float = 0.2,
        num_workers: int = 4,
        noise_std: float = 0.005,
        grad_clip: float = 1.0,
        control_mode: str = "position_delta",
        min_episodes: int = 2,
        seed: int = 0,
    ):
        """
        Train (or continue training) on trajectories under `data_path`.
        Calling this again later will keep improving the current model.
        """
        # Load data
        X, Y, ep = load_episode_data(data_path, pattern)
        print_data_info(X, Y, ep)

        # Validate episodes
        n_episodes = len(np.unique(ep))
        if n_episodes < min_episodes:
            raise ValueError(f"Need at least {min_episodes} episodes, got {n_episodes}")

        # Split by episode
        train_mask, val_mask, val_eps = split_by_episode(ep, val_ratio=val_ratio, seed=seed)
        print(f"Train episodes: {len(np.unique(ep[train_mask]))}, Val episodes: {len(val_eps)}")
        print(f"Validation episodes: {val_eps}")

        # Normalization (initialize once; optional refresh if new dims)
        if self.normalize:
            if self.norm is None:
                # initialize a normalizer based on feature count
                idx_q = np.arange(0, 6)
                idx_qd = np.arange(6, 12)
                idx_ft = np.arange(18, 24) if X.shape[1] >= 24 else np.arange(18, X.shape[1])
                if X.shape[1] == 24:
                    self.norm = BlockNormalizer24(idx_ft)
                else:
                    self.norm = BlockNormalizer(idx_q, idx_qd, np.array([]), idx_ft)
                self.norm.fit(X[train_mask])
                print("Fitted new normalizer.")
                
            X = self.norm.transform(X)
            print(f"Applied normalization to {X.shape[1]} features")

        # Remove NaNs
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            print("Warning: NaN values found; removing affected samples")
            nan_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
            X, Y = X[nan_mask], Y[nan_mask]
            train_mask = train_mask[nan_mask]
            val_mask = val_mask[nan_mask]

        print(f"X stats - mean: {X.mean():.4f}, std: {X.std():.4f}, range: [{X.min():.4f}, {X.max():.4f}]")
        print(f"Y stats - mean: {Y.mean():.4f}, std: {Y.std():.4f}, range: [{Y.min():.4f}, {Y.max():.4f}]")

        # Datasets/Loaders
        Xtr, Ytr = X[train_mask], Y[train_mask]
        Xva, Yva = X[val_mask], Y[val_mask]
        print(f"Training samples: {Xtr.shape[0]}, Validation samples: {Xva.shape[0]}")

        if Xtr.shape[0] < batch_size:
            print(f"Warning: train data ({Xtr.shape[0]}) < batch size ({batch_size}), reducing batch.")
            batch_size = max(1, min(batch_size, Xtr.shape[0] // 2))

        dtr = ILDataset(Xtr, Ytr)
        dva = ILDataset(Xva, Yva)

        tr_loader = DataLoader(
            dtr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=False, persistent_workers=(num_workers > 0)
        )
        va_loader = DataLoader(
            dva, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=False, persistent_workers=(num_workers > 0)
        )

        # Model/optim (initialize only once; reuse on subsequent calls)
        if self.model is None:
            input_dim = X.shape[1]
            output_dim = Y.shape[1]
            self.model = self.model_ctor(
                input_dim=input_dim, output_dim=output_dim,
                hidden=hidden, layers=layers, dropout=dropout, **self.model_kwargs
            ).to(self.device)

            self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(self._get_lr()), weight_decay=float(self._get_wd()))
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=10, min_lr=1e-6)

            print(f"Model: {sum(p.numel() for p in self.model.parameters())} parameters")

            # Save a (train) config snapshot
            self.cfg = TrainConfig(
                input_dim=input_dim, output_dim=output_dim,
                hidden=hidden, layers=layers, dropout=dropout,
                lr=float(self._get_lr()), weight_decay=float(self._get_wd()),
                batch_size=batch_size, epochs=epochs, amp=self.amp_enabled,
                num_workers=num_workers, seed=seed, val_ratio=val_ratio,
                grad_clip=grad_clip, noise_std=noise_std, control_mode=control_mode
            )
            with open(self.config_path, "w") as f:
                json.dump(asdict(self.cfg), f, indent=2)

        # Save normalizer (whenever (re)fit)
        if self.normalize and self.norm is not None:
            with open(self.normalizer_path, "w") as f:
                json.dump(self.norm.state_dict(), f, indent=2)

        # Episode info (for reference)
        episode_info = {
            'total_episodes': int(len(np.unique(ep))),
            'train_episodes': int(len(np.unique(ep[train_mask]))),
            'val_episodes': int(len(np.unique(ep[val_mask]))),
            'val_episode_ids': np.unique(ep[val_mask]).tolist(),
            'episode_lengths': {int(e): int(np.sum(ep == e)) for e in np.unique(ep)}
        }
        with open(self.out_dir / 'episode_info.json', 'w') as f:
            json.dump(episode_info, f, indent=2)

        # Train
        print(f"\nStarting/continuing training for {epochs} epochs on device: {self.device}")
        for epoch in range(1, epochs + 1):
            epoch_t0 = time.time()
            self.model.train()
            pbar = tqdm(tr_loader, desc=f'Epoch {epoch:3d}/{epochs}', leave=False)

            epoch_loss = 0.0
            epoch_samples = 0
            steps = 0

            for xb, yb in pbar:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                xb = add_noise_augmentation(xb, noise_std)

                self.opt.zero_grad(set_to_none=True)
                # autocast: use 'cuda' when on cuda, else fall back to 'cpu'
                device_type = "cuda" if self.device.type == "cuda" else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=self.amp_enabled):
                    pred = self.model(xb)
                    loss = nn.functional.mse_loss(pred, yb)

                self.scaler.scale(loss).backward()

                if grad_clip and grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.scaler.step(self.opt)
                self.scaler.update()

                batch_loss = loss.item()
                bs = xb.size(0)
                epoch_loss += batch_loss * bs
                epoch_samples += bs
                steps += 1
                pbar.set_postfix({'loss': f'{batch_loss:.6f}', 'lr': f'{self.opt.param_groups[0]["lr"]:.2e}'})

            avg_train_loss = epoch_loss / max(epoch_samples, 1)

            val_mse, val_mae = evaluate(self.model, va_loader, self.device)
            if self.sched is not None:
                self.sched.step(val_mse)

            cur_lr = float(self.opt.param_groups[0]['lr'])
            wall = float(time.time() - epoch_t0)

            print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_mse={val_mse:.6f}, "
                  f"val_mae={val_mae:.6f}, lr={cur_lr:.2e}, steps={steps}, time={wall:.2f}s")

            # ---- record metrics ----
            self._record_metric({
                "epoch": epoch if not self.history else self.history[-1]["epoch"] + 1,
                "train_loss": float(avg_train_loss),
                "val_mse": float(val_mse),
                "val_mae": float(val_mae),
                "lr": cur_lr,
                "steps": int(steps),
                "train_samples": int(epoch_samples),
                "time_sec": wall,
                "data_path": str(data_path),
            })

            if val_mse < self.best_val:
                self.best_val = val_mse
                self._save_checkpoint(self.ckpt_path, epoch, avg_train_loss, val_mse, episode_info)
                print(f"✓ Saved new best model (val_loss: {val_mse:.6f})")

        # Final eval (on the best)
        self._load_checkpoint(self.ckpt_path)
        final_mse, final_mae = evaluate(self.model, va_loader, self.device)
        print(f"Final evaluation - MSE: {final_mse:.6f}, MAE: {final_mae:.6f}")

        return {
            "best_val_mse": self.best_val,
            "final_val_mse": final_mse,
            "final_val_mae": final_mae,
            "checkpoint": str(self.ckpt_path),
            "metrics_jsonl": str(self.metrics_jsonl),
            "metrics_csv": str(self.metrics_csv),
        }

    # ---------- Metrics helpers ----------
    def _resume_metrics(self):
        """Load existing metrics from jsonl if present (for continuous logging across fit calls)."""
        if self.metrics_jsonl.exists():
            try:
                with open(self.metrics_jsonl, "r") as f:
                    for line in f:
                        self.history.append(json.loads(line))
            except Exception:
                # If corrupted, start fresh (but don't delete the file)
                pass

    def _record_metric(self, row: Dict[str, Any]):
        """Append both JSONL and CSV and keep in-memory history."""
        self.history.append(row)

        # JSONL append
        with open(self.metrics_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")

        # CSV append (create header if not exists)
        write_header = not self.metrics_csv.exists()
        with open(self.metrics_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch", "train_loss", "val_mse", "val_mae",
                    "lr", "steps", "train_samples", "time_sec", "data_path"
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def get_history(self) -> List[Dict[str, Any]]:
        """In-memory metrics; useful for programmatic plotting."""
        return list(self.history)

    # ---------- Checkpointing ----------
    def _save_checkpoint(self, path: Path, epoch: int, train_loss: float, val_loss: float, episode_info: dict):
        ckpt = {
            'model_state': self.model.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val': self.best_val,
            'cfg': asdict(self.cfg) if self.cfg else None,
            'input_dim': next(self.model.parameters()).shape,  # not strictly needed
            'episode_info': episode_info,
        }
        if self.normalize and self.norm is not None:
            ckpt['normalizer'] = self.norm.state_dict()
        torch.save(ckpt, path)

    def _load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location='cpu')
        # Rebuild model if needed
        if self.model is None:
            if self.cfg is None and 'cfg' in ckpt and ckpt['cfg'] is not None:
                # basic rebuild from saved config
                cfg = ckpt['cfg']
                self.cfg = TrainConfig(**cfg)
                self.model = self.model_ctor(
                    input_dim=self.cfg.input_dim,
                    output_dim=self.cfg.output_dim,
                    hidden=self.cfg.hidden,
                    layers=self.cfg.layers,
                    dropout=self.cfg.dropout,
                    **self.model_kwargs
                ).to(self.device)
            elif self.cfg is None:
                raise RuntimeError("No config available to rebuild model.")
        self.model.load_state_dict(ckpt['model_state'])

        # Recreate optimizer/scheduler if missing
        if self.opt is None:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(self._get_lr()), weight_decay=float(self._get_wd()))
        if self.sched is None:
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=10, min_lr=1e-6)

        # Restore normalizer if present and not already in memory
        if self.normalize and ('normalizer' in ckpt) and (self.norm is None):
            idx_ft = np.arange(18, 24)
            norm_data = ckpt['normalizer']
            print("[norm] Loaded normalizer from checkpoint")
            self.norm = BlockNormalizer24(idx_ft)     
            self.norm.load_state_dict(norm_data)       

        self.best_val = float(ckpt.get('best_val', self.best_val))

    # ---------- Hyperparam helpers ----------
    def _get_lr(self):
        if self.cfg is not None:
            return self.cfg.lr
        return self._base_lr

    def _get_wd(self):
        if self.cfg is not None:
            return self.cfg.weight_decay
        return self._base_wd


def main():
    print("Starting Training...")
    trainer = ILTrainer(
        out_dir="trained_models_no_randomization_plot",
        model_ctor=ILPolicy,      # or ILPolicyV2
        model_kwargs={},          # extra args for your model class if needed
        normalize=True,
        seed=0,
        amp=False,                # set True if training on CUDA and you want mixed precision
        lr=1e-3,
        weight_decay=1e-4,
    )

    trainer.fit(
        data_path="real_data/trial_3_no_randomization",
        pattern="run_*.npz",
        hidden=64,
        layers=4,
        dropout=0.1,
        batch_size=512,
        epochs=50,
        val_ratio=0.2,
        num_workers=4,
        noise_std=0.005,
        grad_clip=1.0,
        control_mode="position_delta",
        min_episodes=2,
        seed=0,
    )

    # trainer.fit(
    #     data_path="real_data/trial_3_no_randomization",
    #     pattern="run_*.npz",
    #     epochs=25,
    # )


if __name__ == '__main__':
    main()
