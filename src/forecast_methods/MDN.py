"""Mixture density network with skewed Student-t components."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import Parallel, delayed
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted
from torch.special import gammaln
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from src.forecast_methods.utils import recalibrate_test_density
from src.utils.setup_logger import setup_logger


class LSTMBackbone(nn.Module):
    """Stacked LSTM encoder for univariate time series.

    Maps a flat lagged input (batch, seq_len) to a feature vector (batch, out_dim)
    feeding the MDN heads. Input is expected to be already standardised by the
    MDN's scaler_x.

    The sequence is treated as a (batch, seq_len, 1) univariate stream: each lag
    is one time step with one feature. The hidden state of the final LSTM layer
    at the last time step is projected to ``out_dim``.
    """

    def __init__(
        self,
        seq_len,
        out_dim,
        hidden_size=64,
        n_layers=1,
        dropout=0.0,
    ):
        """Build the LSTM backbone."""
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, out_dim)

    def forward(self, x):  # x: (batch, seq_len)
        """Return the head output for a batch of lag windows."""
        z = x.unsqueeze(-1)  # (batch, seq_len, 1)
        _, (h_n, _) = self.lstm(z)  # h_n: (n_layers, batch, hidden_size)
        h_last = h_n[-1]  # final layer's last hidden state
        return self.head(self.drop(h_last))  # (batch, out_dim)


def check_tensor(x, dtype=torch.float32):
    """Return ``x`` as a tensor of ``dtype``, without re-wrapping if it is one."""
    if isinstance(x, torch.Tensor):
        return x if x.dtype == dtype else x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


def skewt_pdf(y_grid, mu, sigma, nu, skew):
    """Compute the Fernandez-Steel skew-Student-t PDF on the grid.

    f(x; mu, sigma, xi, nu) = 2 / (sigma (xi + 1/xi)) * t_nu(z * xi^{-sign(z)}),
    with z = (x - mu) / sigma and t_nu the standard Student-t pdf (nu df).

    The `skew` argument is g = log(xi) (so xi = exp(g)); xi > 1 is right-skewed,
    xi < 1 left-skewed, xi = 1 (g = 0) symmetric. Using g = log(xi) keeps the
    normaliser log(2 / (xi + 1/xi)) = log 2 - logaddexp(g, -g) numerically stable
    for any g. The half-axis scaling xi^{-sign(z)} is exp(-sign(z) * g).
    """
    y_grid = check_tensor(y_grid)
    mu = check_tensor(mu)
    sigma = check_tensor(sigma)
    nu = check_tensor(nu)
    skew = check_tensor(skew)

    device = y_grid.device
    mu = mu.to(device)
    sigma = sigma.to(device)
    nu = nu.to(device)
    skew = skew.to(device)

    z = (y_grid - mu) / sigma
    g = skew

    # Two-piece scaled argument: z / xi for z >= 0, z * xi for z < 0.
    arg = z * torch.exp(-torch.sign(z) * g)

    # log t_nu(arg) via the shared stable standard-Student log-pdf.
    log_ft = _log_student_pdf_std(arg, nu)

    # log(2 / (xi + 1/xi)) = log 2 - log(e^g + e^{-g}).
    log_norm = np.log(2.0) - torch.logaddexp(g, -g)

    return torch.exp(log_norm + log_ft - torch.log(sigma))


def _log_student_pdf_std(x, nu):
    """Log of the standard Student-t pdf f_t(x; nu) (location 0, scale 1)."""
    x = check_tensor(x)
    nu = check_tensor(nu).to(x.device)
    pi_t = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
    return (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * torch.log(nu * pi_t)
        - (nu + 1) / 2 * torch.log1p(x * x / nu)
    )


def mixture_nll(log_pi, mu, sigma, nu, skew, target, weights, density):
    """Negative log-likelihood of a univariate mixture.

    All parameter tensors are (batch_size, n_mixtures); ``weights`` reweights the
    per-observation terms.
    """
    if density != "skewt":
        raise ValueError(f"Unsupported density {density!r}; only 'skewt' is supported.")

    batch_size, n_mixtures = mu.shape

    # Expand target to match mixture dimension
    target_expanded = target.unsqueeze(1).expand(batch_size, n_mixtures)

    # Standardize: z_tilde = (Y - mu) / sigma
    z_tilde = (target_expanded - mu) / sigma

    # Fernandez-Steel Skewed Student's t log-likelihood
    log_sigma = torch.log(sigma)
    g = skew
    arg = z_tilde * torch.exp(-torch.sign(z_tilde) * g)
    log_ft = _log_student_pdf_std(arg, nu)  # log t_nu(arg)
    log_norm = np.log(2) - torch.logaddexp(g, -g)
    log_prob_components = log_norm + log_ft - log_sigma

    # Add log mixture weights (exact; no additive epsilon)
    weighted_log_prob = log_pi + log_prob_components

    # Log-sum-exp across mixture components
    log_sum_exp = torch.logsumexp(weighted_log_prob, dim=1)

    loss = (-log_sum_exp * weights).mean()

    return loss


def weighted_mse(pred, target, weights):
    """Weighted MSE with a fixed denominator; weights are pre-normalized."""
    se = (pred - target) ** 2
    return (se * weights).mean()


NU_MIN = 2.0
NU_MAX = 100.0
NU_INIT = 5.0
MAX_PCT_SKIPPED = 5.0
G_MAX = 1.5
SIGMA_MIN = 1e-6

NU_SATURATION = 1000.0
SATURATED_FAMILIES = {
    "gaussian": ("skewt", True, True),
    "skewnorm": ("skewt", True, False),
    "student": ("skewt", False, True),
}


class MixtureDensityNetwork(nn.Module):
    """MDN with several component families for heavy-tailed targets."""

    def __init__(
        self,
        input_dim,
        hidden_layers,
        n_mixtures,
        point_head_layers,
        device,
        n_jobs,
        density,
        dropout,
        point_head_augmented=False,
    ):
        """Build the network.

        ``hidden_layers`` sizes the LSTM backbone: its length is the number of stacked
        recurrent layers, the entries the hidden width, and the last entry the feature
        width fed to the MDN heads.
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_mixtures = n_mixtures
        self.device = torch.device(device)
        self.n_jobs = n_jobs
        self.density = density
        self.dropout = dropout
        self.point_head_augmented = point_head_augmented

        self._model_density, self.saturate_nu, self.saturate_skew = _resolve_family(
            density
        )
        if self._model_density != "skewt":
            raise ValueError(
                f"Unsupported density {density!r}; supported: 'skewt' and its "
                f"saturated families {sorted(SATURATED_FAMILIES)}."
            )

        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()

        # LSTM backbone: maps input to a feature vector of width hidden_layers[-1].
        # nn.LSTM requires all stacked layers to share one hidden_size.
        self.hidden = LSTMBackbone(
            seq_len=input_dim,
            out_dim=hidden_layers[-1],
            hidden_size=hidden_layers[-1],
            n_layers=len(hidden_layers),
            dropout=dropout,
        )

        # Output layers
        # Mixture weights
        self.pi_layer = nn.Linear(hidden_layers[-1], n_mixtures)

        # Distribution parameters for each mixture
        self.mu_layer = nn.Linear(hidden_layers[-1], n_mixtures)
        self.sigma_layer = nn.Linear(hidden_layers[-1], n_mixtures)

        # Point prediction head
        point_layers = []
        in_dim = hidden_layers[-1] + (input_dim + 1 if point_head_augmented else 0)
        for dim in point_head_layers:
            point_layers.append(nn.Linear(in_dim, dim))
            point_layers.append(nn.ReLU())
            in_dim = dim
        point_layers.append(nn.Linear(in_dim, 1))
        self.point_head = nn.Sequential(*point_layers)

        self.nu_layer = nn.Linear(hidden_layers[-1], n_mixtures)
        self.lam_layer = nn.Linear(hidden_layers[-1], n_mixtures)

        # Initialise the tail-shape heads
        if self.nu_layer is not None:
            nn.init.zeros_(self.nu_layer.weight)
            nn.init.constant_(
                self.nu_layer.bias,
                float(np.log((NU_INIT - NU_MIN) / (NU_MAX - NU_INIT))),
            )
        if self.lam_layer is not None:
            nn.init.zeros_(self.lam_layer.weight)
            nn.init.zeros_(self.lam_layer.bias)

    def forward(self, x):
        """Return the mixture parameters (pi, log_pi, mu, sigma, nu, skew)."""
        x = x.to(self.device)
        hidden_state = self.hidden(x)

        # Mixture weights
        log_pi = F.log_softmax(self.pi_layer(hidden_state), dim=-1)
        pi = log_pi.exp()

        # Basic parameters
        mu = self.mu_layer(hidden_state)
        sigma = F.softplus(self.sigma_layer(hidden_state)) + SIGMA_MIN

        nu = NU_MIN + (NU_MAX - NU_MIN) * torch.sigmoid(self.nu_layer(hidden_state))
        skew = G_MAX * torch.tanh(self.lam_layer(hidden_state))
        if self.saturate_nu:
            nu = torch.full_like(mu, NU_SATURATION)
        if self.saturate_skew:
            skew = torch.zeros_like(mu)

        return pi, log_pi, mu, sigma, nu, skew

    def normalize_data(self, X, y):
        """Fit the scalers on the training data and return the scaled tensors."""
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if y.dim() > 1:
            y = y.squeeze()

        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().reshape(-1, 1)

        # y is always globally scaled
        self.scaler_y.fit(y_np)
        y_norm = torch.tensor(
            self.scaler_y.transform(y_np).ravel(), dtype=torch.float32, device=y.device
        )

        self.scaler_x.fit(X_np)
        X_norm = torch.tensor(
            self.scaler_x.transform(X_np), dtype=torch.float32, device=X.device
        )

        return X_norm, y_norm

    def renormalize_test(self, X, y=None):
        """Scale test data with the fitted training scalers."""
        if X.dim() == 1:
            X = X.unsqueeze(1)

        check_is_fitted(self.scaler_x)
        X_norm = torch.tensor(
            self.scaler_x.transform(X.detach().cpu().numpy()),
            dtype=torch.float32,
            device=X.device,
        )

        if y is not None:
            if y.dim() > 1:
                y = y.squeeze()
            y_norm = torch.tensor(
                self.scaler_y.transform(
                    y.detach().cpu().numpy().reshape(-1, 1)
                ).ravel(),
                dtype=torch.float32,
                device=y.device,
            )
            return X_norm, y_norm
        return X_norm

    def denormalize_params(self, pi, mu, sigma, nu, skew):
        """Map the mixture parameters back to the original scale."""
        mu_y = torch.tensor(self.scaler_y.center_[0], dtype=mu.dtype, device=mu.device)
        sigma_y = torch.tensor(
            self.scaler_y.scale_[0], dtype=sigma.dtype, device=sigma.device
        )

        mu_denorm = mu_y + sigma_y * mu
        sigma_denorm = sigma_y * sigma

        return pi, mu_denorm, sigma_denorm, nu, skew

    def prepare_dataloaders(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size,
        weights_train=None,
        weights_val=None,
        sampler=None,
    ):
        """Build the training and validation loaders."""
        train_dataset = TensorDataset(
            X_train.detach().cpu(), y_train.detach().cpu(), weights_train.detach().cpu()
        )

        if sampler:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )

        if X_val is not None and y_val is not None:
            test_dataset = TensorDataset(
                X_val.detach().cpu(),
                y_val.detach().cpu(),
                weights_val.detach().cpu(),
            )
            val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        return train_loader, val_loader

    def _last_lag_in_y_scale(self, X_norm):
        """Return the last observed lag on the normalized target scale."""
        cx = torch.as_tensor(
            self.scaler_x.center_[-1], dtype=X_norm.dtype, device=X_norm.device
        )
        sx = torch.as_tensor(
            self.scaler_x.scale_[-1], dtype=X_norm.dtype, device=X_norm.device
        )
        cy = torch.as_tensor(
            self.scaler_y.center_[0], dtype=X_norm.dtype, device=X_norm.device
        )
        sy = torch.as_tensor(
            self.scaler_y.scale_[0], dtype=X_norm.dtype, device=X_norm.device
        )
        return ((X_norm[:, -1:] * sx + cx) - cy) / sy

    def _predict_point_norm(self, X_norm):
        """Normalized-scale point prediction, shared by every point-head path."""
        hidden_state = self.hidden(X_norm)
        if not self.point_head_augmented:
            return self.point_head(hidden_state).squeeze(-1)

        pi = F.softmax(self.pi_layer(hidden_state), dim=-1)
        mu = self.mu_layer(hidden_state)
        mix_loc = (pi * mu).sum(dim=1, keepdim=True)  # y-normalized scale
        feats = torch.cat([hidden_state, X_norm, mix_loc], dim=1)
        return self.point_head(feats).squeeze(-1) + self._last_lag_in_y_scale(
            X_norm
        ).squeeze(-1)

    def _objective_loss(self, X_batch, y_batch, weights, objective):
        """Per-batch training objective, shared by the train and validation loops."""
        if objective == "mse":
            pred = self._predict_point_norm(X_batch)
            return weighted_mse(pred, y_batch, weights)

        _, log_pi, mu, sigma, nu, skew = self(X_batch)
        return mixture_nll(
            log_pi, mu, sigma, nu, skew, y_batch, weights, density=self._model_density
        )

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        weights_train,
        weights_val,
        max_epochs,
        learning_rate,
        batch_size,
        max_norm,
        patience,
        patience_scheduler,
        factor_scheduler,
        weighting="both",
        objective="nll",
    ):
        """
        Fit the model with data normalization.

        objective="nll" (default) trains the shared backbone + density heads on
        the mixture NLL (the original MDN behaviour; point_head is excluded and
        trained separately).

        objective="mse" trains the shared backbone + point_head as a pure
        regression model on the weighted MSE, with the density heads frozen.
        """
        logger = setup_logger()

        valid_objective = {"nll", "mse"}
        if objective not in valid_objective:
            raise ValueError(
                f"objective must be one of {sorted(valid_objective)}, got {objective!r}"
            )

        valid_weighting = {"both", "sampler", "loss", "none"}
        if weighting not in valid_weighting:
            raise ValueError(
                f"weighting must be one of {sorted(valid_weighting)}, got {weighting!r}"
            )
        use_sampler = weighting in ("both", "sampler")
        use_loss_weights = weighting in ("both", "loss")
        logger.info(
            f"Weighting strategy: {weighting} "
            f"(sampler={use_sampler}, weighted_loss={use_loss_weights})"
        )

        # Normalize data
        logger.info("Normalizing data...")
        X_train_norm, y_train_norm = self.normalize_data(X_train, y_train)

        # Normalize validation data
        if X_val is not None and y_val is not None:
            X_val_norm, y_val_norm = self.renormalize_test(X_val, y_val)
        else:
            X_val_norm = None
            y_val_norm = None

        logger.info(
            f"X: center={self.scaler_x.center_.mean():.3f}, "
            f"scale={self.scaler_x.scale_.mean():.3f}"
        )
        logger.info(
            f"y: center={self.scaler_y.center_[0]:.3f}, "
            f"scale={self.scaler_y.scale_[0]:.3f}"
        )

        # Weighted sampler — square weights when sampler is the only weighting channel
        if use_sampler:
            sampler_weights = weights_train if use_loss_weights else weights_train**2
            sampler_training = WeightedRandomSampler(
                sampler_weights, num_samples=len(X_train_norm), replacement=True
            )
        else:
            sampler_training = None

        if use_loss_weights:
            w = weights_train.float()
            w2_mean = (w * w).mean()
            loss_w_divisor = (
                float(w2_mean / w.mean()) if use_sampler else float(w2_mean)
            )

        # Prepare dataloaders
        train_loader, val_loader = self.prepare_dataloaders(
            X_train=X_train_norm,
            y_train=y_train_norm,
            X_val=X_val_norm,
            y_val=y_val_norm,
            weights_train=weights_train,
            weights_val=weights_val,
            sampler=sampler_training,
            batch_size=batch_size,
        )

        val_w_divisor = None
        if val_loader is not None and (use_sampler or use_loss_weights):
            w_val = weights_val.detach().cpu().float()
            val_w_divisor = float((w_val * w_val).mean())
            logger.info(f"Validation weight divisor: {val_w_divisor:.3f}")

        density_head_prefixes = (
            "pi_layer",
            "mu_layer",
            "sigma_layer",
            "nu_layer",
            "lam_layer",
        )
        if objective == "mse":
            # Regression: train backbone + point_head; freeze the density heads.
            trainable = [
                p
                for name, p in self.named_parameters()
                if not name.startswith(density_head_prefixes)
            ]
        else:
            # NLL: train backbone + density heads; point_head is trained separately.
            trainable = [
                p
                for name, p in self.named_parameters()
                if not name.startswith("point_head")
            ]
        optimizer = optim.RAdam(
            trainable,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=1e-4,
            decoupled_weight_decay=True,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor_scheduler,
            patience=patience_scheduler,
        )

        # Training loop
        train_losses, val_losses, val_losses_unweighted = [], [], []

        grad_norms = []
        clip_events = 0
        total_steps = 0
        skipped_batches = 0

        best_val_loss = float("inf")
        best_val_loss_unweighted = float("nan")
        best_model_state = None
        epochs_without_improvement = 0

        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[fit:{objective}] optimizing {n_trainable}/{n_total} parameters "
            f"({100 * n_trainable / n_total:.1f}%)"
        )

        for epoch in range(max_epochs):
            self.train()

            train_loss = 0.0
            train_count = 0
            for X_batch, y_batch, weights_batch in train_loader:
                total_steps += 1

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                weights_batch = weights_batch.to(self.device)

                optimizer.zero_grad()

                if use_loss_weights:
                    base = weights_batch if use_sampler else weights_batch**2
                    loss_weights = base / loss_w_divisor
                else:
                    loss_weights = torch.ones_like(weights_batch)

                loss = self._objective_loss(X_batch, y_batch, loss_weights, objective)

                if not torch.isfinite(loss):
                    skipped_batches += 1
                    logger.warning(f"Skipping batch {total_steps}: non-finite loss")
                    continue

                loss.backward()

                clipped_norm = nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=max_norm
                )

                if not torch.isfinite(clipped_norm):
                    skipped_batches += 1
                    logger.warning(
                        f"Skipping batch {total_steps}: non-finite gradients"
                    )
                    optimizer.zero_grad()
                    continue

                grad_norms.append(float(clipped_norm))
                if clipped_norm > max_norm:
                    clip_events += 1

                optimizer.step()
                train_loss += loss.item() * len(X_batch)
                train_count += len(X_batch)

            train_loss /= max(train_count, 1)
            train_losses.append(train_loss)

            # Validation
            if val_loader is not None:
                self.eval()
                val_loss_w = 0.0
                val_loss_unw = 0.0

                with torch.no_grad():
                    for X_val_b, y_val_b, weights_val_b in val_loader:
                        X_val_b = X_val_b.to(self.device)
                        y_val_b = y_val_b.to(self.device)
                        weights_val_b = weights_val_b.to(self.device)

                        if val_w_divisor is None:
                            val_weights = torch.ones_like(weights_val_b)
                        else:
                            val_weights = weights_val_b**2 / val_w_divisor

                        loss_w = self._objective_loss(
                            X_val_b, y_val_b, val_weights, objective
                        )
                        val_loss_w += loss_w.item() * len(X_val_b)

                        ones = torch.ones(len(X_val_b), device=self.device)
                        loss_unw = self._objective_loss(
                            X_val_b, y_val_b, ones, objective
                        )
                        val_loss_unw += loss_unw.item() * len(X_val_b)

                val_loss_w /= len(val_loader.dataset)
                val_loss_unw /= len(val_loader.dataset)
                val_losses.append(val_loss_w)
                val_losses_unweighted.append(val_loss_unw)
                early_stop_loss = val_loss_w
                val_str = f" | Val Loss: {val_loss_w:.3f} ({val_loss_unw:.3f})"
            else:
                early_stop_loss = train_loss
                val_str = ""

            scheduler.step(early_stop_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch + 1:02d} | LR: {current_lr:.1e} | "
                f"Train Loss: {train_loss:.3f}{val_str}"
            )

            # Early stopping
            if early_stop_loss < best_val_loss:
                best_val_loss = early_stop_loss
                best_val_loss_unweighted = (
                    val_losses_unweighted[-1] if val_losses_unweighted else train_loss
                )
                best_model_state = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
                logger.info(f"- New best model saved (val_loss: {best_val_loss:.3f}) -")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                    logger.info(
                        f"Restored best model with val_loss: {best_val_loss:.3f}"
                    )
                break

        grad_norms_tensor = torch.tensor(grad_norms)
        logger.info(f"\n{'=' * 30}")
        logger.info("GRADIENT NORM DIAGNOSIS")
        logger.info(f"{'=' * 30}")
        logger.info(f"Total gradient updates: {total_steps}")
        logger.info(
            f"Skipped batches: {skipped_batches}/{total_steps} "
            f"({100 * skipped_batches / total_steps:.1f}%)"
        )
        logger.info(f"Average gradient norm: {grad_norms_tensor.mean():.3f}")
        logger.info(f"Median gradient norm: {grad_norms_tensor.median():.3f}")
        logger.info(f"Max gradient norm: {grad_norms_tensor.max():.3f}")
        logger.info(
            f"Gradient clipping events: {clip_events}/{total_steps} "
            f"({100 * clip_events / total_steps:.1f}%)"
        )
        logger.info(f"Clipping threshold: {max_norm}")

        self.last_fit_val_loss = {
            "weighted": float(best_val_loss),
            "unweighted": float(best_val_loss_unweighted),
        }

        denom = max(total_steps, 1)
        self.last_fit_diagnostics = {
            "total_steps": int(total_steps),
            "clip_events": int(clip_events),
            "skipped_batches": int(skipped_batches),
            "pct_clipped": 100.0 * clip_events / denom,
            "pct_skipped": 100.0 * skipped_batches / denom,
        }

        return best_val_loss

    def pred(self, X, grid):
        """Predict densities on ``grid``, on the original scale."""
        device = self.device
        dtype = torch.float32
        self.eval()

        with torch.no_grad():
            X = X.to(device)
            try:
                check_is_fitted(self.scaler_x)
                X_norm = torch.tensor(
                    self.scaler_x.transform(X.detach().cpu().numpy()),
                    dtype=torch.float32,
                    device=device,
                )
            except Exception:
                X_norm = X

            # Get normalized parameters (log_pi unused on the prediction path)
            pi, _, mu_norm, sigma_norm, nu, skew = self.forward(X_norm)

            # Denormalize parameters to original scale
            try:
                check_is_fitted(self.scaler_y)
                pi, mu, sigma, nu, skew = self.denormalize_params(
                    pi, mu_norm, sigma_norm, nu, skew
                )
            except Exception:
                mu, sigma = mu_norm, sigma_norm

            batch_size = X.shape[0]

            # Move to CPU for parallel processing
            pi_cpu = pi.detach().cpu()
            mu_cpu = mu.detach().cpu()
            sigma_cpu = sigma.detach().cpu()

            nu_cpu = nu.detach().cpu()
            lam_cpu = skew.detach().cpu()

            grid_t = torch.as_tensor(grid, dtype=dtype)
            density = torch.zeros(batch_size, grid_t.numel(), dtype=dtype)

            def _one_row(i):
                row = torch.zeros(grid_t.numel(), dtype=dtype)
                for k in range(self.n_mixtures):
                    comp_pdf = skewt_pdf(
                        grid_t,
                        mu_cpu[i, k],
                        sigma_cpu[i, k],
                        nu_cpu[i, k],
                        lam_cpu[i, k],
                    )
                    row += pi_cpu[i, k] * comp_pdf
                return i, row

            # Compute densities with optional parallelization
            if self.n_jobs == 1:
                for i in tqdm(range(batch_size), desc="Computing density", leave=False):
                    _, row = _one_row(i)
                    density[i] = row
            else:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_one_row)(i)
                    for i in tqdm(
                        range(batch_size), desc="Computing density", leave=False
                    )
                )
                for i, row in results:
                    density[i] = row

            return density.to(device)

    def pred_point(self, X):
        """Predict scalar point estimates with the trained point head."""
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            X_norm = torch.tensor(
                self.scaler_x.transform(X.detach().cpu().numpy()),
                dtype=torch.float32,
                device=self.device,
            )
            pred_norm = self._predict_point_norm(X_norm)
            mu_y = torch.tensor(
                self.scaler_y.center_[0], dtype=pred_norm.dtype, device=self.device
            )
            sigma_y = torch.tensor(
                self.scaler_y.scale_[0], dtype=pred_norm.dtype, device=self.device
            )
            return mu_y + sigma_y * pred_norm

    def fit_mse(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        weights_train,
        weights_val,
        max_epochs,
        learning_rate,
        batch_size,
        max_norm,
        patience,
        patience_scheduler,
        factor_scheduler,
        weighting="both",
    ):
        """Train the backbone and point_head as a pure regression model."""
        return self.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            weights_train=weights_train,
            weights_val=weights_val,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_norm=max_norm,
            patience=patience,
            patience_scheduler=patience_scheduler,
            factor_scheduler=factor_scheduler,
            weighting=weighting,
            objective="mse",
        )

    def fit_point_head(
        self,
        X_train,
        y_train,
        learning_rate,
        batch_size,
        max_norm,
        patience,
        patience_scheduler,
        factor_scheduler,
        max_epochs,
        loss_fn,
        X_val=None,
        y_val=None,
    ):
        """Freeze every layer except point_head and train it as a regression head."""
        logger = setup_logger()

        try:
            check_is_fitted(self.scaler_y)
        except Exception:
            raise RuntimeError(
                "Model must be fitted first via fit() before fit_point_head()."
            )

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        # Freeze everything except point_head
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith("point_head")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Training point_head only: {trainable} trainable parameters.")

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_train_norm, y_train_norm = self.renormalize_test(X_train, y_train)

        train_dataset = TensorDataset(
            X_train_norm.detach().cpu(), y_train_norm.detach().cpu()
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            X_val_norm, y_val_norm = self.renormalize_test(X_val, y_val)
            val_dataset = TensorDataset(
                X_val_norm.detach().cpu(), y_val_norm.detach().cpu()
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.RAdam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-6,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor_scheduler,
            patience=patience_scheduler,
        )

        best_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0

        grad_norms = []
        clip_events = 0
        total_steps = 0
        skipped_batches = 0

        for epoch in range(max_epochs):
            # The frozen backbone stays in eval mode
            self.eval()
            self.point_head.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                total_steps += 1
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Backbone frozen
                pred = self._predict_point_norm(X_batch)
                loss = loss_fn(pred, y_batch)

                if not torch.isfinite(loss):
                    skipped_batches += 1
                    logger.warning(
                        f"[PointHead] Skipping batch {total_steps}: non-finite loss"
                    )
                    continue

                optimizer.zero_grad()
                loss.backward()

                # clip_grad_norm_ returns the pre-clip total norm.
                clipped_norm = nn.utils.clip_grad_norm_(
                    self.point_head.parameters(), max_norm=max_norm
                )
                if not torch.isfinite(clipped_norm):
                    skipped_batches += 1
                    logger.warning(
                        f"[PointHead] Skipping batch {total_steps}: "
                        f"non-finite gradients"
                    )
                    optimizer.zero_grad()
                    continue
                grad_norms.append(float(clipped_norm))
                if clipped_norm > max_norm:
                    clip_events += 1

                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_loader.dataset)

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val_b, y_val_b in val_loader:
                        X_val_b = X_val_b.to(self.device)
                        y_val_b = y_val_b.to(self.device)
                        pred = self._predict_point_norm(X_val_b)
                        val_loss += loss_fn(pred, y_val_b).item() * len(X_val_b)
                val_loss /= len(val_loader.dataset)
                early_stop_loss = val_loss
                val_str = f" | Val Loss: {val_loss:.4f}"
            else:
                early_stop_loss = train_loss
                val_str = ""

            scheduler.step(early_stop_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"[PointHead] Epoch {epoch + 1:03d} | LR: {current_lr:.1e} | "
                f"Train Loss: {train_loss:.4f}{val_str}"
            )

            if early_stop_loss < best_loss:
                best_loss = early_stop_loss
                best_model_state = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
                logger.info(f"- New best model saved (loss: {best_loss:.4f}) -")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(
                    f"\n[PointHead] Early stopping triggered at epoch {epoch + 1}."
                )
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                    logger.info(f"Restored best model with loss: {best_loss:.4f}")
                break

        # Gradient-health diagnosis, mirroring fit().
        grad_norms_tensor = torch.tensor(grad_norms)
        denom = max(total_steps, 1)
        logger.info(f"\n{'=' * 30}")
        logger.info("POINT-HEAD GRADIENT NORM DIAGNOSIS")
        logger.info(f"{'=' * 30}")
        logger.info(f"Total gradient updates: {total_steps}")
        logger.info(
            f"Skipped batches: {skipped_batches}/{denom} "
            f"({100 * skipped_batches / denom:.1f}%)"
        )
        if grad_norms:
            logger.info(f"Average gradient norm: {grad_norms_tensor.mean():.3f}")
            logger.info(f"Median gradient norm: {grad_norms_tensor.median():.3f}")
            logger.info(f"Max gradient norm: {grad_norms_tensor.max():.3f}")
        logger.info(
            f"Gradient clipping events: {clip_events}/{denom} "
            f"({100 * clip_events / denom:.1f}%)"
        )
        logger.info(f"Clipping threshold: {max_norm}")

        self.last_point_head_diagnostics = {
            "total_steps": int(total_steps),
            "clip_events": int(clip_events),
            "skipped_batches": int(skipped_batches),
            "pct_clipped": 100.0 * clip_events / denom,
            "pct_skipped": 100.0 * skipped_batches / denom,
        }

        # Unfreeze all parameters
        for param in self.parameters():
            param.requires_grad = True


# Backbone builders + tuning/eval helpers (shared by Optuna tuning AND
# evaluation, so the model retrained for the test-set evaluation is
# architecturally identical to the tuned one; also reused by the point-head
# and conformal flows, which load/build the same backbones). `p` is a plain
# dict keyed by the Optuna parameter names.
def _resolve_family(density):
    """Map a requested density to (model_density, saturate_nu, saturate_skew).

    The gaussian / skewnorm / student ablation families are realized as a
    saturated skew-t (see SATURATED_FAMILIES); every other density is built
    natively with no saturation.
    """
    return SATURATED_FAMILIES.get(density, (density, False, False))


def build_lstm(p, device, n_jobs, density, point_head_layers):
    """Build an LSTM-backbone MDN from an Optuna-style param dict `p`."""
    hidden_layers = [p["lstm_width"]] * p["lstm_depth"]
    return MixtureDensityNetwork(
        input_dim=p["lags"],
        hidden_layers=hidden_layers,
        n_mixtures=p["n_mixtures"],
        point_head_layers=p.get("point_head_layers", point_head_layers),
        device=device,
        n_jobs=n_jobs,
        density=density,
        dropout=p["dropout"],
        point_head_augmented=p.get("ph_augmented", False),
    ).to(device)


def fit_and_score(
    mdn,
    data,
    lr,
    trial,
    *,
    max_epochs,
    batch_size,
    max_norm,
    patience,
    patience_scheduler,
    factor_scheduler,
    weighting,
    logger=None,
):
    """Train one model and return the validation NLL (Optuna objective value)."""
    logger = logger or setup_logger()
    try:
        val_nll = mdn.fit(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_cali"],
            y_val=data["y_cali"],
            weights_train=data["weights_train"],
            weights_val=data["weights_val"],
            max_epochs=max_epochs,
            learning_rate=lr,
            batch_size=batch_size,
            max_norm=max_norm,
            patience=patience,
            patience_scheduler=patience_scheduler,
            factor_scheduler=factor_scheduler,
            weighting=weighting,
        )
    except Exception as e:  # numerical blow-ups, invalid configs, etc.
        logger.warning(f"Trial {trial.number} failed: {e}")
        return float("inf")

    if not np.isfinite(val_nll):
        return float("inf")

    # Numerically unstable configs
    pct_skipped = float(
        getattr(mdn, "last_fit_diagnostics", {}).get("pct_skipped", 0.0)
    )
    trial.set_user_attr("pct_skipped", pct_skipped)
    if pct_skipped > MAX_PCT_SKIPPED:
        logger.warning(
            f"Trial {trial.number}: {pct_skipped:.1f}% batches skipped "
            f"(> {MAX_PCT_SKIPPED}%) -- marked infeasible."
        )
        return float("inf")

    n_params = sum(
        p.numel()
        for name, p in mdn.named_parameters()
        if not name.startswith("point_head")
    )
    trial.set_user_attr("n_density_params", n_params)
    return val_nll


def recalibrate_test(mdn, data, grid_y, device, n_jobs, artifacts=None):
    """Recalibrate on the calibration split; return (raw, recalibrated) densities."""
    return recalibrate_test_density(
        lambda X: mdn.pred(X, grid_y).cpu().numpy(),
        data,
        grid_y,
        device,
        n_jobs,
        artifacts=artifacts,
    )
