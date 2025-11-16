"""Loss functions for StyleTTS2 training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """Reconstruction loss (L1 or L2) for mel/STFT."""

    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            predicted: Predicted mel/STFT [batch, n_mels, time]
            target: Target mel/STFT [batch, n_mels, time]

        Returns:
            Scalar loss
        """
        if self.loss_type == "l1":
            return F.l1_loss(predicted, target)
        elif self.loss_type == "l2":
            return F.mse_loss(predicted, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class KLDivergenceLoss(nn.Module):
    """KL divergence loss for style encoder latent."""

    def forward(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence with standard normal.

        KL(q || p) where:
            q = N(mean, exp(log_var))
            p = N(0, 1)

        Args:
            mean: Mean of latent distribution [batch, dim]
            log_var: Log variance of latent [batch, dim]

        Returns:
            Scalar KL divergence
        """
        # KL(q || p) = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # Normalize by batch size and dimensions
        kl = kl / (mean.size(0) * mean.size(1))
        return kl


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training."""

    def __init__(self, loss_type: str = "hinge"):
        super().__init__()
        self.loss_type = loss_type

    def forward_discriminator(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Discriminator loss.

        Args:
            real_logits: Discriminator output for real samples
            fake_logits: Discriminator output for fake samples

        Returns:
            Scalar discriminator loss
        """
        if self.loss_type == "hinge":
            # Hinge loss
            loss_real = torch.mean(F.relu(1.0 - real_logits))
            loss_fake = torch.mean(F.relu(1.0 + fake_logits))
            return loss_real + loss_fake
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward_generator(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Generator loss.

        Args:
            fake_logits: Discriminator output for generated samples

        Returns:
            Scalar generator loss
        """
        if self.loss_type == "hinge":
            # Generator wants discriminator to output high values
            return -torch.mean(fake_logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class DurationLoss(nn.Module):
    """Duration prediction loss (MSE in log scale)."""

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute duration loss.

        Args:
            predicted: Predicted log durations [batch, seq_len]
            target: Target durations [batch, seq_len]
            lengths: Actual sequence lengths [batch]

        Returns:
            Scalar loss
        """
        # Convert target to log scale
        target_log = torch.log(target.clamp(min=1.0))

        # Create mask for valid positions
        batch_size, max_len = predicted.shape
        mask = torch.arange(max_len, device=predicted.device)[None, :] < lengths[:, None]

        # MSE loss only on valid positions
        loss = F.mse_loss(predicted * mask, target_log * mask, reduction='sum')
        loss = loss / mask.sum()  # Normalize by number of valid elements

        return loss
