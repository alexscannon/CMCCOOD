from typing import Tuple
import torch

class EnergyBasedOODDetector():
    """Energy-based out-of-distribution detection."""

    def __init__(self, temperature=1.0, threshold=None):
        self.temperature = temperature
        self.threshold = threshold
        self.energy_stats = {
            'mean': None,
            'std': None
        }
        # Higher energy means more likely to be OOD, so no inversion needed
        self.score_needs_inversion = False

    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute energy score from logits."""
        # Energy = -T * log(sum(exp(logits/T)))
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        return energy

    def calibrate(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
        """Calibrate the detector on in-distribution data."""
        model.eval()
        energies = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                logits = model(inputs)
                energy = self.compute_energy(logits)
                energies.append(energy)

        # Concatenate all energies
        all_energies = torch.cat(energies, dim=0)

        # Compute statistics
        self.energy_stats['mean'] = all_energies.mean().item()
        self.energy_stats['std'] = all_energies.std().item()

        # Set threshold based on statistics (e.g., mean + 2*std)
        if self.threshold is None:
            self.threshold = self.energy_stats['mean'] + 2 * self.energy_stats['std']

    def predict(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict if samples are OOD based on energy score."""
        energy_scores = self.compute_energy(logits)
        is_ood_preds = energy_scores > self.threshold
        return is_ood_preds, energy_scores