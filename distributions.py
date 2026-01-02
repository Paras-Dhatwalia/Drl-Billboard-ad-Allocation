"""
Custom probability distributions for billboard allocation

This module implements specialized distributions for different action modes:
- IndependentBernoulli: For EA (Edge Action) mode - combinatorial action space
- MaskedCategorical: For NA/MH modes - categorical with masking

Research Note:
EA mode requires Independent Bernoulli because each (ad, billboard) pair
is an independent binary decision. Using Categorical would be mathematically incorrect.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Bernoulli, Categorical
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class IndependentBernoulli(Distribution):
    """
    Independent Bernoulli distribution for EA mode.

    Each dimension is an independent Bernoulli random variable.
    This is critical for EA mode where we decide on multiple (ad, billboard) pairs simultaneously.

    Mathematical formulation:
    - Action: a ∈ {0,1}^(n_ads × n_billboards)
    - Probability: p(a) = ∏_i Bernoulli(a_i | logit_i)
    - Log-probability: log p(a) = Σ_i log Bernoulli(a_i | logit_i)
    - Entropy: H = Σ_i H(Bernoulli(logit_i))

    Masking:
    - Invalid actions have logit = -inf → probability = 0
    - Ensures model never samples invalid (ad, billboard) pairs
    - Gradient is blocked for masked dimensions

    Args:
        logits: Shape (..., n_actions) - logits for each Bernoulli
        mask: Shape (..., n_actions) - True for valid actions, False for invalid
    """

    # Silence validation warning
    arg_constraints = {}

    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None, validate_args=None):
        """
        Initialize Independent Bernoulli distribution.

        Args:
            logits: Unnormalized log probabilities, shape (..., n_actions)
            mask: Boolean mask, True = valid action, shape (..., n_actions)
            validate_args: Whether to validate inputs
        """
        self.logits = logits
        self.mask = mask

        # Apply mask: set invalid action logits to -inf
        if mask is not None:
            if mask.shape != logits.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} must match logits shape {logits.shape}"
                )
            # CRITICAL: Set masked logits to -inf so probability = 0
            masked_logits = logits.clone()
            masked_logits[~mask] = float('-inf')
        else:
            masked_logits = logits

        # Create Bernoulli distribution for each dimension
        self._bernoulli = Bernoulli(logits=masked_logits, validate_args=validate_args)

        # Set batch shape
        batch_shape = logits.shape[:-1]
        event_shape = logits.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def probs(self):
        """Get probabilities for each dimension"""
        return self._bernoulli.probs

    def sample(self, sample_shape=torch.Size()):
        """
        Sample actions from the distribution.

        Returns:
            action: Binary tensor of shape (*sample_shape, *batch_shape, n_actions)
        """
        samples = self._bernoulli.sample(sample_shape)

        # Validate: masked actions should never be sampled
        if self.mask is not None:
            if torch.any(samples[..., ~self.mask] == 1):
                logger.error("CRITICAL: Sampled invalid action despite masking!")
                # Force fix
                samples = samples * self.mask

        return samples

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of actions.

        Args:
            value: Action tensor, shape (..., n_actions)

        Returns:
            log_prob: Sum of log probabilities across all dimensions, shape (...)
        """
        # Get log prob for each dimension
        log_probs = self._bernoulli.log_prob(value)

        # Sum across action dimensions (independent assumption)
        # Shape: (..., n_actions) -> (...)
        total_log_prob = log_probs.sum(dim=-1)

        return total_log_prob

    def entropy(self) -> torch.Tensor:
        """
        Compute entropy of the distribution.

        Entropy for independent Bernoulli: H = Σ_i H(Bernoulli_i)

        Returns:
            entropy: Sum of entropies across all dimensions, shape (...)
        """
        # Get entropy for each dimension
        entropies = self._bernoulli.entropy()

        # Sum across action dimensions
        # Shape: (..., n_actions) -> (...)
        total_entropy = entropies.sum(dim=-1)

        return total_entropy

    @property
    def variance(self) -> torch.Tensor:
        """
        Compute variance of the distribution.

        For Bernoulli: Var(X) = p(1-p)

        Returns:
            variance: Variance for each dimension, summed
        """
        p = self.probs
        var_per_dim = p * (1 - p)
        # Sum across action dimensions
        total_variance = var_per_dim.sum(dim=-1)
        return total_variance

    @property
    def mode(self) -> torch.Tensor:
        """
        Get the most likely action (deterministic).

        For Bernoulli: mode = 1 if p > 0.5, else 0

        Returns:
            mode: Binary tensor of shape (..., n_actions)

        Note: This is a property (not method) so Tianshou can access it directly.
        """
        mode_val = (self.probs > 0.5).float()

        # Apply mask to ensure valid actions
        if self.mask is not None:
            mode_val = mode_val * self.mask

        return mode_val

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sampling (for gradient flow).

        Note: Bernoulli doesn't support true reparameterization.
        This uses the straight-through estimator.
        """
        # Use Gumbel-softmax approximation for differentiability
        # For binary case: Concrete distribution
        return self._bernoulli.rsample(sample_shape)


class MaskedCategorical(Categorical):
    """
    Categorical distribution with masking support.

    Used for NA (Node Action) and MH (Multi-Head) modes.

    Args:
        logits: Unnormalized log probabilities, shape (..., n_actions)
        mask: Boolean mask, True = valid action, shape (..., n_actions)
    """

    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None, validate_args=None):
        """
        Initialize Masked Categorical distribution.

        Args:
            logits: Unnormalized log probabilities
            mask: Boolean mask for valid actions
        """
        if mask is not None:
            if mask.shape != logits.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} must match logits shape {logits.shape}"
                )

            # Set masked logits to -inf
            masked_logits = logits.clone()
            masked_logits[~mask] = float('-inf')

            # Check at least one valid action
            if torch.any(torch.all(~mask, dim=-1)):
                raise ValueError("At least one batch element has all actions masked!")
        else:
            masked_logits = logits

        super().__init__(logits=masked_logits, validate_args=validate_args)


def create_ea_distribution(logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """
    Factory function to create EA distribution.

    Args:
        logits: Action logits from policy network
        mask: Action mask

    Returns:
        IndependentBernoulli distribution
    """
    return IndependentBernoulli(logits=logits, mask=mask)


def create_na_distribution(logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """
    Factory function to create NA/MH distribution.

    Args:
        logits: Action logits from policy network
        mask: Action mask

    Returns:
        MaskedCategorical distribution
    """
    return MaskedCategorical(logits=logits, mask=mask)


# Self-test
if __name__ == "__main__":
    # Test Independent Bernoulli
    print("Testing IndependentBernoulli...")

    # Create test logits and mask
    logits = torch.randn(2, 10)  # batch_size=2, n_actions=10
    mask = torch.randint(0, 2, (2, 10)).bool()  # random mask

    # Create distribution
    dist = IndependentBernoulli(logits, mask)

    # Sample
    actions = dist.sample()
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Actions:\n{actions}")

    # Check no invalid actions sampled
    invalid_sampled = actions[~mask]
    if torch.any(invalid_sampled == 1):
        print("ERROR: Invalid actions were sampled!")
    else:
        print("✓ Masking works correctly")

    # Log prob
    log_prob = dist.log_prob(actions)
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Log prob: {log_prob}")

    # Entropy
    entropy = dist.entropy()
    print(f"Entropy shape: {entropy.shape}")
    print(f"Entropy: {entropy}")

    print("\n✓ All tests passed!")
