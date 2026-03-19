import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # breakpoint()
        # # >> obs.shape — what did the caller pass in? (check utils.py: ob[None, :])
        # #    The [None, :] adds a batch dimension: (ob_dim,) -> (1, ob_dim). (1, 4) for CartPole.
        # # >> type(obs)  — it's numpy here, but forward() needs torch
        # # >> After you get the distribution and sample:
        # #    action.shape — for discrete, should be (1,) or scalar; for continuous, (1, ac_dim)
        # #    Think: does env.step() expect a scalar int or a numpy array?
        # TODO: implement get_action
        obs = ptu.from_numpy(obs)
        distributions = self.forward(obs)
        action = distributions.sample()
        action = action.squeeze(0)  # remove batch dim only: (1,) -> scalar for discrete, (1, ac_dim) -> (ac_dim,) for continuous
        return action.cpu().numpy()

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            # breakpoint()
            # # >> obs.shape          — should be (batch_size, ob_dim)
            # # >> self.logits_net    — look at the architecture: how many layers? what's the output size?
            # # >> After you compute logits: logits.shape — should be (batch_size, ac_dim)
            # #    For CartPole: ac_dim=2 (left or right)
            # # >> The distribution you return — try calling .sample() and .log_prob() on it at the breakpoint
            logits = self.logits_net(obs)  # (batch_size, ac_dim)
            distributions = D.Categorical(logits=logits)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            # breakpoint()
            # # >> obs.shape              — (batch_size, ob_dim)
            # # >> mean.shape after mean_net — should be (batch_size, ac_dim)
            # # >> self.logstd.shape      — (ac_dim,) — notice it's NOT per-observation, it's shared
            # # >> torch.exp(self.logstd) — these are the actual standard deviations. Are they reasonable?
            mean = self.mean_net(obs)  # (batch_size, ac_dim)
            std = torch.exp(self.logstd)  # (ac_dim,)
            distributions = D.Normal(mean, std)
        return distributions

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        
        # breakpoint()
        # # >> obs.shape        — (batch_size, ob_dim), e.g. (1020, 4) for CartPole
        # # >> actions.shape    — (batch_size,) for discrete e.g. (1020,) for CartPole, (batch_size, ac_dim) for continuous
        # # >> advantages.shape — (batch_size,)
        # # >> After computing distribution and log_probs:
        # #    log_probs.shape  — should match actions.shape. If continuous with multi-dim actions,
        # #                        you may get (batch_size, ac_dim) — do you need to sum across ac_dim?
        # # >> advantages * log_probs — check this broadcasts correctly (shapes must align)

        # TODO: compute the policy gradient actor loss
        distributions = self.forward(obs)
        log_probs = distributions.log_prob(actions)
        if not self.discrete:
            log_probs = log_probs.sum(dim=-1)  # if continuous with multi-dim actions, sum log probs across action dimensions
        loss = -torch.mean(log_probs * advantages)

        # TODO: perform an optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }
