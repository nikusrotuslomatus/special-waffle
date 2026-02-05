from typing import Optional, Tuple

import torch
import torch.nn as nn


class ActionDiscriminator(nn.Module):
	"""Gaussian discriminator q_phi(U|X) with a simple MLP head."""
	def __init__(self, x_dim: int, u_dim: int, hidden_dim: int = 256,
				 log_std_min: float = -10.0, log_std_max: float = 2.0):
		super().__init__()
		self._net = nn.Sequential(
			nn.Linear(x_dim, hidden_dim), nn.ELU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
		)
		self._mean = nn.Linear(hidden_dim, u_dim)
		self._log_std = nn.Linear(hidden_dim, u_dim)
		self._log_std_min = log_std_min
		self._log_std_max = log_std_max

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self._net(x)
		mean = self._mean(h)
		log_std = self._log_std(h).clamp(self._log_std_min, self._log_std_max)
		return mean, log_std

	def log_prob(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
		mean, log_std = self(x)
		std = torch.exp(log_std)
		dist = torch.distributions.Normal(mean, std)
		return dist.log_prob(u).sum(dim=-1)


class MITrainer:
	"""
	Side-trainer for MI-based sampler. Uses stop-grad latent rollouts.
	"""
	def __init__(
		self,
		generator: nn.Module,
		discriminator: nn.Module,
		horizon: int,
		action_dim: int,
		alpha: float = 1.0,
		num_samples: int = 1,
		smooth_coef: float = 0.0,
		energy_coef: float = 0.0,
		lr: float = 1e-4,
		weight_decay: float = 0.0,
		max_grad_norm: Optional[float] = None,
		device: str = "cuda",
	):
		self.generator = generator.to(device)
		self.discriminator = discriminator.to(device)
		self.horizon = horizon
		self.action_dim = action_dim
		self.alpha = alpha
		self.num_samples = num_samples
		self.smooth_coef = smooth_coef
		self.energy_coef = energy_coef
		self.max_grad_norm = max_grad_norm
		self.device = torch.device(device)

		params = list(self.generator.parameters()) + list(self.discriminator.parameters())
		self.optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

	def _rollout_latent(self, z0: torch.Tensor, u_seq: torch.Tensor, dyn_model) -> torch.Tensor:
		"""Rollout latent dynamics with stop-grad. Returns [B,H,latent_dim]."""
		with torch.no_grad():
			z = z0
			zs = []
			for t in range(self.horizon):
				z, _ = dyn_model.next(z, u_seq[:, t])
				zs.append(z)
			return torch.stack(zs, dim=1)

	def _traj_features(self, z_traj: torch.Tensor) -> torch.Tensor:
		"""Flatten latent trajectory features."""
		return z_traj.reshape(z_traj.shape[0], -1)

	def train_step(self, replay_batch, encoder, dyn_model, ctx: Optional[torch.Tensor] = None):
		"""
		One MI update step.
		replay_batch: tuple returned by ReplayBuffer.sample()
		encoder: TOLD encoder (obs -> z0)
		dyn_model: TOLD model (uses .next)
		"""
		obs = replay_batch[0] if isinstance(replay_batch, (tuple, list)) else replay_batch['obs']
		z0 = encoder(obs).detach()

		self.generator.train()
		self.discriminator.train()
		self.optim.zero_grad(set_to_none=True)

		# Sample actions from generator (no grad needed for sampling itself)
		u_samples, _ = self.generator.sample(z0, ctx=ctx, num_samples=self.num_samples)
		u_samples = u_samples.reshape(-1, self.horizon, self.action_dim)
		z0_rep = z0.repeat_interleave(self.num_samples, dim=0)
		ctx_rep = None if ctx is None else ctx.repeat_interleave(self.num_samples, dim=0)

		# Latent rollout (stop-grad)
		z_traj = self._rollout_latent(z0_rep, u_samples, dyn_model)
		x = self._traj_features(z_traj)

		u_flat = u_samples.reshape(u_samples.shape[0], -1)
		log_q = self.discriminator.log_prob(x, u_flat)
		log_pi = self.generator.log_prob(u_samples, z0_rep, ctx=ctx_rep)

		loss_info = -log_q.mean()
		loss_ent = log_pi.mean()
		loss = loss_info + self.alpha * loss_ent

		smooth_loss = None
		if self.smooth_coef > 0.0 and self.horizon > 1:
			diff = u_samples[:, 1:] - u_samples[:, :-1]
			smooth_loss = (diff ** 2).sum(dim=-1).mean()
			loss = loss + self.smooth_coef * smooth_loss

		energy_loss = None
		if self.energy_coef > 0.0:
			energy_loss = (u_samples ** 2).sum(dim=-1).mean()
			loss = loss + self.energy_coef * energy_loss

		loss.backward()
		if self.max_grad_norm is not None:
			torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm, error_if_nonfinite=False)
			torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm, error_if_nonfinite=False)
		self.optim.step()

		metrics = {
			'mi_loss': float(loss.item()),
			'mi_info_loss': float(loss_info.item()),
			'mi_ent_loss': float(loss_ent.item()),
		}
		if smooth_loss is not None:
			metrics['mi_smooth_loss'] = float(smooth_loss.item())
		if energy_loss is not None:
			metrics['mi_energy_loss'] = float(energy_loss.item())
		return metrics
