import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import time
import random
import math
import csv
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
from typing import Optional

from mi_sampler import ConditionalFlowSampler
from mi_trainer import MITrainer, ActionDiscriminator

__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def collect_episode(env, agent: TDMPC, cfg, step: int, eval_mode: bool = False) -> Episode:
	obs = env.reset()
	episode = Episode(cfg, obs)
	while not episode.done:
		action = agent.plan(obs, eval_mode=eval_mode, step=step, t0=episode.first)
		obs, reward, done, _ = env.step(action.cpu().numpy())
		episode += (obs, action, reward, done)
	return episode


def _mi_output_dir(cfg) -> Path:
	base = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed) / 'mi_sampler'
	custom = cfg.get('mi_sampler_out_dir', '')
	return Path(custom) if custom else base


def _is_better(value: float, best: Optional[float], mode: str) -> bool:
	if best is None:
		return True
	if mode == 'max':
		return value > best
	if mode == 'min':
		return value < best
	raise ValueError(f'Unknown checkpoint mode: {mode}')


def _save_mi_weights(sampler: ConditionalFlowSampler, discriminator: ActionDiscriminator, output_dir: Path, tag: str):
	sampler_path = output_dir / f'sampler_{tag}.pt'
	discriminator_path = output_dir / f'discriminator_{tag}.pt'
	torch.save(sampler.state_dict(), sampler_path)
	torch.save(discriminator.state_dict(), discriminator_path)
	return sampler_path, discriminator_path


def _rollout_latent(agent: TDMPC, z0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
	"""Rollout latent dynamics for trajectories shaped [N,H,action_dim]."""
	with torch.no_grad():
		z = z0
		zs = []
		for t in range(u_seq.shape[1]):
			z, _ = agent.model.next(z, u_seq[:, t])
			zs.append(z)
		return torch.stack(zs, dim=1)


def _diag_gaussian_entropy(x: torch.Tensor, eps: float) -> float:
	"""Stable batch entropy proxy assuming diagonal covariance."""
	var = x.var(dim=0, unbiased=False)
	const = 2.0 * math.pi * math.e
	ent = 0.5 * torch.log(const * var + eps).sum()
	return float(ent.item())


def _pairwise_spread(x: torch.Tensor) -> float:
	if x.shape[0] < 2:
		return 0.0
	dist = torch.pdist(x, p=2)
	if dist.numel() == 0:
		return 0.0
	return float(dist.mean().item())


def _sample_baseline_actions(cfg, num_trajs: int, device: torch.device) -> Optional[torch.Tensor]:
	mode = str(cfg.get('mi_report_baseline', 'gaussian_clip')).lower()
	shape = (num_trajs, cfg.horizon, cfg.action_dim)
	if mode == 'none':
		return None
	if mode == 'uniform':
		return torch.empty(shape, device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
	if mode in {'gaussian', 'gaussian_clip'}:
		std = float(cfg.get('mi_report_gaussian_std', 2.0))
		u = std * torch.randn(shape, device=device, dtype=torch.float32)
		return torch.clamp(u, -1.0, 1.0)
	raise ValueError(f'Unknown mi_report_baseline: {mode}')


def compute_coverage_report(cfg, agent: TDMPC, sampler: ConditionalFlowSampler, replay_batch):
	"""
	Compute coverage metrics for sampler outputs in latent trajectory space.
	Returns metrics for NF samples and optional gain over baseline samples.
	"""
	obs = replay_batch[0]
	with torch.no_grad():
		z0 = agent.model.h(obs).detach()
		num_states = min(int(cfg.mi_report_batch_size), z0.shape[0])
		num_samples = int(cfg.mi_report_samples_per_state)
		eps = float(cfg.mi_report_entropy_eps)
		z0 = z0[:num_states]
		u_nf, _ = sampler.sample_no_grad(z0, num_samples=num_samples)
		u_nf = u_nf.reshape(-1, cfg.horizon, cfg.action_dim)
		z0_rep = z0.repeat_interleave(num_samples, dim=0)
		z_nf = _rollout_latent(agent, z0_rep, u_nf)
		x_nf = z_nf.reshape(z_nf.shape[0], -1)
		report = {
			'coverage_nf_entropy': _diag_gaussian_entropy(x_nf, eps),
			'coverage_nf_pairwise_spread': _pairwise_spread(x_nf),
			'coverage_nf_action_spread': _pairwise_spread(u_nf.reshape(u_nf.shape[0], -1)),
		}

		u_base = _sample_baseline_actions(cfg, num_trajs=u_nf.shape[0], device=z0.device)
		if u_base is None:
			return report
		z_base = _rollout_latent(agent, z0_rep, u_base)
		x_base = z_base.reshape(z_base.shape[0], -1)
		base_entropy = _diag_gaussian_entropy(x_base, eps)
		base_spread = _pairwise_spread(x_base)
		base_action_spread = _pairwise_spread(u_base.reshape(u_base.shape[0], -1))
		report.update({
			'coverage_base_entropy': base_entropy,
			'coverage_base_pairwise_spread': base_spread,
			'coverage_base_action_spread': base_action_spread,
			'coverage_entropy_gain': report['coverage_nf_entropy'] - base_entropy,
			'coverage_pairwise_spread_gain': report['coverage_nf_pairwise_spread'] - base_spread,
			'coverage_action_spread_gain': report['coverage_nf_action_spread'] - base_action_spread,
		})
		return report


def train_mi(cfg):
	assert torch.cuda.is_available(), 'MI training requires CUDA in this TD-MPC setup.'
	set_seed(cfg.seed)
	cfg.use_mi_warmstart = bool(cfg.get('mi_collect_use_warmstart', False))
	cfg.mi_model_path = str(cfg.get('mi_collect_mi_model_path', ''))

	env = make_env(cfg)
	agent = TDMPC(cfg)
	buffer = ReplayBuffer(cfg)

	tdmpc_model_path = cfg.get('mi_tdmpc_model_path', '')
	if tdmpc_model_path:
		agent.load(tdmpc_model_path)
		print(f'[MI] Loaded TD-MPC weights: {tdmpc_model_path}')
	else:
		print('[MI] Warning: mi_tdmpc_model_path is empty. Training MI on untrained latent model may be ineffective.')

	sampler = ConditionalFlowSampler(
		z_dim=cfg.latent_dim,
		action_dim=cfg.action_dim,
		horizon=cfg.horizon,
		ctx_dim=cfg.mi_ctx_dim,
		hidden_dim=cfg.mi_sampler_hidden_dim,
		num_layers=cfg.mi_sampler_layers,
		zero_init_coupling=bool(cfg.get('mi_sampler_zero_init', False)),
		device=cfg.device,
	)
	discriminator = ActionDiscriminator(
		x_dim=cfg.horizon * cfg.latent_dim,
		u_dim=cfg.horizon * cfg.action_dim,
		hidden_dim=cfg.mi_discriminator_hidden_dim,
	)
	mi_trainer = MITrainer(
		generator=sampler,
		discriminator=discriminator,
		horizon=cfg.horizon,
		action_dim=cfg.action_dim,
		alpha=cfg.mi_alpha,
		num_samples=cfg.mi_train_num_samples,
		smooth_coef=cfg.mi_smooth_coef,
		energy_coef=cfg.mi_energy_coef,
		lr=cfg.mi_train_lr,
		weight_decay=cfg.mi_train_wd,
		max_grad_norm=cfg.mi_train_grad_clip,
		device=cfg.device,
	)

	output_dir = _mi_output_dir(cfg)
	output_dir.mkdir(parents=True, exist_ok=True)
	coverage_log_path = output_dir / 'coverage.log'
	checkpoint_log_path = output_dir / 'checkpoints.log'
	ckpt_metric = str(cfg.get('mi_ckpt_metric', 'coverage_pairwise_spread_gain'))
	ckpt_mode = str(cfg.get('mi_ckpt_mode', 'max')).lower()
	best_ckpt_value = None
	best_ckpt_episode = -1

	start_time = time.time()
	global_step = 0
	last_metrics = {}
	last_coverage = {}
	coverage_writer = None
	checkpoint_writer = None
	with open(coverage_log_path, 'w', newline='', encoding='utf-8') as cov_file, \
		 open(checkpoint_log_path, 'w', newline='', encoding='utf-8') as ckpt_file:
		for episode_idx in range(cfg.mi_train_episodes):
			episode = collect_episode(
				env=env,
				agent=agent,
				cfg=cfg,
				step=global_step,
				eval_mode=bool(cfg.mi_collect_eval_mode),
			)
			global_step += cfg.episode_length
			buffer += episode

			# Optional TD-MPC side-updates while collecting MI data.
			for i in range(int(cfg.mi_joint_tdmpc_updates)):
				agent.update(buffer, global_step + i)

			if episode_idx + 1 >= cfg.mi_warmup_episodes:
				for _ in range(int(cfg.mi_updates_per_episode)):
					batch = buffer.sample()
					last_metrics = mi_trainer.train_step(batch, agent.model.h, agent.model)

			if (episode_idx + 1) >= int(cfg.mi_warmup_episodes) and (episode_idx + 1) % int(cfg.mi_report_every) == 0:
				report_batch = buffer.sample()
				last_coverage = compute_coverage_report(cfg, agent, sampler, report_batch)
				row = {
					'episode': int(episode_idx + 1),
					'global_step': int(global_step),
					**{k: float(v) for k, v in last_coverage.items()},
				}
				if coverage_writer is None:
					coverage_writer = csv.DictWriter(cov_file, fieldnames=list(row.keys()))
					coverage_writer.writeheader()
				coverage_writer.writerow(row)
				cov_file.flush()

				metric_value = last_coverage.get(ckpt_metric, None)
				if metric_value is not None and bool(cfg.get('mi_save_best_ckpt', True)) and _is_better(metric_value, best_ckpt_value, ckpt_mode):
					best_ckpt_value = float(metric_value)
					best_ckpt_episode = int(episode_idx + 1)
					sampler_best_path, disc_best_path = _save_mi_weights(
						sampler=sampler,
						discriminator=discriminator,
						output_dir=output_dir,
						tag='best',
					)
					print(
						f"[MI] best checkpoint updated: episode={best_ckpt_episode} "
						f"{ckpt_metric}={best_ckpt_value:.6f} sampler={sampler_best_path.name}"
					)
					ckpt_row = {
						'episode': best_ckpt_episode,
						'global_step': int(global_step),
						'tag': 'best',
						'metric_name': ckpt_metric,
						'metric_value': float(best_ckpt_value),
						'sampler_path': str(sampler_best_path),
						'discriminator_path': str(disc_best_path),
					}
					if checkpoint_writer is None:
						checkpoint_writer = csv.DictWriter(ckpt_file, fieldnames=list(ckpt_row.keys()))
						checkpoint_writer.writeheader()
					checkpoint_writer.writerow(ckpt_row)
					ckpt_file.flush()

				save_periodic = bool(cfg.get('mi_save_periodic_ckpt', False))
				periodic_every = int(cfg.get('mi_periodic_ckpt_every', 0))
				episode_num = int(episode_idx + 1)
				if save_periodic and periodic_every > 0 and (episode_num % periodic_every == 0):
					tag = f'ep{episode_num:04d}'
					sampler_ep_path, disc_ep_path = _save_mi_weights(
						sampler=sampler,
						discriminator=discriminator,
						output_dir=output_dir,
						tag=tag,
					)
					ckpt_row = {
						'episode': episode_num,
						'global_step': int(global_step),
						'tag': tag,
						'metric_name': ckpt_metric,
						'metric_value': float(metric_value) if metric_value is not None else float('nan'),
						'sampler_path': str(sampler_ep_path),
						'discriminator_path': str(disc_ep_path),
					}
					if checkpoint_writer is None:
						checkpoint_writer = csv.DictWriter(ckpt_file, fieldnames=list(ckpt_row.keys()))
						checkpoint_writer.writeheader()
					checkpoint_writer.writerow(ckpt_row)
					ckpt_file.flush()

			if (episode_idx + 1) % int(cfg.mi_log_every) == 0:
				msg = (
					f"[MI] episode={episode_idx+1}/{cfg.mi_train_episodes} "
					f"reward={episode.cumulative_reward:.1f} "
					f"time={time.time()-start_time:.1f}s"
				)
				if last_metrics:
					msg += (
						f" mi_loss={last_metrics.get('mi_loss', float('nan')):.4f}"
						f" mi_info={last_metrics.get('mi_info_loss', float('nan')):.4f}"
						f" mi_ent={last_metrics.get('mi_ent_loss', float('nan')):.4f}"
						f" gen_gn={last_metrics.get('mi_gen_grad_norm', float('nan')):.3f}"
						f" disc_gn={last_metrics.get('mi_disc_grad_norm', float('nan')):.3f}"
					)
				if last_coverage:
					msg += (
						f" cov_nf_ent={last_coverage.get('coverage_nf_entropy', float('nan')):.3f}"
						f" cov_nf_spread={last_coverage.get('coverage_nf_pairwise_spread', float('nan')):.3f}"
						f" cov_gain={last_coverage.get('coverage_pairwise_spread_gain', float('nan')):.3f}"
					)
				print(msg)

	sampler_path = output_dir / 'sampler.pt'
	discriminator_path = output_dir / 'discriminator.pt'
	meta_path = output_dir / 'meta.yaml'
	torch.save(sampler.state_dict(), sampler_path)
	torch.save(discriminator.state_dict(), discriminator_path)
	with open(meta_path, 'w', encoding='utf-8') as f:
		f.write(OmegaConf.to_yaml(cfg))

	print(f'[MI] Saved sampler weights to: {sampler_path}')
	print(f'[MI] Saved discriminator weights to: {discriminator_path}')
	print(f'[MI] Saved run config to: {meta_path}')
	print(f'[MI] Saved coverage report to: {coverage_log_path}')
	print(f'[MI] Saved checkpoint report to: {checkpoint_log_path}')
	if best_ckpt_value is not None:
		print(
			f"[MI] Best checkpoint: episode={best_ckpt_episode} "
			f"{ckpt_metric}={best_ckpt_value:.6f} "
			f"path={output_dir / 'sampler_best.pt'}"
		)


if __name__ == '__main__':
	train_mi(parse_cfg(Path().cwd() / __CONFIG__))
