import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import csv
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC

__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def _parse_int_list(value: str) -> List[int]:
	items = [x.strip() for x in str(value).split(',') if x.strip()]
	return [int(x) for x in items]


def _parse_path_list(value: str) -> List[str]:
	items = [x.strip() for x in str(value).split(',') if x.strip()]
	return items


def _discover_sampler_paths(cfg) -> List[str]:
	paths = _parse_path_list(cfg.get('ab_sampler_paths', ''))
	sampler_dir = str(cfg.get('ab_sampler_dir', ''))
	if not paths and sampler_dir:
		pattern = str(cfg.get('ab_sampler_glob', 'sampler*.pt'))
		paths = [str(p) for p in sorted(Path(sampler_dir).glob(pattern))]
	# Deduplicate while preserving order.
	seen = set()
	result = []
	for p in paths:
		if p not in seen:
			seen.add(p)
			result.append(p)
	return result


def _file_sha256(path: str) -> str:
	h = hashlib.sha256()
	with open(path, 'rb') as f:
		while True:
			chunk = f.read(1024 * 1024)
			if not chunk:
				break
			h.update(chunk)
	return h.hexdigest()


def _state_signature(path: str) -> str:
	"""Hash tensor contents in checkpoint state for robust equality checks."""
	state = torch.load(path, map_location='cpu')
	if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
		state = state['state_dict']
	if not isinstance(state, dict):
		return 'non_dict_state'
	h = hashlib.sha256()
	for key in sorted(state.keys()):
		t = state[key]
		if not torch.is_tensor(t):
			continue
		h.update(key.encode('utf-8'))
		h.update(str(tuple(t.shape)).encode('utf-8'))
		h.update(t.detach().cpu().numpy().tobytes())
	return h.hexdigest()


def evaluate_with_planner_metrics(env, agent: TDMPC, num_episodes: int, step: int) -> Dict[str, float]:
	episode_rewards = []
	plan_sums = {}
	plan_count = 0
	for _ in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0.0, 0
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t == 0)
			plan_metrics = getattr(agent, '_last_plan_metrics', {})
			for k, v in plan_metrics.items():
				v = float(v)
				if np.isfinite(v):
					plan_sums[k] = plan_sums.get(k, 0.0) + v
			plan_count += 1
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += float(reward)
			t += 1
		episode_rewards.append(ep_reward)
	result = {'episode_reward': float(np.mean(episode_rewards))}
	if plan_count > 0:
		for k, v in plan_sums.items():
			result[k] = v / plan_count
	return result


def _run_one(cfg, seed: int, tdmpc_model_path: str, sampler_path: Optional[str]) -> Dict[str, float]:
	set_seed(seed)
	cfg.seed = int(seed)
	cfg.use_mi_warmstart = sampler_path is not None
	cfg.mi_model_path = '' if sampler_path is None else str(sampler_path)
	env = make_env(cfg)
	agent = TDMPC(cfg)
	agent.load(tdmpc_model_path)
	metrics = evaluate_with_planner_metrics(
		env=env,
		agent=agent,
		num_episodes=int(cfg.get('ab_eval_episodes', 10)),
		step=int(cfg.get('ab_eval_step', cfg.seed_steps)),
	)
	try:
		env.close()
	except Exception:
		pass
	return metrics


def _default_out_csv(cfg) -> Path:
	return Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / 'ab_eval.csv'


def run_ab(cfg):
	assert torch.cuda.is_available(), 'A/B evaluation requires CUDA in this TD-MPC setup.'

	tdmpc_model_path = str(cfg.get('ab_tdmpc_model_path', '') or cfg.get('mi_tdmpc_model_path', ''))
	if not tdmpc_model_path:
		raise ValueError('Set ab_tdmpc_model_path=<path_to_tdmpc_model.pt>')
	if not os.path.isfile(tdmpc_model_path):
		raise FileNotFoundError(f'TD-MPC model not found: {tdmpc_model_path}')

	seeds = _parse_int_list(str(cfg.get('ab_eval_seeds', str(cfg.seed))))
	if len(seeds) == 0:
		seeds = [int(cfg.seed)]
	sampler_paths = _discover_sampler_paths(cfg)
	if len(sampler_paths) == 0:
		raise ValueError('No sampler checkpoints found. Set ab_sampler_paths or ab_sampler_dir.')
	sampler_meta = {}
	for p in sampler_paths:
		if not os.path.isfile(p):
			continue
		file_hash = _file_sha256(p)
		state_sig = _state_signature(p)
		sampler_meta[p] = {'file_hash': file_hash, 'state_sig': state_sig}
		print(
			f"[AB] sampler={Path(p).stem} "
			f"file_hash={file_hash[:10]} state_sig={state_sig[:10]}"
		)
	sig_to_paths = {}
	for p, meta in sampler_meta.items():
		sig_to_paths.setdefault(meta['state_sig'], []).append(p)
	duplicate_groups = [paths for paths in sig_to_paths.values() if len(paths) > 1]
	if duplicate_groups:
		print('[AB] WARNING: checkpoints with identical sampler tensors detected:')
		for group in duplicate_groups:
			print('[AB]   ' + ' | '.join(group))

	rows = []
	base_cfg = OmegaConf.to_container(cfg, resolve=True)
	for seed in seeds:
		cfg_seed = OmegaConf.create(base_cfg)
		baseline_metrics = _run_one(cfg_seed, seed, tdmpc_model_path, sampler_path=None)
		baseline_reward = baseline_metrics['episode_reward']
		row_base = {
			'seed': int(seed),
			'label': 'baseline',
			'sampler_path': '',
			'delta_reward_vs_baseline': 0.0,
			**baseline_metrics,
		}
		rows.append(row_base)
		print(f"[AB] seed={seed} baseline reward={baseline_reward:.3f}")

		for sampler_path in sampler_paths:
			if not os.path.isfile(sampler_path):
				print(f"[AB] skip missing sampler: {sampler_path}")
				continue
			cfg_seed = OmegaConf.create(base_cfg)
			metrics = _run_one(cfg_seed, seed, tdmpc_model_path, sampler_path=sampler_path)
			reward = metrics['episode_reward']
			delta = reward - baseline_reward
			label = Path(sampler_path).stem
			row = {
				'seed': int(seed),
				'label': label,
				'sampler_path': sampler_path,
				'sampler_file_hash': sampler_meta.get(sampler_path, {}).get('file_hash', ''),
				'sampler_state_sig': sampler_meta.get(sampler_path, {}).get('state_sig', ''),
				'delta_reward_vs_baseline': float(delta),
				**metrics,
			}
			rows.append(row)
			warmstart_rate = float(metrics.get('mi_warmstart_used', float('nan')))
			print(
				f"[AB] seed={seed} sampler={label} reward={reward:.3f} "
				f"delta={delta:+.3f} warmstart_rate={warmstart_rate:.3f}"
			)

	out_csv = str(cfg.get('ab_out_csv', ''))
	out_path = Path(out_csv) if out_csv else _default_out_csv(cfg)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = []
	for row in rows:
		for k in row.keys():
			if k not in fieldnames:
				fieldnames.append(k)
	with open(out_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)
	print(f'[AB] Saved detailed results to: {out_path}')

	# Aggregate mean deltas across seeds for each sampler label.
	agg = {}
	for row in rows:
		label = row['label']
		agg.setdefault(label, {'reward': [], 'delta': []})
		agg[label]['reward'].append(float(row['episode_reward']))
		agg[label]['delta'].append(float(row['delta_reward_vs_baseline']))
	print('[AB] Mean results across seeds:')
	for label, vals in sorted(agg.items(), key=lambda kv: np.mean(kv[1]['delta']), reverse=True):
		print(
			f"[AB] {label:<24} mean_reward={np.mean(vals['reward']):.3f} "
			f"mean_delta={np.mean(vals['delta']):+.3f}"
		)


if __name__ == '__main__':
	run_ab(parse_cfg(Path().cwd() / __CONFIG__))
