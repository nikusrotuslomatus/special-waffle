import torch
import numpy as np
from pathlib import Path

from tdmpc.src.cfg import parse_cfg
from tdmpc.src.algorithm.tdmpc import TDMPC
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.append(str(root / "tdmpc" / "src"))

print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for TD-MPC in this repo.")

# Load config and set minimal fields that env normally sets
cfg = parse_cfg(Path("tdmpc/cfgs"))
cfg.obs_shape = (10,)
cfg.action_dim = 2
cfg.action_shape = (2,)

# Make planning cheap
cfg.seed_steps = 0
cfg.num_samples = 32
cfg.num_elites = 8
cfg.iterations = 2
cfg.mixture_coef = 0.0
cfg.horizon = 5
cfg.horizon_schedule = "5"
cfg.std_schedule = "0.5"

# Enable MI warm-start
cfg.use_mi_warmstart = True
cfg.mi_num_samples = 8
cfg.mi_delta = -1.0  # allow warm-start even if only slightly better

agent = TDMPC(cfg)

obs = np.zeros(cfg.obs_shape, dtype=np.float32)
a = agent.plan(obs, eval_mode=True, step=1, t0=True)

print("action shape/min/max:", a.shape, float(a.min()), float(a.max()))
print("plan metrics:", agent._last_plan_metrics)
