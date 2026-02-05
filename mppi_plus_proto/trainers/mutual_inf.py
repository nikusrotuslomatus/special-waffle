import shutil
import torch
import torch.nn as nn

from pathlib import Path
from mppi_plus_proto.trainers.data_samplers import SimpleActionSampler
from mppi_plus_proto.trainers.state_tracker import ModelStateTracker
from mppi_plus_proto.dynamics_models.models import AbstractDynamicsModel


class RolloutCollector:

    def __init__(self, 
                 dynamics: AbstractDynamicsModel,
                 horizon: int,
                 action_lb: tuple[float, ...],
                 action_ub: tuple[float, ...],
                 action_mid: tuple[float, ...] | None = None,
                 state_dim_cut: int | None = None) -> None:
        assert dynamics.backend == "torch"
        self._dynamcis = dynamics
        self._horizon = horizon
        self._action_lb = torch.tensor(action_lb, 
                                       device=dynamics.device, 
                                       dtype=torch.float32)
        self._action_ub = torch.tensor(action_ub, 
                                       device=dynamics.device, 
                                       dtype=torch.float32)
        if action_mid is not None:
            self._action_mid = torch.tensor(action_mid,
                                            device=dynamics.device,
                                            dtype=torch.float32)
        else:
            self._action_mid = None
        self._unnormalize_scale = (self._action_ub - self._action_lb) / 2.
        self._unnormalize_shift = (self._action_ub + self._action_lb) / 2.
        if state_dim_cut is not None:
            self._state_dim = state_dim_cut
        else:
            self._state_dim = dynamics.state_dim
    
    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._dynamcis.action_dim

    @property
    def action_lb(self) -> torch.Tensor:
        return self._action_lb

    @property
    def action_ub(self) -> torch.Tensor:
        return self._action_ub

    @property
    def action_mid(self) -> torch.Tensor:
        return self._action_mid

    def unnormalize(self, u_seq: torch.Tensor) -> torch.Tensor:
        u_seq = self._unnormalize_scale * u_seq + self._unnormalize_shift
        return u_seq

    def rollout(self, 
                u_seq: torch.Tensor,
                x_init: torch.Tensor | None = None,
                clip: bool = False) -> torch.Tensor:
        if clip:
            u_seq = torch.clamp(u_seq, 
                                self.action_lb, 
                                self.action_ub)
        x_seq = self._dynamcis.rollout(u_seq, x_init)
        x_seq = x_seq[:, :, :self._state_dim]
        return x_seq


class MIJointTrainer:

    def __init__(self,
                 rollout_collector: RolloutCollector,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 batch_size: int,
                 lr: float,
                 scheduler: bool,
                 clip_dynamics: bool,
                #  checkpoint_root: str | Path,
                #  experiment_name: str,
                 wd: float = 0.,
                 alpha: float = 1.,
                 max_grad_norm: float | None = None,
                 checkpoint_epochs: list[int] | None = None,
                 device: str = "cuda"):
        self._rollout_collector = rollout_collector
        self._batch_size = batch_size
        self._lr = lr
        self._wd = wd
        self._alpha = alpha
        self._use_scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self._device = device
        self._generator = generator
        self._discriminator = discriminator
        self._clip_dynamics = clip_dynamics
        if checkpoint_epochs is None:
            self._checkpoint_epochs = []
        else:
            self._checkpoint_epochs = checkpoint_epochs

    def train(self,
             checkpoint_root: str | Path,
             experiment_name: str,
             n_epochs: int,
             logging_freq: int = 1):
        checkpoint_dir = Path(checkpoint_root) / experiment_name
        try:
            self._do_train(checkpoint_dir=checkpoint_dir,
                           n_epochs=n_epochs,
                           logging_freq=logging_freq)
        except KeyboardInterrupt:
            pass

        checkpoint_epochs = checkpoint_dir / "generator_epochs_all.pth"
        torch.save(self._generator.state_dict(), checkpoint_epochs)

    def _do_train(self,
                  checkpoint_dir: Path,
                  n_epochs: int,
                  logging_freq: int = 1):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True)

        trackers = {
            "total_loss": ModelStateTracker(checkpoint_dir / "generator_total_loss.pth"),
            "info_loss": ModelStateTracker(checkpoint_dir / "generator_info_loss.pth")
        }

        self._generator.to(self._device)
        self._generator.train()
        self._discriminator.to(self._device)
        self._discriminator.train()

        optimizer = torch.optim.Adam(
            list(self._generator.parameters()) +\
            list(self._discriminator.parameters()),
            lr=self._lr,
            weight_decay=self._wd)
        if self._use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   'min', 
                                                                   factor=0.5, 
                                                                   patience=50)
        else:
            scheduler = None

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            u, log_prob_u = self._generator.sample(self._batch_size)
            u_ent = -log_prob_u.mean()

            with torch.no_grad():
                x = self._rollout_collector.rollout(u.reshape(self._batch_size, self._rollout_collector.horizon, -1),
                                                    clip=self._clip_dynamics)
            x = x[:, 1:, :]  # Ignore first state since we are unconditioned
            x = x.reshape(self._batch_size, x.shape[1] * x.shape[2])
            log_prob_u_given_x = self._discriminator.log_prob(x, u)

            loss_info = -log_prob_u_given_x.mean()
            loss_ent = -u_ent
            loss = loss_info + self._alpha * loss_ent

            loss.backward()
            if self._max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._generator.parameters(), 
                                               self._max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self._discriminator.parameters(), 
                                               self._max_grad_norm)
            optimizer.step()

            loss = loss.item()
            loss_info = loss_info.item()
            loss_ent = loss_ent.item()
            if scheduler is not None:
                scheduler.step(loss)

            trackers["total_loss"].update(self._generator, loss)
            trackers["info_loss"].update(self._generator, loss_info)
            if (epoch + 1) in self._checkpoint_epochs:
                torch.save(self._generator.state_dict(), checkpoint_dir / f"generator_epochs_{(epoch+1)}.pth")

            if epoch % logging_freq == 0:
                print(f"Epoch {epoch}, Loss: Total {loss}, Info {loss_info:.2f}, Ent: {loss_ent:.2f}")

