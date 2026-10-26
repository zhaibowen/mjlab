from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  balance_car_flat_env_cfg,
)
from .rl_cfg import balance_car_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Balance-Car-V2",
  env_cfg=balance_car_flat_env_cfg(),
  play_env_cfg=balance_car_flat_env_cfg(play=True),
  rl_cfg=balance_car_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
