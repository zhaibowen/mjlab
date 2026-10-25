from pathlib import Path

import mujoco
import mujoco.viewer as viewer

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

G1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "balance_car_v2" / "xmls" / "balance_car.xml"
)
assert G1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(G1_XML))

ACTUATOR_2312S = ElectricActuator(
  reflected_inertia=4.25e-6,
  velocity_limit=22.0,
  effort_limit=0.048,
)

BC_ACTUATOR_2312S = BuiltinMotorActuatorCfg(
  target_names_expr=(".*_motor_joint", ),
  effort_limit=ACTUATOR_2312S.effort_limit,
  armature=ACTUATOR_2312S.reflected_inertia,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.0625),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={".*_collision": 3},
  priority={".*_collision": 0},
  friction={".*_collision": (0.6, 0.005, 0.0001)},
)

BC_ARTICULATION = EntityArticulationInfoCfg(actuators=(BC_ACTUATOR_2312S,))

def get_balance_car_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=BC_ARTICULATION,
  )

BALANCE_CAR_ACTION_SCALE: dict[str, float] = {}
for a in BC_ARTICULATION.actuators:
  assert isinstance(a, BuiltinMotorActuatorCfg)
  e = a.effort_limit
  names = a.target_names_expr
  for n in names:
    BALANCE_CAR_ACTION_SCALE[n] = 0.5 * e

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_balance_car_cfg())

  viewer.launch(robot.spec.compile())