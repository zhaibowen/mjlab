from pathlib import Path

import mujoco
import mujoco.viewer as viewer

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

G1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "balance_car" / "xmls" / "balance_car.xml"
)
assert G1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(G1_XML))

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=10.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 1.0
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

BC_ACTUATOR_4010 = BuiltinVelocityActuatorCfg(
  target_names_expr=(".*_wheel_joint", ),
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.03),
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

BC_ARTICULATION = EntityArticulationInfoCfg(actuators=(BC_ACTUATOR_4010,))

def get_balance_car_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=BC_ARTICULATION,
  )

BALANCE_CAR_ACTION_SCALE: dict[str, float] = {}
for a in BC_ARTICULATION.actuators:
  assert isinstance(a, BuiltinVelocityActuatorCfg)
  names = a.target_names_expr
  for n in names:
    BALANCE_CAR_ACTION_SCALE[n] = 0.8 * ACTUATOR_4010.velocity_limit

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_balance_car_cfg())

  viewer.launch(robot.spec.compile())