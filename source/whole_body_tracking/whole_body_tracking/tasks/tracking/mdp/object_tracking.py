"""Object-interaction tracking: a dynamic box in the scene, tracked against the
OmniRetarget object reference trajectory.

What this adds on top of plain motion tracking
----------------------------------------------
The base tracking task simulates the robot alone; the OmniRetarget clips, however, are
*interaction* data -- the robot pushes a box along a recorded path. Training on the body
motion alone teaches the robot to mime the push with nothing to push. This module puts the
box in the simulation as a **dynamic** rigid body and rewards the robot for moving it along
its reference trajectory, so contact actually has to happen.

Contrast with ``obstacle_reach``: those boxes are ``kinematic_enabled=True`` (immovable
colliders, the robot bounces off them). Here the box must be pushable, so it is a normal
dynamic body with mass and the reward compares its *simulated* pose to the reference.

Geometry / mass
---------------
``OmniRetarget_Dataset/models/largebox/`` ships ``largebox.obj`` (13551 verts / 27581 faces)
whose axis-aligned bounding box measures 0.4712 x 0.4587 x 0.4079 m. 90% of its vertices sit
more than 2 cm inside that box, i.e. it is a *volumetric* mesh rather than a surface shell,
which makes ``convexDecomposition`` collision both slow and unpredictable. Since the object
is box-shaped, a ``CuboidCfg`` primitive at the measured extents is exact *and* cheap -- and
it matches how this repo already spawns its obstacle colliders.

The source URDF declares ``mass = 0.1 kg`` with ``ixx=iyy=izz=0.002``. That is far lighter
than a real 47 cm box and was presumably chosen for retargeting stability; at 0.1 kg the box
is trivially pushed but also trivially knocked away. It is exposed as ``DEFAULT_MASS`` so it
can be raised without touching the asset.

Frames
------
``MultiMotionLoader.object_pos_w`` is in the *motion's* world frame, exactly like
``root_pos_w``. ``MultiMotionCommand._resample_command`` places the robot at
``root_pos_w + scene.env_origins`` (plus optional randomisation), so the box is placed with
the *same* net translation -- otherwise multi-env offsets or reset randomisation would pull
the robot and the box apart. With the shipped config (``pose_range = {}``,
``reset_base_xy_to_origin = False``) that delta is exactly ``env_origins``, but the delta is
computed rather than assumed so enabling randomisation later does not silently break the
robot/box relationship.

Still to wire up (not in this module): the env cfg that attaches the box to the scene and
registers the observation/reward terms, and the gym task id.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_error_magnitude, quat_mul, quat_rotate_inverse

from .commands import MultiMotionCommand, MultiMotionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Scene key the command and the obs/reward terms agree on.
OBJECT_ASSET_NAME = "tracked_object"

# Axis-aligned extents of OmniRetarget's largebox.obj, measured from the mesh.
LARGEBOX_SIZE: tuple[float, float, float] = (0.4712, 0.4587, 0.4079)
# From largebox.urdf. Deliberately surfaced: 0.1 kg is very light for this volume.
DEFAULT_MASS: float = 0.1
_OBJECT_COLOR = (0.70, 0.80, 0.90)


def make_object_cfg(
    size: tuple[float, float, float] = LARGEBOX_SIZE,
    mass: float = DEFAULT_MASS,
    static_friction: float = 0.9,
    dynamic_friction: float = 0.9,
) -> RigidObjectCfg:
    """A pushable box matching OmniRetarget's largebox.

    Spawned far below the ground so that the very first physics step cannot have the robot
    collide with a box that has not been positioned yet; the command teleports it into place
    on every reset (same trick as ``make_obstacle_collider_cfgs``).
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TrackedObject",
        spawn=sim_utils.CuboidCfg(
            size=size,
            # Dynamic (kinematic_enabled defaults to False): the robot must actually push it.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_OBJECT_COLOR),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=static_friction, dynamic_friction=dynamic_friction
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
    )


class ObjectMultiMotionCommand(MultiMotionCommand):
    """``MultiMotionCommand`` that also owns the tracked object's reset and reference."""

    cfg: "ObjectMultiMotionCommandCfg"

    def __init__(self, cfg: "ObjectMultiMotionCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._object = env.scene[cfg.object_asset_name]

    # -- reference accessors (world frame, env-origin offset applied) -------------------
    @property
    def object_ref_pos_w(self) -> torch.Tensor:
        """Reference object position for every env at its current frame, (N, 3)."""
        return self._gather_all("object_pos_w") + self._env.scene.env_origins

    @property
    def object_ref_quat_w(self) -> torch.Tensor:
        """Reference object orientation (w, x, y, z) for every env, (N, 4)."""
        return self._gather_all("object_quat_w")

    @property
    def object_has_ref(self) -> torch.Tensor:
        """(N,) bool -- False for motions that carry no object trajectory."""
        return self.motion_dir_loader.motion_has_object.to(self.device)[self.env_motion_indices]

    def _gather_all(self, attr: str) -> torch.Tensor:
        all_ids = torch.arange(self.num_envs, device=self.device)
        return self._gather_by_motion_for_envs(attr, all_ids)

    # -- reset -------------------------------------------------------------------------
    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        # The base class places the robot at ``root_pos_w + env_origins`` (see
        # MultiMotionCommand._resample_command), so the box gets the same offset and the
        # recorded robot/box geometry is preserved. Two config knobs would break that
        # assumption, and both are off in the shipped cfg -- fail loudly rather than
        # silently placing the box somewhere the robot is not:
        #   * ``pose_range``: adds an unstored random delta to the robot root only. Reading
        #     the robot pose back from the sim to recover it is not reliable here because
        #     Isaac Lab 2.x distinguishes root *link* and root *com* frames.
        #   * ``reset_base_xy_to_origin``: discards the motion's XY entirely.
        if self.cfg.pose_range:
            raise NotImplementedError(
                "ObjectMultiMotionCommand does not support commands.motion.pose_range yet: the "
                "random root delta is not observable from here, so the box would be offset "
                f"from the robot. Got pose_range={self.cfg.pose_range!r}."
            )
        if self.cfg.reset_base_xy_to_origin:
            raise NotImplementedError(
                "ObjectMultiMotionCommand does not support reset_base_xy_to_origin=True: the "
                "robot's XY is replaced by the env origin while the box would keep the "
                "motion's XY, so they would not line up."
            )

        delta = self._env.scene.env_origins[env_ids_t]
        obj_pos = self._gather_by_motion_for_envs("object_pos_w", env_ids_t) + delta
        obj_quat = self._gather_by_motion_for_envs("object_quat_w", env_ids_t)
        # Zero linear/angular velocity: the reference gives pose only, and carrying stale
        # velocity across a reset would launch the box.
        zeros = torch.zeros((env_ids_t.numel(), 6), device=self.device)
        self._object.write_root_state_to_sim(
            torch.cat([obj_pos, obj_quat, zeros], dim=-1), env_ids=env_ids_t
        )


@configclass
class ObjectMultiMotionCommandCfg(MultiMotionCommandCfg):
    """Cfg for :class:`ObjectMultiMotionCommand`."""

    class_type: type = ObjectMultiMotionCommand
    object_asset_name: str = OBJECT_ASSET_NAME


# ---------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------
def _anchor_frame(env: "ManagerBasedRLEnv", command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Position and orientation of the robot's anchor body, (N, 3) and (N, 4).

    Uses the command term's own ``robot_anchor_*`` properties so this stays consistent with
    the existing ``motion_anchor_*`` observations instead of re-deriving the index.
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.robot_anchor_pos_w, cmd.robot_anchor_quat_w


def object_pose_anchor_b(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Simulated object pose in the robot anchor frame: position (3) + quaternion (4).

    Anchor-relative rather than world so the policy sees the same quantity regardless of
    where in the terrain the episode happens to start -- matching how the existing
    ``motion_anchor_*`` observations are built.
    """
    cmd = env.command_manager.get_term(command_name)
    a_pos, a_quat = _anchor_frame(env, command_name)
    obj = env.scene[cmd.cfg.object_asset_name]
    pos_b = quat_rotate_inverse(a_quat, obj.data.root_pos_w - a_pos)
    quat_b = quat_mul(quat_conjugate(a_quat), obj.data.root_quat_w)
    return torch.cat([pos_b, quat_b], dim=-1)


def object_ref_pose_anchor_b(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Reference object pose in the robot anchor frame: position (3) + quaternion (4)."""
    cmd = env.command_manager.get_term(command_name)
    a_pos, a_quat = _anchor_frame(env, command_name)
    pos_b = quat_rotate_inverse(a_quat, cmd.object_ref_pos_w - a_pos)
    quat_b = quat_mul(quat_conjugate(a_quat), cmd.object_ref_quat_w)
    return torch.cat([pos_b, quat_b], dim=-1)


def object_ref_error(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Reference-minus-simulated object position in the anchor frame, (N, 3).

    A compact "where should the box go from here" signal; cheaper for the policy to use than
    differencing the two absolute poses itself.
    """
    cmd = env.command_manager.get_term(command_name)
    _, a_quat = _anchor_frame(env, command_name)
    obj = env.scene[cmd.cfg.object_asset_name]
    return quat_rotate_inverse(a_quat, cmd.object_ref_pos_w - obj.data.root_pos_w)


# ---------------------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------------------
def object_position_tracking(
    env: "ManagerBasedRLEnv", command_name: str, std: float
) -> torch.Tensor:
    """exp(-||p_sim - p_ref||^2 / std^2), zero for motions with no object reference."""
    cmd = env.command_manager.get_term(command_name)
    obj = env.scene[cmd.cfg.object_asset_name]
    err = torch.sum(torch.square(obj.data.root_pos_w - cmd.object_ref_pos_w), dim=-1)
    return torch.exp(-err / std**2) * cmd.object_has_ref.float()


def object_orientation_tracking(
    env: "ManagerBasedRLEnv", command_name: str, std: float
) -> torch.Tensor:
    """exp(-angle^2 / std^2) between simulated and reference object orientation."""
    cmd = env.command_manager.get_term(command_name)
    obj = env.scene[cmd.cfg.object_asset_name]
    # Same form as motion_global_anchor_orientation_error_exp: reuse the repo's angle metric
    # rather than hand-rolling acos, so object and body orientation rewards share a scale.
    error = quat_error_magnitude(cmd.object_ref_quat_w, obj.data.root_quat_w) ** 2
    return torch.exp(-error / std**2) * cmd.object_has_ref.float()


def object_position_error(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Raw object position error in metres -- for metrics, not reward shaping."""
    cmd = env.command_manager.get_term(command_name)
    obj = env.scene[cmd.cfg.object_asset_name]
    return torch.norm(obj.data.root_pos_w - cmd.object_ref_pos_w, dim=-1)


def object_lost(
    env: "ManagerBasedRLEnv", command_name: str, threshold: float
) -> torch.Tensor:
    """Termination: the box drifted further than ``threshold`` from its reference.

    Without this the episode keeps paying body-tracking reward long after the box has been
    knocked away, which teaches the robot to ignore it.
    """
    cmd = env.command_manager.get_term(command_name)
    obj = env.scene[cmd.cfg.object_asset_name]
    dist = torch.norm(obj.data.root_pos_w - cmd.object_ref_pos_w, dim=-1)
    return (dist > threshold) & cmd.object_has_ref
