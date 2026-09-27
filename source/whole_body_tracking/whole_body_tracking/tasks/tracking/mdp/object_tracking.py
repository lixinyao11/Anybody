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
``root_pos_w + scene.env_origins`` and then perturbs it in place (``pose_range`` is non-empty
in the shipped cfg: +-0.05 m in x/y and +-0.2 rad of yaw). The box is placed at
``object_pos_w + env_origins`` with no perturbation, so that randomisation becomes
randomisation of the robot's pose *relative to* the box -- which is what it is for. See
``_resample_command`` for why copying the delta would be wrong.

Still to wire up (not in this module): the env cfg that attaches the box to the scene and
registers the observation/reward terms, and the gym task id.
"""

from __future__ import annotations

import os
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
LARGEBOX_VISUAL_SIZE: tuple[float, float, float] = (0.4712, 0.4587, 0.4079)

# The simulated box must be smaller than the visual mesh, and the data says by how much.
#
# OmniRetarget's reference holds the box centre at a constant z = 0.183 m. Resting on flat
# ground that implies a collision half-height of 0.183, i.e. 0.021 m less than the visual
# mesh's 0.204 -- their collision proxy is ~2.1 cm smaller per side than the render mesh.
# Independently, measuring the reference motion against the full-size box gives robot bodies
# penetrating it on 257 of 664 frames with a MEDIAN depth of 1.9 cm (90th pct 5.6 cm): the
# retargeted motion is only kinematically consistent with a box about that much smaller.
#
# Using the visual size for collision therefore breaks three things at once: the box starts
# 2.1 cm underground and is popped up at every reset, and the reference hand pose is
# permanently inside the box, so physics resolves it by shoving the box off its path. That is
# what produced a stable 0.47 m tracking error which neither more mass nor a 6x larger reward
# weight could touch.
COLLISION_SHRINK_PER_SIDE: float = 0.021
LARGEBOX_SIZE: tuple[float, float, float] = tuple(
    v - 2.0 * COLLISION_SHRINK_PER_SIDE for v in LARGEBOX_VISUAL_SIZE
)  # (0.4292, 0.4167, 0.3659) -> half-height 0.1830, matching the reference z exactly
# largebox.urdf declares 0.1 kg, which is implausible for a 47 cm box and measurably wrong
# here: at 0.1 kg the box travelled 0.48 m against a 0.30 m reference (1.6x too far) and picked
# up 0.96 rad of spurious rotation -- hand contact launches it instead of sliding it. The
# reference itself slides smoothly at constant z over ~1.5 m, i.e. a quasi-static push against
# friction, which implies real mass. 3 kg is plausible for a box this size and needs only
# ~26 N to push at mu=0.9, well within a ~35 kg G1's means.
# Override with OBJECT_MASS_KG to sweep without editing code.
DEFAULT_MASS: float = float(os.environ.get("OBJECT_MASS_KG", "3.0"))
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
        # Log-only, in metres and radians. The reward terms are exp(-err^2/std^2), which
        # saturates near zero once the error exceeds ~std and therefore cannot tell "the box is
        # 0.4 m off" from "the box is 4 m off" -- exactly the distinction needed to decide
        # whether the object weight is too low or the task is simply not being attempted.
        self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_rot"] = torch.zeros(self.num_envs, device=self.device)
        # Displacement since reset, for the box and for its reference. A large position error
        # alone cannot say WHY the box is off, and the two causes need opposite fixes:
        #   travel_object ~ 0 while travel_ref > 0  -> the robot never really touches it,
        #                                              so the object reward is too weak;
        #   travel_object >> travel_ref             -> a 0.1 kg box is being batted away by
        #                                              hand contact, so the mass is wrong.
        self.metrics["travel_object"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["travel_object_ref"] = torch.zeros(self.num_envs, device=self.device)
        self._object_start_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._object_ref_start_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        has = self.object_has_ref
        pos_err = torch.norm(self._object.data.root_pos_w - self.object_ref_pos_w, dim=-1)
        rot_err = quat_error_magnitude(self.object_ref_quat_w, self._object.data.root_quat_w)
        # Zero (not NaN) for motions with no object, so the logged mean stays finite.
        self.metrics["error_object_pos"].copy_(torch.where(has, pos_err, torch.zeros_like(pos_err)))
        self.metrics["error_object_rot"].copy_(torch.where(has, rot_err, torch.zeros_like(rot_err)))
        t_obj = torch.norm(self._object.data.root_pos_w - self._object_start_pos, dim=-1)
        t_ref = torch.norm(self.object_ref_pos_w - self._object_ref_start_pos, dim=-1)
        self.metrics["travel_object"].copy_(torch.where(has, t_obj, torch.zeros_like(t_obj)))
        self.metrics["travel_object_ref"].copy_(torch.where(has, t_ref, torch.zeros_like(t_ref)))

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

        # The box is placed at its ABSOLUTE reference pose (plus the env origin), and
        # deliberately does *not* receive the robot's reset randomisation.
        #
        # MultiMotionCommand._resample_command perturbs the robot only, and in place:
        #     root_pos += rand_samples[:, 0:3]                    # pure translation
        #     root_ori  = quat_mul(orientations_delta, root_ori)   # spin about its own root
        # That is not a rigid transform of the scene, so there is no "same transform" to
        # apply to the box. Copying the delta onto the box would instead *cancel* the
        # randomisation, making the robot-relative-to-box pose identical on every reset --
        # the opposite of what the randomisation is for.
        #
        # With the shipped cfg the robot therefore starts offset from the box by up to
        # 0.05 m in x/y, 0.01 m in z and 0.2 rad of yaw (~13 cm of lateral error at the
        # ~0.67 m pelvis-to-box distance these clips hold), on top of the +-0.52 rad joint
        # randomisation. That is real task difficulty, and it is the same randomisation the
        # body-only teacher trained under.
        #
        # reset_base_xy_to_origin is a different matter: it *replaces* the robot's XY with
        # the env origin, discarding the motion's own XY, so the robot would not be anywhere
        # near the box. Fail loudly instead of training on a broken relationship.
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
        # Anchor the travel metrics to this reset.
        self._object_start_pos[env_ids_t] = obj_pos
        self._object_ref_start_pos[env_ids_t] = obj_pos


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
