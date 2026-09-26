#!/usr/bin/env python3
"""Widen a trained checkpoint's observation input so it can warm-start a task with extra
observations, without changing what the policy initially does.

Why
---
Adding object observations changes the policy/critic input dimension, which normally makes an
existing checkpoint unloadable and forces training from scratch. Re-learning locomotion and
whole-body tracking is the expensive part (~6 GPU-hours for the 10k-iteration GMT teacher);
learning to *use* the object channel is the cheap part.

Appending the new inputs at the end of the observation vector and zeroing the corresponding
input-layer columns makes the widened network **numerically identical** to the original for
any input, because that column contributes ``0 * x`` to every unit. Training then discovers
non-zero weights for those columns from there.

What gets modified
------------------
1. ``actor.0.weight``  (H, D_pol)  -> (H, D_pol + n_policy),  new columns zero
2. ``critic.0.weight`` (H, D_crit) -> (H, D_crit + n_critic), new columns zero
3. ``obs_norm_state_dict``            ``_mean`` padded with 0, ``_var``/``_std`` padded with 1
4. ``privileged_obs_norm_state_dict`` same
5. ``optimizer_state_dict``           Adam ``exp_avg``/``exp_avg_sq`` for those two weight
   tensors padded with 0

The optimizer state is keyed by parameter *index*, not name, so entries are matched by tensor
shape instead of guessing the index order. This is unambiguous here: the script asserts that
exactly one parameter has each of the two shapes before touching anything.

Padding ``_var``/``_std`` with 1 (not 0) matters twice: it keeps the normaliser from dividing
by zero, and it makes the new channels pass through unscaled until enough samples accumulate.

The iteration counter is preserved, so ``--resume`` continues the run's numbering; pass
``--reset_iter`` to start the widened model at 0 instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ACTOR_FIRST = "actor.0.weight"
CRITIC_FIRST = "critic.0.weight"


def pad_cols(t: torch.Tensor, n: int, value: float = 0.0) -> torch.Tensor:
    """Append ``n`` columns filled with ``value`` to a 2-D tensor."""
    if n == 0:
        return t
    pad = torch.full((t.shape[0], n), value, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=1)


def pad_optimizer_state(opt: dict, old_shape: tuple[int, int], n_new: int) -> int:
    """Pad every Adam moment whose shape matches ``old_shape``. Returns how many were hit."""
    if n_new == 0:
        return 0
    hits = 0
    for entry in opt.get("state", {}).values():
        for key in ("exp_avg", "exp_avg_sq"):
            m = entry.get(key)
            if isinstance(m, torch.Tensor) and tuple(m.shape) == old_shape:
                entry[key] = pad_cols(m, n_new)
                hits += 1
    return hits


def count_shape(sd: dict, shape: tuple[int, int]) -> int:
    return sum(
        1 for v in sd.values() if isinstance(v, torch.Tensor) and tuple(v.shape) == shape
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="checkpoint to widen, e.g. model_9999.pt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--new_policy_dims", type=int, required=True,
                    help="observations appended to the POLICY group")
    ap.add_argument("--new_critic_dims", type=int, required=True,
                    help="observations appended to the CRITIC (privileged) group")
    ap.add_argument("--reset_iter", action="store_true",
                    help="set iter to 0 so the widened model trains as a fresh run")
    a = ap.parse_args()

    ckpt = torch.load(a.input, map_location="cpu", weights_only=False)
    msd = ckpt["model_state_dict"]

    for key in (ACTOR_FIRST, CRITIC_FIRST):
        if key not in msd:
            print(f"!! {key} missing -- is this a plain actor-critic checkpoint?", file=sys.stderr)
            return 1

    old_actor = tuple(msd[ACTOR_FIRST].shape)
    old_critic = tuple(msd[CRITIC_FIRST].shape)
    d_pol, d_crit = old_actor[1], old_critic[1]

    # Shape-based optimizer matching is only safe if the shapes are unique in the model.
    for name, shape in ((ACTOR_FIRST, old_actor), (CRITIC_FIRST, old_critic)):
        n = count_shape(msd, shape)
        if n != 1:
            print(f"!! {n} parameters share {name}'s shape {shape}; shape-based optimizer "
                  f"matching would be ambiguous. Aborting.", file=sys.stderr)
            return 1

    print(f"  policy obs {d_pol} -> {d_pol + a.new_policy_dims}   "
          f"({ACTOR_FIRST} {old_actor})")
    print(f"  critic obs {d_crit} -> {d_crit + a.new_critic_dims}   "
          f"({CRITIC_FIRST} {old_critic})")

    msd[ACTOR_FIRST] = pad_cols(msd[ACTOR_FIRST], a.new_policy_dims)
    msd[CRITIC_FIRST] = pad_cols(msd[CRITIC_FIRST], a.new_critic_dims)

    for norm_key, n_new, expect in (
        ("obs_norm_state_dict", a.new_policy_dims, d_pol),
        ("privileged_obs_norm_state_dict", a.new_critic_dims, d_crit),
    ):
        if norm_key not in ckpt:
            print(f"  {norm_key}: absent, skipped")
            continue
        norm = ckpt[norm_key]
        got = tuple(norm["_mean"].shape)[-1]
        if got != expect:
            print(f"!! {norm_key}._mean has {got} channels but the layer expects {expect}",
                  file=sys.stderr)
            return 1
        norm["_mean"] = pad_cols(norm["_mean"], n_new, 0.0)
        norm["_var"] = pad_cols(norm["_var"], n_new, 1.0)
        if "_std" in norm:
            norm["_std"] = pad_cols(norm["_std"], n_new, 1.0)
        print(f"  {norm_key}: {expect} -> {expect + n_new}  (mean+0, var/std+1)")

    if "optimizer_state_dict" in ckpt:
        h_a = pad_optimizer_state(ckpt["optimizer_state_dict"], old_actor, a.new_policy_dims)
        h_c = pad_optimizer_state(ckpt["optimizer_state_dict"], old_critic, a.new_critic_dims)
        print(f"  optimizer Adam moments padded: actor {h_a}, critic {h_c} "
              f"(2 each = exp_avg + exp_avg_sq)")
        if a.new_policy_dims and h_a != 2:
            print(f"!! expected 2 actor moment tensors, padded {h_a}", file=sys.stderr)
            return 1
        if a.new_critic_dims and h_c != 2:
            print(f"!! expected 2 critic moment tensors, padded {h_c}", file=sys.stderr)
            return 1

    if a.reset_iter:
        ckpt["iter"] = 0
        print("  iter reset to 0")
    else:
        print(f"  iter kept at {ckpt.get('iter')}")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, a.output)
    print(f"  written {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
