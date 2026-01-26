from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PretrainedLoadReport:
    loaded: int
    skipped_unexpected: int
    skipped_shape_mismatch: int
    missing_after_load: int


def _extract_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    """
    Accept either:
    - a raw state_dict (Mapping[str, Tensor])
    - a checkpoint dict containing `model` or `state_dict`
    """
    if isinstance(obj, Mapping):
        maybe_model = obj.get("model")
        if isinstance(maybe_model, Mapping):
            return maybe_model  # type: ignore[return-value]
        maybe_sd = obj.get("state_dict")
        if isinstance(maybe_sd, Mapping):
            return maybe_sd  # type: ignore[return-value]
    return obj  # type: ignore[return-value]


def _strip_prefix_if_all_keys_have_it(sd: Mapping[str, torch.Tensor], prefix: str) -> Mapping[str, torch.Tensor]:
    if prefix == "":
        return sd
    keys = list(sd.keys())
    if len(keys) == 0:
        return sd
    if not all(k.startswith(prefix) for k in keys):
        return sd
    return {k[len(prefix) :]: v for k, v in sd.items()}


def load_pretrained_weights(
    model: nn.Module,
    pretrained_path: Path,
    *,
    prefix_to_strip: Optional[str] = None,
) -> PretrainedLoadReport:
    """
    Load pretrained weights into `model`, permissively:
    - only loads keys that exist in the current model AND have matching tensor shapes
    - ignores missing keys (new layers) and unexpected keys (old/other layers)

    This supports initializing a larger model from a smaller one (e.g. fewer layers).
    """
    path = Path(pretrained_path).expanduser().resolve()
    ckpt = torch.load(path, map_location="cpu")
    sd = dict(_extract_state_dict(ckpt))

    # Common DDP prefix.
    sd = dict(_strip_prefix_if_all_keys_have_it(sd, "module."))
    if prefix_to_strip:
        sd = {k[len(prefix_to_strip) :]: v for k, v in sd.items() if k.startswith(prefix_to_strip)}

    model_sd = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped_unexpected = 0
    skipped_shape_mismatch = 0
    for k, v in sd.items():
        if k not in model_sd:
            skipped_unexpected += 1
            continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            skipped_shape_mismatch += 1
            continue
        filtered[k] = v

    incompatible = model.load_state_dict(filtered, strict=False)
    missing_after_load = len(list(getattr(incompatible, "missing_keys", [])))
    return PretrainedLoadReport(
        loaded=len(filtered),
        skipped_unexpected=int(skipped_unexpected),
        skipped_shape_mismatch=int(skipped_shape_mismatch),
        missing_after_load=int(missing_after_load),
    )
