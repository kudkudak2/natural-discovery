from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arc_aug import AugmentSpec
from arc_train_utils import (
    LearningCurves,
    TensorizedDataset,
    VOCAB_SIZE,
    assert_disjoint_datasets,
    cap_dataset,
    concat_datasets,
    count_params,
    evaluate_accuracy,
    load_skill_split,
    pad_dataset_to,
    maybe_load_external_arc_splits,
    maybe_load_skill_split,
    maybe_move_train_pool,
    plot_learning_curves,
    prepare_batch,
    progress as progress_iter,
    prompt_seq_len,
    run_unit_tests,
    show_one_example,
    split_dataset,
    write_learning_curves_csv,
)
from checkpointing import load_pretrained_weights


DEFAULT_TRAIN_SKILLS = (11, 12, 14, 15, 16)
DEFAULT_TRAIN_WITH_OOD_SKILLS = (11, 12, 14, 15, 16)


def _prompt_rows_cols(
    *,
    t: int,
    input_grid_size: int,
    output_grid_size: int,
    num_demos: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-token (row, col) coordinates for the ARC prompt sequence, plus an is_sep mask.
    Input grids are input_grid_size x input_grid_size; output grids are output_grid_size x output_grid_size.
    Row/col are in [0, max(g_in,g_out)-1] for embedding indexing.
    """
    g_in = int(input_grid_size)
    g_out = int(output_grid_size)
    nd = int(num_demos)
    if g_in <= 0 or g_out <= 0:
        raise ValueError(f"input_grid_size and output_grid_size must be >= 1, got {g_in}, {g_out}")
    g_max = max(g_in, g_out)
    in_tokens = g_in * g_in
    out_tokens = g_out * g_out
    demo_block = in_tokens + 1 + out_tokens + 1
    demos_total = nd * demo_block
    test_block = in_tokens + 1
    total = demos_total + test_block
    if t > total:
        raise ValueError(f"Unexpected t={t} for input_g={g_in} output_g={g_out} num_demos={nd} (expected <= {total})")

    pos = torch.arange(int(t), device=device)
    row = torch.zeros(int(t), device=device, dtype=torch.long)
    col = torch.zeros(int(t), device=device, dtype=torch.long)
    is_sep = torch.zeros(int(t), device=device, dtype=torch.bool)

    # Demo region: (x SEP y SEP) * num_demos
    in_demos = pos < int(demos_total)
    if in_demos.any():
        p = pos[in_demos]
        did = (p // int(demo_block)).to(torch.long)
        within = (p % int(demo_block)).to(torch.long)
        in_x = within < int(in_tokens)
        is_sep_demo = (within == int(in_tokens)) | (within == int(in_tokens + 1 + out_tokens))
        in_y = (within > int(in_tokens)) & (within < int(in_tokens + 1 + out_tokens))
        if in_x.any():
            cell = within[in_x]
            idx = p[in_x]
            row[idx] = (cell // g_in).clamp(max=g_max - 1)
            col[idx] = (cell % g_in).clamp(max=g_max - 1)
        if in_y.any():
            cell = (within[in_y] - int(in_tokens + 1)).to(torch.long)
            idx = p[in_y]
            row[idx] = (cell // g_out).clamp(max=g_max - 1)
            col[idx] = (cell % g_out).clamp(max=g_max - 1)
        is_sep[in_demos] = is_sep_demo

    # Test region: test_x SEP
    in_test = pos >= int(demos_total)
    if in_test.any():
        p = pos[in_test]
        within = (p - int(demos_total)).to(torch.long)
        in_x = within < int(in_tokens)
        is_sep[in_test] = (within == int(in_tokens))
        if in_x.any():
            cell = within[in_x]
            idx = p[in_x]
            row[idx] = (cell // g_in).clamp(max=g_max - 1)
            col[idx] = (cell % g_in).clamp(max=g_max - 1)
    return row, col, is_sep


def _prompt_demo_rows_cols(
    *,
    t: int,
    input_grid_size: int,
    output_grid_size: int,
    num_demos: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-token (row, col, demo_id) for demo-level 2D layout.
    Input grids are input_grid_size x input_grid_size; output grids are output_grid_size x output_grid_size.
    demo_col for y tokens: offset by g_in+1 so y starts at col=g_in+1.
    """
    tt = int(t)
    g_in = int(input_grid_size)
    g_out = int(output_grid_size)
    nd = int(num_demos)
    if g_in <= 0 or g_out <= 0:
        raise ValueError(f"input_grid_size and output_grid_size must be >= 1, got {g_in}, {g_out}")
    if nd <= 0:
        raise ValueError(f"num_demos must be >= 1, got {nd}")
    in_tokens = g_in * g_in
    out_tokens = g_out * g_out
    demo_block = in_tokens + 1 + out_tokens + 1
    demos_total = nd * demo_block
    test_block = in_tokens + 1
    total = int(demos_total + test_block)
    if tt > total:
        raise ValueError(f"Unexpected t={tt} for input_g={g_in} output_g={g_out} num_demos={nd} (expected <= {total})")

    pos = torch.arange(int(tt), device=device)
    demo_row = torch.zeros(int(tt), device=device, dtype=torch.long)
    demo_col = torch.zeros(int(tt), device=device, dtype=torch.long)
    demo_id = torch.full((int(tt),), -1, device=device, dtype=torch.long)

    in_demos = pos < int(demos_total)
    if in_demos.any():
        p = pos[in_demos]
        did = (p // int(demo_block)).to(torch.long)
        within = (p % int(demo_block)).to(torch.long)
        in_x = within < int(in_tokens)
        in_y = (within > int(in_tokens)) & (within < int(in_tokens + 1 + out_tokens))
        if in_x.any():
            cell = within[in_x]
            idx = p[in_x]
            demo_row[idx] = (cell // g_in).to(torch.long)
            demo_col[idx] = (cell % g_in).to(torch.long)
            demo_id[idx] = did[in_x]
        if in_y.any():
            cell = (within[in_y] - int(in_tokens + 1)).to(torch.long)
            idx = p[in_y]
            demo_row[idx] = (cell // g_out).to(torch.long)
            demo_col[idx] = (cell % g_out).to(torch.long) + int(g_in + 1)
            demo_id[idx] = did[in_y]

    in_test = pos >= int(demos_total)
    if in_test.any():
        p = pos[in_test]
        within = (p - int(demos_total)).to(torch.long)
        in_x = within < int(in_tokens)
        if in_x.any():
            cell = within[in_x]
            idx = p[in_x]
            demo_row[idx] = (cell // g_in).to(torch.long)
            demo_col[idx] = (cell % g_in).to(torch.long)
            demo_id[idx] = int(nd)
    return demo_row, demo_col, demo_id


def _prompt_token_types(
    *,
    t: int,
    input_grid_size: int,
    output_grid_size: int,
    num_demos: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Return per-token role/type IDs for the ARC prompt.
    Types: 0 demo_x, 1 demo_y, 2 test_x, 3 sep.
    """
    tt = int(t)
    g_in = int(input_grid_size)
    g_out = int(output_grid_size)
    nd = int(num_demos)
    in_tokens = g_in * g_in
    out_tokens = g_out * g_out
    demo_block = in_tokens + 1 + out_tokens + 1
    demos_total = nd * demo_block
    test_block = in_tokens + 1
    total = int(demos_total + test_block)
    if tt > total:
        raise ValueError(f"Unexpected t={tt} for input_g={g_in} output_g={g_out} num_demos={nd} (expected <= {total})")

    pos = torch.arange(int(tt), device=device)
    token_type = torch.full((int(tt),), 3, device=device, dtype=torch.long)

    in_demos = pos < int(demos_total)
    if in_demos.any():
        p = pos[in_demos]
        within = (p % int(demo_block)).to(torch.long)
        in_x = within < int(in_tokens)
        in_y = (within > int(in_tokens)) & (within < int(in_tokens + 1 + out_tokens))
        token_type[p[in_x]] = 0
        token_type[p[in_y]] = 1

    in_test = pos >= int(demos_total)
    if in_test.any():
        p = pos[in_test]
        within = (p - int(demos_total)).to(torch.long)
        in_x = within < int(in_tokens)
        token_type[p[in_x]] = 2
    return token_type


class RelPosBias2D(nn.Module):
    """
    Learned 2D relative position bias (per head), added to self-attention logits.

    SEP tokens get 0 bias to avoid polluting attention through separators.
    """

    def __init__(self, *, grid_size: int, num_heads: int) -> None:
        super().__init__()
        g = int(grid_size)
        h = int(num_heads)
        if g <= 0:
            raise ValueError(f"grid_size must be >= 1, got {g}")
        if h <= 0:
            raise ValueError(f"num_heads must be >= 1, got {h}")
        self.grid_size = int(g)
        self.num_heads = int(h)
        self._span = int(2 * g - 1)
        self._rel_size = int(self._span * self._span)
        # (rel_size -> num_heads)
        self.bias = nn.Embedding(int(self._rel_size), int(self.num_heads))

    def forward(self, *, row: torch.Tensor, col: torch.Tensor, is_sep: torch.Tensor) -> torch.Tensor:
        """
        Args:
          row/col: (T,) long
          is_sep: (T,) bool
        Returns:
          bias: (H, T, T) float32
        """
        if row.ndim != 1 or col.ndim != 1 or is_sep.ndim != 1:
            raise ValueError("row/col/is_sep must be 1D tensors")
        if int(row.shape[0]) != int(col.shape[0]) or int(row.shape[0]) != int(is_sep.shape[0]):
            raise ValueError("row/col/is_sep must have the same length")

        t = int(row.shape[0])
        g = int(self.grid_size)
        span = int(self._span)

        # dr,dc in [-(g-1)..(g-1)] -> [0..2g-2]
        dr = (row[:, None] - row[None, :]).clamp(min=-(g - 1), max=(g - 1)) + (g - 1)
        dc = (col[:, None] - col[None, :]).clamp(min=-(g - 1), max=(g - 1)) + (g - 1)
        idx = (dr * span + dc).to(torch.long)  # (T, T)

        # Zero any pair involving SEP tokens.
        valid = (~is_sep).to(torch.bool)
        valid_pair = (valid[:, None] & valid[None, :]).to(torch.float32)  # (T, T)

        b = self.bias(idx.reshape(-1)).reshape(t, t, int(self.num_heads)).permute(2, 0, 1).contiguous()
        return b * valid_pair.unsqueeze(0)


class RelPosBias2DWithinDemo(nn.Module):
    """
    Learned 2D relative position bias (per head) for a demo-level x|gap|y layout.
    demo_row in [0..max(g_in,g_out)-1]; demo_col in [0..g_in-1] for x, [g_in+1..g_in+g_out] for y.
    """

    def __init__(
        self,
        *,
        input_grid_size: int,
        output_grid_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        g_in = int(input_grid_size)
        g_out = int(output_grid_size)
        h = int(num_heads)
        if g_in <= 0 or g_out <= 0:
            raise ValueError(f"input_grid_size and output_grid_size must be >= 1, got {g_in}, {g_out}")
        if h <= 0:
            raise ValueError(f"num_heads must be >= 1, got {h}")
        self.grid_size = int(max(g_in, g_out))  # for row span
        self.num_heads = int(h)
        self._span_r = int(2 * self.grid_size - 1)
        self._span_c = int(2 * (g_in + g_out) + 1)
        self._rel_size = int(self._span_r * self._span_c)
        self.bias = nn.Embedding(int(self._rel_size), int(self.num_heads))

    def forward(
        self,
        *,
        demo_row: torch.Tensor,
        demo_col: torch.Tensor,
        demo_id: torch.Tensor,
        is_sep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          demo_row/demo_col: (T,) long
          demo_id: (T,) long (>=0 for non-SEP tokens, -1 for SEP tokens)
          is_sep: (T,) bool
        Returns:
          bias: (H, T, T) float32
        """
        if demo_row.ndim != 1 or demo_col.ndim != 1 or demo_id.ndim != 1 or is_sep.ndim != 1:
            raise ValueError("demo_row/demo_col/demo_id/is_sep must be 1D tensors")
        t = int(demo_row.shape[0])
        if int(demo_col.shape[0]) != t or int(demo_id.shape[0]) != t or int(is_sep.shape[0]) != t:
            raise ValueError("demo_row/demo_col/demo_id/is_sep must have the same length")

        span_r = int(self._span_r)
        span_c = int(self._span_c)
        half_c = (span_c - 1) // 2
        dr = (demo_row[:, None] - demo_row[None, :]).clamp(min=-(self.grid_size - 1), max=(self.grid_size - 1)) + (self.grid_size - 1)
        dc = (demo_col[:, None] - demo_col[None, :]).clamp(min=-half_c, max=half_c) + half_c
        idx = (dr * span_c + dc).to(torch.long)  # (T, T)

        # Apply bias only within the same demo, and never involving SEP.
        valid = (~is_sep).to(torch.bool)
        same_demo = (demo_id[:, None] == demo_id[None, :]) & (demo_id[:, None] >= 0) & (demo_id[None, :] >= 0)
        valid_pair = (valid[:, None] & valid[None, :] & same_demo).to(torch.float32)  # (T, T)

        b = self.bias(idx.reshape(-1)).reshape(t, t, int(self.num_heads)).permute(2, 0, 1).contiguous()
        return b * valid_pair.unsqueeze(0)


class EncoderLayerRelPos2D(nn.Module):
    """A minimal Transformer encoder layer with learned 2D relative position bias."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        rel_pos: RelPosBias2D,
        rel_pos_demo: Optional[RelPosBias2DWithinDemo] = None,
    ) -> None:
        super().__init__()
        d = int(embed_dim)
        h = int(num_heads)
        if d <= 0:
            raise ValueError(f"embed_dim must be >= 1, got {d}")
        if h <= 0:
            raise ValueError(f"num_heads must be >= 1, got {h}")
        if d % h != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads, got {d} % {h} != 0")
        self.embed_dim = int(d)
        self.num_heads = int(h)
        self.head_dim = int(d // h)
        self.scale = float(self.head_dim) ** -0.5
        self.rel_pos = rel_pos
        self.rel_pos_demo = rel_pos_demo

        self.ln1 = nn.LayerNorm(int(d))
        self.ln2 = nn.LayerNorm(int(d))
        self.qkv = nn.Linear(int(d), int(3 * d), bias=True)
        self.proj = nn.Linear(int(d), int(d), bias=True)
        self.drop = nn.Dropout(float(dropout))

        self.ff1 = nn.Linear(int(d), int(ff_dim))
        self.ff2 = nn.Linear(int(ff_dim), int(d))
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        *,
        row: torch.Tensor,
        col: torch.Tensor,
        demo_row: Optional[torch.Tensor],
        demo_col: Optional[torch.Tensor],
        demo_id: Optional[torch.Tensor],
        is_sep: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, T, D)
        b, t, d = x.shape
        if int(d) != int(self.embed_dim):
            raise ValueError(f"Unexpected embed dim: got {int(d)} expected {int(self.embed_dim)}")

        # Pre-norm attention
        h1 = self.ln1(x)
        qkv = self.qkv(h1)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(int(b), int(t), int(self.num_heads), int(self.head_dim)).permute(0, 2, 1, 3)
        k = k.reshape(int(b), int(t), int(self.num_heads), int(self.head_dim)).permute(0, 2, 1, 3)
        v = v.reshape(int(b), int(t), int(self.num_heads), int(self.head_dim)).permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-2, -1)) * float(self.scale)  # (B, H, T, T)
        bias = self.rel_pos(row=row, col=col, is_sep=is_sep).to(dtype=logits.dtype)  # (H, T, T)
        if self.rel_pos_demo is not None:
            if demo_row is None or demo_col is None or demo_id is None:
                raise ValueError("demo_row/demo_col/demo_id must be provided when rel_pos_demo is enabled")
            demo_bias = self.rel_pos_demo(demo_row=demo_row, demo_col=demo_col, demo_id=demo_id, is_sep=is_sep).to(
                dtype=logits.dtype
            )
            bias = bias + demo_bias
        attn = F.softmax(logits + bias.unsqueeze(0), dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)  # (B, H, T, Hd)
        out = out.permute(0, 2, 1, 3).reshape(int(b), int(t), int(self.embed_dim))
        out = self.proj(out)
        out = self.drop(out)
        x = x + out

        # Pre-norm FFN
        h2 = self.ln2(x)
        ff = self.ff2(self.drop(self.act(self.ff1(h2))))
        ff = self.drop(ff)
        return x + ff


class EncoderRelPos2D(nn.Module):
    """Stack of EncoderLayerRelPos2D layers sharing the same RelPosBias2D table."""

    def __init__(
        self,
        *,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        input_grid_size: int,
        output_grid_size: int,
        demo_rel_pos_bias_2d: bool = True,
    ) -> None:
        super().__init__()
        g_max = max(int(input_grid_size), int(output_grid_size))
        rel = RelPosBias2D(grid_size=int(g_max), num_heads=int(num_heads))
        self.rel = rel
        self.demo_rel: Optional[RelPosBias2DWithinDemo] = (
            RelPosBias2DWithinDemo(
                input_grid_size=int(input_grid_size),
                output_grid_size=int(output_grid_size),
                num_heads=int(num_heads),
            )
            if bool(demo_rel_pos_bias_2d)
            else None
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayerRelPos2D(
                    embed_dim=int(embed_dim),
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    dropout=float(dropout),
                    rel_pos=rel,
                    rel_pos_demo=self.demo_rel,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        row: torch.Tensor,
        col: torch.Tensor,
        demo_row: Optional[torch.Tensor],
        demo_col: Optional[torch.Tensor],
        demo_id: Optional[torch.Tensor],
        is_sep: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                row=row,
                col=col,
                demo_row=demo_row,
                demo_col=demo_col,
                demo_id=demo_id,
                is_sep=is_sep,
            )
        return x


def _unique_ints(xs: list[int]) -> list[int]:
    """Stable unique (preserves first occurrence order)."""
    out: list[int] = []
    seen: set[int] = set()
    for x in xs:
        xi = int(x)
        if xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out


def _curriculum_delay_from_phases(
    *,
    phase1_skills: list[int],
    phase2_skills: list[int],
    phase2_start_step: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Convert a 2-phase curriculum into (train_skills, delay_train_skills, delay_train_until_steps).

    Phase 1: train only on `phase1_skills` starting at step 0.
    Phase 2: train on `phase2_skills` starting at `phase2_start_step`.
    """
    p1 = _unique_ints([int(s) for s in phase1_skills])
    p2 = _unique_ints([int(s) for s in phase2_skills])
    if len(p1) == 0:
        raise ValueError("phase1_skills must be non-empty")
    if len(p2) == 0:
        raise ValueError("phase2_skills must be non-empty")

    start = int(phase2_start_step)
    if start < 0:
        raise ValueError(f"phase2_start_step must be >= 0, got {start}")

    train_skills = _unique_ints(p1 + p2)
    phase2_only = [s for s in p2 if s not in set(p1)]
    delay_skills = _unique_ints(phase2_only)
    delay_steps = [int(start) for _ in delay_skills]
    return train_skills, delay_skills, delay_steps


def _curriculum_delay_from_3phases(
    *,
    phase1_skills: list[int],
    phase2_skills: list[int],
    phase2_start_step: int,
    phase3_skills: list[int],
    phase3_start_step: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Convert a 3-phase curriculum into (train_skills, delay_train_skills, delay_train_until_steps).

    Phase 1: train only on `phase1_skills` starting at step 0.
    Phase 2: train on `phase2_skills` starting at `phase2_start_step`.
    Phase 3: train on `phase3_skills` starting at `phase3_start_step`.

    Note: the delayed-skill mechanism is *additive* (skills can be introduced later but not removed),
    so we require phase sets to be monotonic: phase1 ⊆ phase2 ⊆ phase3.
    """
    p1 = _unique_ints([int(s) for s in phase1_skills])
    p2 = _unique_ints([int(s) for s in phase2_skills])
    p3 = _unique_ints([int(s) for s in phase3_skills])
    if len(p1) == 0:
        raise ValueError("phase1_skills must be non-empty")
    if len(p2) == 0:
        raise ValueError("phase2_skills must be non-empty")
    if len(p3) == 0:
        raise ValueError("phase3_skills must be non-empty")

    s2 = int(phase2_start_step)
    s3 = int(phase3_start_step)
    if s2 < 0:
        raise ValueError(f"phase2_start_step must be >= 0, got {s2}")
    if s3 < 0:
        raise ValueError(f"phase3_start_step must be >= 0, got {s3}")
    if s3 < s2:
        raise ValueError(f"phase3_start_step must be >= phase2_start_step, got {s3} < {s2}")

    p1_set = set(p1)
    p2_set = set(p2)
    p3_set = set(p3)
    if not p1_set.issubset(p2_set):
        missing = sorted(p1_set - p2_set)
        raise ValueError(f"phase2_skills must include all phase1_skills (missing: {missing})")
    if not p2_set.issubset(p3_set):
        missing = sorted(p2_set - p3_set)
        raise ValueError(f"phase3_skills must include all phase2_skills (missing: {missing})")

    train_skills = _unique_ints(p1 + p2 + p3)

    phase2_only = [s for s in p2 if s not in p1_set]
    phase3_only = [s for s in p3 if s not in p2_set]
    delay_skills = _unique_ints(phase2_only + phase3_only)
    delay_steps = [int(s2) for _ in phase2_only] + [int(s3) for _ in phase3_only]
    # Align delay_steps with delay_skills' stable-unique behavior.
    step_by_skill: dict[int, int] = {}
    for sid, until in zip(phase2_only, [int(s2) for _ in phase2_only]):
        step_by_skill[int(sid)] = int(until)
    for sid, until in zip(phase3_only, [int(s3) for _ in phase3_only]):
        step_by_skill[int(sid)] = int(until)
    delay_steps_aligned = [int(step_by_skill[int(sid)]) for sid in delay_skills]
    return train_skills, delay_skills, delay_steps_aligned
class ARCTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = VOCAB_SIZE,
        grid_size: int = 5,
        num_demos: int = 3,
        output_grid_size: Optional[int] = None,
        pos_encoding: str = "2d",
        rel_pos_bias_2d: bool = True,
        demo_rel_pos_bias_2d: bool = True,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        max_len: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)  # input grid size
        self.output_grid_size = int(output_grid_size if output_grid_size is not None else grid_size)
        if self.grid_size <= 0 or self.output_grid_size <= 0:
            raise ValueError(f"grid_size and output_grid_size must be >= 1, got {self.grid_size}, {self.output_grid_size}")
        self.num_demos = int(num_demos)
        if self.num_demos <= 0:
            raise ValueError(f"num_demos must be >= 1, got {self.num_demos}")
        self.grid_tokens = self.output_grid_size * self.output_grid_size  # number of tokens we predict
        self._grid_size_max = max(self.grid_size, self.output_grid_size)  # for embeddings / rel_pos

        self.pos_encoding = str(pos_encoding).lower()
        if self.pos_encoding not in {"2d", "1d"}:
            raise ValueError(f"pos_encoding must be one of {{'2d','1d'}}, got {pos_encoding!r}")

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.global_pos_enc = nn.Parameter(torch.randn(1, int(max_len), embed_dim) * 0.02)

        self._N_TOKEN_TYPES = 4
        self.token_type_embed = nn.Embedding(int(self._N_TOKEN_TYPES), embed_dim)
        self.demo_id_embed = nn.Embedding(int(self.num_demos + 1), embed_dim)

        if self.pos_encoding == "2d":
            self.row_embed = nn.Embedding(self._grid_size_max, embed_dim)
            self.col_embed = nn.Embedding(self._grid_size_max, embed_dim)

        self.rel_pos_bias_2d = bool(rel_pos_bias_2d)
        if self.rel_pos_bias_2d:
            self.transformer_rel = EncoderRelPos2D(
                num_layers=int(num_layers),
                embed_dim=int(embed_dim),
                num_heads=int(num_heads),
                ff_dim=int(ff_dim),
                dropout=float(dropout),
                input_grid_size=self.grid_size,
                output_grid_size=self.output_grid_size,
                demo_rel_pos_bias_2d=bool(demo_rel_pos_bias_2d),
            )
            self.transformer = None
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                batch_first=True,
                dropout=dropout,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.transformer_rel = None
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, *, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T)
        _b, t = x.shape
        if t > int(self.global_pos_enc.shape[1]):
            raise ValueError(
                f"Sequence too long: t={t} > max_len={int(self.global_pos_enc.shape[1])}. "
                "Increase max_len."
            )

        emb = self.embed(x) + self.global_pos_enc[:, :t, :]
        if key_padding_mask is not None:
            if key_padding_mask.shape != x.shape:
                raise ValueError(f"key_padding_mask must have shape {tuple(x.shape)}, got {tuple(key_padding_mask.shape)}")
            emb = emb.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        row, col, is_sep = _prompt_rows_cols(
            t=int(t),
            input_grid_size=int(self.grid_size),
            output_grid_size=int(self.output_grid_size),
            num_demos=int(self.num_demos),
            device=x.device,
        )
        demo_row, demo_col, demo_id = _prompt_demo_rows_cols(
            t=int(t),
            input_grid_size=int(self.grid_size),
            output_grid_size=int(self.output_grid_size),
            num_demos=int(self.num_demos),
            device=x.device,
        )
        token_type = _prompt_token_types(
            t=int(t),
            input_grid_size=int(self.grid_size),
            output_grid_size=int(self.output_grid_size),
            num_demos=int(self.num_demos),
            device=x.device,
        )

        # Add role + demo id embeddings.
        # - demo_id is -1 on SEP tokens; we clamp for indexing and then explicitly zero-out SEP contributions.
        role_emb = self.token_type_embed(token_type).unsqueeze(0)  # (1, T, D)
        did = demo_id.clamp(min=0, max=int(self.num_demos)).to(torch.long)
        did_emb = self.demo_id_embed(did).masked_fill(is_sep.unsqueeze(-1), 0.0).unsqueeze(0)  # (1, T, D)
        emb = emb + role_emb + did_emb

        if self.pos_encoding == "2d" and (not bool(self.rel_pos_bias_2d)):
            pos_emb_2d = self.row_embed(row) + self.col_embed(col)  # (T, D)
            pos_emb_2d = pos_emb_2d.masked_fill(is_sep.unsqueeze(-1), 0.0)
            emb = emb + pos_emb_2d.unsqueeze(0)  # (B, T, D)

        if self.rel_pos_bias_2d:
            assert self.transformer_rel is not None
            h = self.transformer_rel(
                emb,
                row=row,
                col=col,
                demo_row=demo_row,
                demo_col=demo_col,
                demo_id=demo_id,
                is_sep=is_sep,
            )
        else:
            assert self.transformer is not None
            if key_padding_mask is not None:
                h = self.transformer(emb, src_key_padding_mask=key_padding_mask)
            else:
                h = self.transformer(emb)
        return self.fc_out(h)  # (B, T, vocab)


def main(
    data_dir: Path | list[Path] = Path("tmp"),
    grid_size: int = 0,
    num_demos: int = 0,
    max_seq_len: int = 500,
    pos_encoding: str = "2d",
    rel_pos_bias_2d: bool = True,
    demo_rel_pos_bias_2d: bool = True,
    pretrained: Optional[Path] = None,
    gradual_unfreeze_new_layers: bool = False,
    gradual_unfreeze_steps: int = 1000,
    train_skills: Optional[list[int]] = None,
    delay_train_skill: Optional[int] = None,
    delay_train_until_step: int = 0,
    delay_train_skills: Optional[list[int]] = None,
    delay_train_until_steps: Optional[list[int]] = None,
    probe_skill: int = 8,
    cap_train_skill3: Optional[int] = None,
    cap_train_skill: Optional[int] = None,
    cap_train_n: Optional[int] = None,
    train_with_ood_skills: Optional[list[int]] = None,
    ood_train_frac: float = 0.5,
    test_frac: float = 0.2,
    steps: int = 3000,
    batch_size: int = 32,
    lr: float = 5e-4,
    lr_decay: str = "cosine",
    min_lr: float = 0.0,
    warmup_steps: int = 2000,
    weight_decay: float = 0.01, 
    seed: int = 0,
    device: str = "cuda", # if torch.cuda.is_available() else "cpu",
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    ff_dim: int = 256,
    dropout: float = 0.0,
    eval_every: int = 500,
    save_every: int = 500,
    eval_tasks: int = 128,
    eval_batch_size: int = 256,
    plot_unsolved_n: int = 3,
    plot_solved_n: int = 3,
    plot_augmented_n: int = 3,
    print_solved_n: int = 0,
    progress: bool = False,
    out_dir: Path = Path("arc_train_runs"),
    no_plots: bool = False,
    dataset_device: str = "gpu",
    aug: bool = True,
    aug_geom_prob: float = 1.0,
    aug_color_prob: float = 1.0,
    aug_translate_prob: float = 1.0,
    aug_translate_max: int = -1,
    aug_keep_background: bool = True,
    eval_vote_augs: int = 0,
    run_tests: bool = True,
) -> None:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    if bool(run_tests):
        run_unit_tests(test_paths=[Path(__file__).resolve().with_name("test_arc_aug.py")])

    grid_size_arg = int(grid_size)
    num_demos_arg = int(num_demos)
    max_seq_len_i = int(max_seq_len)
    if grid_size_arg < 0:
        raise ValueError(f"grid_size must be >= 0 (0=infer), got {grid_size_arg}")
    if num_demos_arg < 0:
        raise ValueError(f"num_demos must be >= 0 (0=infer), got {num_demos_arg}")
    if max_seq_len_i < 0:
        raise ValueError(f"max_seq_len must be >= 0 (0=disable), got {max_seq_len_i}")
    device = torch.device(device)
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    plots_enabled = not bool(no_plots)
    plot_unsolved_n_i = int(plot_unsolved_n)
    if plot_unsolved_n_i < 0:
        raise ValueError(f"plot_unsolved_n must be >= 0, got {plot_unsolved_n_i}")
    plot_solved_n_i = int(plot_solved_n)
    if plot_solved_n_i < 0:
        raise ValueError(f"plot_solved_n must be >= 0, got {plot_solved_n_i}")
    plot_augmented_n_i = int(plot_augmented_n)
    if plot_augmented_n_i < 0:
        raise ValueError(f"plot_augmented_n must be >= 0, got {plot_augmented_n_i}")

    if device.type == "cuda":
        # More throughput on Ampere+; safe for this kind of toy transformer.
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    train_skills = list(DEFAULT_TRAIN_SKILLS) if train_skills is None else [int(s) for s in train_skills]
    for sid in train_skills:
        if sid < 1:
            raise ValueError(f"Invalid skill id: {sid}")

    # Delayed introduction schedule: skill -> step at which it becomes available (included when step >= until_step).
    # Back-compat: allow the old single-skill flags, but prefer the multi-skill form.
    delay_until_by_skill: dict[int, int] = {}
    if (delay_train_skills is not None) or (delay_train_until_steps is not None):
        if (delay_train_skill is not None) or (int(delay_train_until_step) != 0):
            raise ValueError(
                "Use either (--delay_train_skill, --delay_train_until_step) OR "
                "(--delay_train_skills, --delay_train_until_steps), not both."
            )
        skills_list = [] if delay_train_skills is None else [int(s) for s in delay_train_skills]
        until_list = [] if delay_train_until_steps is None else [int(s) for s in delay_train_until_steps]
        if len(skills_list) != len(until_list):
            raise ValueError(
                f"--delay_train_skills and --delay_train_until_steps must have the same length, "
                f"got {len(skills_list)} vs {len(until_list)}"
            )
        for sid, until in zip(skills_list, until_list):
            if until < 0:
                raise ValueError(f"delay step must be >= 0, got {until} for skill {sid}")
            if sid not in train_skills:
                raise ValueError(f"Delayed skill {sid} must be included in train_skills")
            if until > 0:
                delay_until_by_skill[sid] = int(until)
    else:
        if delay_train_until_step < 0:
            raise ValueError(f"delay_train_until_step must be >= 0, got {delay_train_until_step}")
        if delay_train_skill is not None:
            delay_train_skill = int(delay_train_skill)
            if delay_train_skill not in train_skills:
                raise ValueError(f"delay_train_skill={delay_train_skill} must be included in train_skills")
            if int(delay_train_until_step) > 0:
                delay_until_by_skill[int(delay_train_skill)] = int(delay_train_until_step)

    probe_skill = int(probe_skill)
    if probe_skill < 1:
        raise ValueError(f"Invalid --probe_skill: {probe_skill}")

    test_frac_f = float(test_frac)
    if not (0.0 <= test_frac_f < 1.0):
        raise ValueError(f"--test_frac must be in [0,1), got {test_frac_f}")
    if test_frac_f > 0.5:
        raise ValueError(f"--test_frac too large ({test_frac_f}); pick <= 0.5 for a meaningful train split.")
    train_frac_f = 1.0 - test_frac_f

    # Back-compat: if --cap_train_skill3 is set and the generalized flags aren't, map it.
    cap_skill: Optional[int] = int(cap_train_skill) if cap_train_skill is not None else None
    cap_n: Optional[int] = int(cap_train_n) if cap_train_n is not None else None
    if cap_train_skill3 is not None and cap_skill is None:
        cap_skill = 3
        cap_n = int(cap_train_skill3)
    if cap_skill is not None and cap_n is None:
        raise ValueError("--cap_train_n must be provided when --cap_train_skill is set.")

    # Normalize to a list of dataset roots.
    data_dirs: list[Path] = [Path(p).expanduser().resolve() for p in (data_dir if isinstance(data_dir, list) else [data_dir])]

    # Load datasets from disk (no on-the-fly generation) and do a deterministic train/test split.
    #
    # Important: reported accuracies are computed on the held-out "test" portions to avoid leakage.
    train_sets: dict[int, TensorizedDataset] = {}
    eval_id_sets: dict[int, TensorizedDataset] = {}
    ood_train_pools: dict[int, TensorizedDataset] = {}
    eval_ood_sets: dict[int, TensorizedDataset] = {}

    ext = maybe_load_external_arc_splits(
        data_dirs=data_dirs,
        grid_size=int(grid_size_arg),
        num_demos=int(num_demos_arg),
        rng=rng,
        train_frac_for_unsplit=float(train_frac_f),
        max_seq_len=int(max_seq_len_i) if int(max_seq_len_i) > 0 else None,
    )
    external_mode = ext is not None

    if external_mode:
        assert ext is not None
        # Assign stable synthetic "skill ids" 1..N so existing training code (which expects sid>=1) works.
        train_skills = list(range(1, int(len(ext)) + 1))
        for i, (name, tr, ev) in enumerate(ext, start=1):
            out_g_tr = tr.effective_output_grid_size()
            out_g_ev = ev.effective_output_grid_size()
            train_sets[i] = TensorizedDataset(
                skill_id=int(i),
                split=f"train[{name}]",
                grid_size=tr.grid_size,
                num_demos=tr.num_demos,
                src_list=tr.src_list,
                tgt_list=tr.tgt_list,
                grid_size_each=tr.grid_size_each,
                num_demos_each=tr.num_demos_each,
                output_grid_size=out_g_tr,
            )
            eval_id_sets[i] = TensorizedDataset(
                skill_id=int(i),
                split=f"test[{name}]",
                grid_size=ev.grid_size,
                num_demos=ev.num_demos,
                src_list=ev.src_list,
                tgt_list=ev.tgt_list,
                grid_size_each=ev.grid_size_each,
                num_demos_each=ev.num_demos_each,
                output_grid_size=out_g_ev,
            )
        ood_train_pools = {}
        eval_ood_sets = {}
        if grid_size_arg == 0:
            grid_size_arg = max(int(ds.grid_size) for ds in list(train_sets.values()) + list(eval_id_sets.values()))
        if num_demos_arg == 0:
            num_demos_arg = max(int(ds.num_demos) for ds in list(train_sets.values()) + list(eval_id_sets.values()))
    else:
        # Load full splits first so we can infer *max* grid_size/num_demos across skills.
        train_full: dict[int, TensorizedDataset] = {}
        ood_full: dict[int, TensorizedDataset] = {}
        for sid in train_skills:
            train_full[sid] = load_skill_split(
                data_dir=data_dirs,
                skill_id=sid,
                split="train",
                max_seq_len=int(max_seq_len_i) if int(max_seq_len_i) > 0 else None,
            )
            ood_full[sid] = load_skill_split(
                data_dir=data_dirs,
                skill_id=sid,
                split="ood",
                max_seq_len=int(max_seq_len_i) if int(max_seq_len_i) > 0 else None,
            )

        max_g_loaded = max(int(ds.grid_size) for ds in train_full.values())
        max_nd_loaded = max(int(ds.num_demos) for ds in train_full.values())
        max_out_g_loaded = max(
            int(ds.effective_output_grid_size()) for ds in list(train_full.values()) + list(ood_full.values())
        )
        # OOD can have larger maxima too; include it.
        max_g_loaded = max(int(max_g_loaded), max(int(ds.grid_size) for ds in ood_full.values()))
        max_nd_loaded = max(int(max_nd_loaded), max(int(ds.num_demos) for ds in ood_full.values()))

        if grid_size_arg == 0:
            grid_size_arg = int(max_g_loaded)
        if num_demos_arg == 0:
            num_demos_arg = int(max_nd_loaded)
        if int(grid_size_arg) < int(max_g_loaded):
            raise ValueError(f"--grid_size={int(grid_size_arg)} is smaller than dataset max grid_size={int(max_g_loaded)}")
        if int(num_demos_arg) < int(max_nd_loaded):
            raise ValueError(f"--num_demos={int(num_demos_arg)} is smaller than dataset max num_demos={int(max_nd_loaded)}")
        if int(max_seq_len_i) > 0:
            target_seq_len = int(
                prompt_seq_len(
                    grid_size=int(grid_size_arg),
                    num_demos=int(num_demos_arg),
                    output_grid_size=int(max_out_g_loaded),
                )
            )
            if int(target_seq_len) > int(max_seq_len_i):
                raise ValueError(
                    f"Chosen token budget seq_len={int(target_seq_len)} exceeds max_seq_len={int(max_seq_len_i)}. "
                    "Reduce --grid_size/--num_demos or increase --max_seq_len (or set it to 0 to disable)."
                )

        # Normalize all datasets to the chosen maxima before splitting.
        for sid in train_skills:
            train_full[sid] = pad_dataset_to(
                train_full[sid],
                grid_size=int(grid_size_arg),
                num_demos=int(num_demos_arg),
                output_grid_size=int(max_out_g_loaded),
            )
            ood_full[sid] = pad_dataset_to(
                ood_full[sid],
                grid_size=int(grid_size_arg),
                num_demos=int(num_demos_arg),
                output_grid_size=int(max_out_g_loaded),
            )

        for sid in train_skills:
            ds_train, ds_train_test = split_dataset(train_full[sid], train_frac=train_frac_f, rng=rng)
            train_sets[sid] = ds_train
            eval_id_sets[sid] = ds_train_test
            assert_disjoint_datasets(a=ds_train, b=ds_train_test, label=f"skill_{sid}: id train vs id heldout")

            ds_ood_train, ds_ood_test = split_dataset(ood_full[sid], train_frac=train_frac_f, rng=rng)
            ood_train_pools[sid] = ds_ood_train
            eval_ood_sets[sid] = ds_ood_test
            assert_disjoint_datasets(a=ds_ood_train, b=ds_ood_test, label=f"skill_{sid}: ood train vs ood heldout")

    if grid_size_arg <= 0:
        raise ValueError("Could not infer grid_size (no training datasets loaded).")
    if num_demos_arg <= 0:
        raise ValueError("Could not infer num_demos (no training datasets loaded).")

    grid_size = int(grid_size_arg)
    num_demos = int(num_demos_arg)
    ds_for_out = list(train_sets.values()) + list(eval_id_sets.values()) + list(eval_ood_sets.values())
    output_grid_size = max(
        (int(ds.effective_output_grid_size()) for ds in ds_for_out),
        default=int(grid_size),
    )
    grid_tokens = int(output_grid_size) * int(output_grid_size)
    # With per-batch padding, seq_len is the model's max context (each example already filtered in tensorize).
    seq_len = int(max_seq_len_i) if int(max_seq_len_i) > 0 else 8192

    # Include OOD examples for selected skills in training (synthetic mode only).
    # OOD test remains disjoint and is what we report in the printed metrics.
    if not external_mode:
        if train_with_ood_skills is None:
            train_with_ood_skills = list(DEFAULT_TRAIN_WITH_OOD_SKILLS)
        train_with_ood = {int(s) for s in train_with_ood_skills} & set(train_skills)
        # Never include probe skill OOD in training (strict OOD probe).
        train_with_ood.discard(probe_skill)
        for sid in sorted(train_with_ood):
            ood_pool = ood_train_pools[sid]
            ood_train = ood_pool
            if float(ood_train_frac) < 1.0:
                ood_train, _ood_unused = split_dataset(ood_pool, train_frac=float(ood_train_frac), rng=rng)
            if ood_train.n > 0:
                train_sets[sid] = concat_datasets(
                    [train_sets[sid], ood_train],
                    skill_id=sid,
                    split=f"train+ood{sid}",
                    grid_size=grid_size,
                )
    else:
        train_with_ood = set()

    # Optional artificial cap for any skill: reduce training data to force learning only when possible.
    if cap_skill is not None and cap_skill in train_sets:
        train_sets[cap_skill] = cap_dataset(train_sets[cap_skill], cap=int(cap_n), rng=rng)

    # Always report the strict OOD probe (held-out OOD test), even if probe_skill is not in train_skills.
    # (Synthetic mode only.)
    probe_ood_full = (
        None
        if external_mode
        else maybe_load_skill_split(
            data_dir=data_dirs,
            skill_id=probe_skill,
            split="ood",
            max_seq_len=int(max_seq_len_i) if int(max_seq_len_i) > 0 else None,
        )
    )
    eval_probe_ood = None
    probe_ood_train: Optional[TensorizedDataset] = None
    if probe_ood_full is not None:
        out_g_probe = probe_ood_full.effective_output_grid_size()
        output_grid_size = max(int(output_grid_size), int(out_g_probe))
        grid_tokens = int(output_grid_size) * int(output_grid_size)
        probe_ood_full = pad_dataset_to(
            probe_ood_full,
            grid_size=int(grid_size),
            num_demos=int(num_demos),
            output_grid_size=int(output_grid_size),
        )
        probe_ood_train, probe_ood_test = split_dataset(probe_ood_full, train_frac=train_frac_f, rng=rng)
        eval_probe_ood = probe_ood_test
        assert_disjoint_datasets(
            a=probe_ood_train,
            b=probe_ood_test,
            label=f"probe_skill_{probe_skill}: ood train vs ood heldout",
        )

    # Sanity: dataset maxima must not exceed chosen grid_size/num_demos (with per-batch padding they may be smaller).
    ds_to_check = list(train_sets.values()) + list(eval_id_sets.values()) + list(eval_ood_sets.values())
    if eval_probe_ood is not None:
        ds_to_check.append(eval_probe_ood)
    for ds in ds_to_check:
        if int(ds.grid_size) > int(grid_size):
            raise ValueError(f"Dataset grid_size={ds.grid_size} > chosen grid_size={grid_size}")
        if int(ds.effective_output_grid_size()) > int(output_grid_size):
            raise ValueError(
                f"Dataset output_grid_size={ds.effective_output_grid_size()} > chosen output_grid_size={output_grid_size}"
            )
        if int(ds.num_demos) > int(num_demos):
            raise ValueError(f"Dataset num_demos={ds.num_demos} > chosen num_demos={num_demos}")

    # Build mixed training pools (variable-length; padding is per-batch).
    train_pool_all = concat_datasets(
        [train_sets[sid] for sid in train_skills],
        skill_id=-1,
        split="train_mix",
        grid_size=grid_size,
    )

    def build_pool(active_skills: list[int], *, split: str) -> TensorizedDataset:
        return concat_datasets(
            [train_sets[sid] for sid in active_skills],
            skill_id=-1,
            split=split,
            grid_size=grid_size,
        )

    # Precompute phase pools keyed by the step at which that pool becomes active.
    phase_starts = [0]
    if len(delay_until_by_skill) > 0:
        phase_starts += sorted(set(int(v) for v in delay_until_by_skill.values() if int(v) > 0))

    train_pool_phases: list[tuple[int, TensorizedDataset]] = []
    for start in phase_starts:
        active_skills = [sid for sid in train_skills if int(delay_until_by_skill.get(sid, 0)) <= int(start)]
        if len(active_skills) == 0:
            raise ValueError("Delaying all training skills at step 0 would leave an empty training pool.")
        if len(active_skills) == len(train_skills):
            pool = train_pool_all
        else:
            excluded = [sid for sid in train_skills if sid not in active_skills]
            excluded_s = "_".join(f"s{sid}" for sid in excluded)
            pool = build_pool(active_skills, split=f"train_mix_excl_{excluded_s}_from_{int(start)}")
        train_pool_phases.append((int(start), pool))

    # Ensure training pools won't bottleneck on CPU (avoid moving duplicate references twice).
    moved_cache: dict[int, TensorizedDataset] = {}

    def move_pool(pool: TensorizedDataset) -> TensorizedDataset:
        k = id(pool)
        if k not in moved_cache:
            moved_cache[k] = maybe_move_train_pool(pool, device=device, dataset_device=str(dataset_device))
        return moved_cache[k]

    train_pool_phases = [(start, move_pool(pool)) for start, pool in train_pool_phases]

    model = ARCTransformer(
        vocab_size=VOCAB_SIZE,
        grid_size=grid_size,
        num_demos=int(num_demos),
        output_grid_size=int(output_grid_size),
        pos_encoding=str(pos_encoding),
        rel_pos_bias_2d=bool(rel_pos_bias_2d),
        demo_rel_pos_bias_2d=bool(demo_rel_pos_bias_2d),
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
        num_layers=int(num_layers),
        ff_dim=int(ff_dim),
        max_len=seq_len,
        dropout=float(dropout),
    ).to(device)

    # Optional: identify params that were NOT loaded from the pretrained checkpoint (new/random-init).
    # When enabled, we temporarily run those "new" params at lr/100 for the first K steps.
    NEW_LAYER_LR_MULT = 0.01
    new_param_names: set[str] = set()

    if pretrained is not None:
        report = load_pretrained_weights(model, pretrained)
        print(
            "Loaded pretrained weights:"
            f" loaded={report.loaded}"
            f" skipped_unexpected={report.skipped_unexpected}"
            f" skipped_shape_mismatch={report.skipped_shape_mismatch}"
            f" missing_after_load={report.missing_after_load}",
            flush=True,
        )
        if bool(gradual_unfreeze_new_layers):
            if int(gradual_unfreeze_steps) <= 0:
                raise ValueError(f"--gradual_unfreeze_steps must be >= 1, got {gradual_unfreeze_steps}")
            new_param_names = set(getattr(report, "missing_keys", ()))

    total_params, trainable_params = count_params(model)
    print(f"Model params: total={total_params:,} trainable={trainable_params:,}")

    # AdamW + fairly high weight decay. We avoid decaying biases and normalization weights.
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    decay_new_params: list[torch.nn.Parameter] = []
    no_decay_new_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_new = bool(new_param_names) and (name in new_param_names)
        name_l = name.lower()
        is_no_decay = bool(p.ndim == 1 or name_l.endswith(".bias") or "norm" in name_l)
        if bool(is_new):
            if bool(is_no_decay):
                no_decay_new_params.append(p)
            else:
                decay_new_params.append(p)
        else:
            if bool(is_no_decay):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

    use_gradual_unfreeze = bool(pretrained is not None) and bool(gradual_unfreeze_new_layers) and (
        (len(decay_new_params) + len(no_decay_new_params)) > 0
    )
    if use_gradual_unfreeze:
        n_new = len(decay_new_params) + len(no_decay_new_params)
        print(
            f"Gradual unfreeze enabled: {n_new:,} 'new' parameters will use lr/100 for the first "
            f"{int(gradual_unfreeze_steps)} steps.",
            flush=True,
        )
        # Group order is important; we reference these indices in the training loop.
        # 0: old/loaded decay, 1: old/loaded no_decay, 2: new decay, 3: new no_decay
        opt = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": float(weight_decay)},
                {"params": no_decay_params, "weight_decay": 0.0},
                {"params": decay_new_params, "weight_decay": float(weight_decay)},
                {"params": no_decay_new_params, "weight_decay": 0.0},
            ],
            lr=float(lr),
        )
        old_decay_idx, old_no_decay_idx, new_decay_idx, new_no_decay_idx = 0, 1, 2, 3
    else:
        opt = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": float(weight_decay)},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=float(lr),
        )
        old_decay_idx = old_no_decay_idx = new_decay_idx = new_no_decay_idx = -1
    lr_decay = str(lr_decay).lower().strip()
    if lr_decay not in {"cosine", "none"}:
        raise ValueError(f"--lr_decay must be one of {{'cosine','none'}}, got {lr_decay!r}")
    if float(min_lr) < 0.0:
        raise ValueError(f"--min_lr must be >= 0, got {min_lr}")
    warmup_steps_i = int(warmup_steps)
    if warmup_steps_i < 0:
        raise ValueError(f"--warmup_steps must be >= 0, got {warmup_steps_i}")
    base_lr = float(lr)
    min_lr_f = float(min_lr)
    total_steps_i = int(steps)

    def lr_at_step(step_i: int) -> float:
        """
        LR schedule:
        - linear warmup for the first `warmup_steps` steps (default on)
        - then either constant LR (--lr_decay=none) or cosine decay to --min_lr
        """
        s = int(step_i) + 1  # 1-indexed
        if warmup_steps_i > 0 and s <= warmup_steps_i:
            return base_lr * (float(s) / float(warmup_steps_i))
        if lr_decay == "none":
            return base_lr
        # cosine
        denom = max(1, int(total_steps_i - warmup_steps_i))
        t = float(max(0, s - warmup_steps_i)) / float(denom)
        if t >= 1.0:
            return min_lr_f
        return min_lr_f + 0.5 * (base_lr - min_lr_f) * (1.0 + math.cos(math.pi * t))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def save_latest_checkpoint(*, step: int) -> None:
        ckpt = {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "seed": int(seed),
            "grid_size": int(grid_size),
            "num_demos": int(num_demos),
            "seq_len": int(seq_len),
            "train_skills": [int(s) for s in train_skills],
            "delay_until_by_skill": {int(k): int(v) for k, v in delay_until_by_skill.items()},
        }
        torch.save(ckpt, out_dir / "checkpoints" / "latest.pt")

    def save_best_val_checkpoint(*, step: int, val_score: float) -> None:
        ckpt = {
            "step": int(step),
            "val_score": float(val_score),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "seed": int(seed),
            "grid_size": int(grid_size),
            "num_demos": int(num_demos),
            "seq_len": int(seq_len),
            "train_skills": [int(s) for s in train_skills],
            "delay_until_by_skill": {int(k): int(v) for k, v in delay_until_by_skill.items()},
        }
        torch.save(ckpt, out_dir / "checkpoints" / "best_val.pt")

    # Baseline training: mixed skills, only ID prompts (no OOD in train; notably no Skill 3 OOD).
    model.train()
    best_val = float("-inf")
    curves = LearningCurves(
        steps=[],
        loss=[],
        acc_train={},
        acc_id={},
        acc_ood={},
        probe_train_ood=[],
        probe_fully_heldout_ood=[],
    )
    gen_cpu = torch.Generator().manual_seed(int(seed))
    aug_spec = AugmentSpec(
        enabled=bool(aug),
        geom_prob=float(aug_geom_prob),
        color_prob=float(aug_color_prob),
        translate_prob=float(aug_translate_prob),
        translate_max=int(aug_translate_max),
        keep_background=bool(aug_keep_background),
    )
    vote_spec = (
        AugmentSpec(
            enabled=True,
            geom_prob=1.0,
            color_prob=1.0,
            translate_prob=1.0,
            translate_max=int(aug_translate_max),
            keep_background=bool(aug_keep_background),
        )
        if int(eval_vote_augs) > 0
        else None
    )
    steps_iter = progress_iter(range(int(steps)), total=int(steps), desc="train", enabled=bool(progress))
    phase_idx = 0
    for step in steps_iter:
        while (phase_idx + 1) < len(train_pool_phases) and int(step) >= int(train_pool_phases[phase_idx + 1][0]):
            phase_idx += 1
        active_pool = train_pool_phases[phase_idx][1]
        batch = prepare_batch(
            batch_size=int(batch_size),
            train_pool=active_pool,
            device=device,
            cpu_generator=gen_cpu,
            augment=aug_spec if bool(aug_spec.enabled) else None,
            grid_size=int(grid_size),
            num_demos=int(num_demos),
            T_max=seq_len,
            G_max=grid_tokens,
        )
        src = batch.src
        tgt = batch.tgt  # (B, grid_tokens)

        # Update learning rates (warmup + optional decay). Applied to all param groups.
        lr_step = float(lr_at_step(int(step)))
        if use_gradual_unfreeze:
            mult = float(NEW_LAYER_LR_MULT) if int(step) < int(gradual_unfreeze_steps) else 1.0
            opt.param_groups[int(old_decay_idx)]["lr"] = float(lr_step)
            opt.param_groups[int(old_no_decay_idx)]["lr"] = float(lr_step)
            opt.param_groups[int(new_decay_idx)]["lr"] = float(lr_step) * float(mult)
            opt.param_groups[int(new_no_decay_idx)]["lr"] = float(lr_step) * float(mult)
        else:
            for pg in opt.param_groups:
                pg["lr"] = float(lr_step)

        opt.zero_grad(set_to_none=True)
        logits = model(src, key_padding_mask=batch.key_padding_mask)  # (B, T, V)
        grid_tokens_batch = int(tgt.shape[1])
        pred_logits = logits[:, -(grid_tokens_batch + 1) : -1, :]  # predict from test-x positions

        loss = loss_fn(pred_logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        loss.backward()
        opt.step()

        if (int(save_every) > 0) and ((step % int(save_every) == 0) or (step == int(steps) - 1)):
            save_latest_checkpoint(step=int(step))

        do_eval = (int(eval_every) > 0) and ((step % int(eval_every) == 0) or (step == int(steps) - 1))
        if do_eval:
            model.eval()
            eval_ids = sorted(eval_id_sets.keys())
            print_solved_n_i = int(print_solved_n)
            acc_train = {}
            acc_id = {}
            acc_ood = {}
            # Save unsolved examples at every eval (files are keyed by step+idx so they won't clobber).
            unsolved_dir = (out_dir / "plots" / "unsolved_examples") if int(plot_unsolved_n_i) > 0 else None
            if unsolved_dir is not None:
                unsolved_dir.mkdir(parents=True, exist_ok=True)
            solved_dir = (out_dir / "plots" / "solved_examples") if int(plot_solved_n_i) > 0 else None
            if solved_dir is not None:
                solved_dir.mkdir(parents=True, exist_ok=True)
            augmented_dir = (out_dir / "plots" / "augmented_examples") if int(plot_augmented_n_i) > 0 else None
            if augmented_dir is not None:
                augmented_dir.mkdir(parents=True, exist_ok=True)

            first_sid = int(eval_ids[0]) if (len(eval_ids) > 0) else -1
            for sid in eval_ids:
                acc_train[sid] = evaluate_accuracy(
                    model=model,
                    rng=rng,
                    n_tasks=int(eval_tasks),
                    device=device,
                    grid_tokens=grid_tokens,
                    dataset=train_sets[sid],
                    eval_batch_size=int(eval_batch_size),
                    vote_augs=int(eval_vote_augs),
                    vote_spec=vote_spec,
                    show_progress=bool(progress),
                )
                acc_id[sid] = evaluate_accuracy(
                    model=model,
                    rng=rng,
                    n_tasks=int(eval_tasks),
                    device=device,
                    grid_tokens=grid_tokens,
                    dataset=eval_id_sets[sid],
                    eval_batch_size=int(eval_batch_size),
                    vote_augs=int(eval_vote_augs),
                    vote_spec=vote_spec,
                    save_unsolved_dir=unsolved_dir,
                    save_unsolved_max=int(plot_unsolved_n_i),
                    save_unsolved_step=int(step),
                    save_unsolved_tag="id",
                    save_solved_dir=solved_dir,
                    save_solved_max=int(plot_solved_n_i),
                    save_solved_step=int(step),
                    save_solved_tag="id",
                    save_augmented_dir=augmented_dir,
                    save_augmented_max=int(plot_augmented_n_i),
                    save_augmented_step=int(step),
                    save_augmented_tag="id",
                    save_augmented_spec=aug_spec if bool(aug_spec.enabled) else None,
                    print_solved_max=int(print_solved_n_i) if int(sid) == int(first_sid) else 0,
                    print_solved_step=int(step),
                    print_solved_tag="id",
                    show_progress=bool(progress),
                )
                if sid in eval_ood_sets:
                    acc_ood[sid] = evaluate_accuracy(
                        model=model,
                        rng=rng,
                        n_tasks=int(eval_tasks),
                        device=device,
                        grid_tokens=grid_tokens,
                        dataset=eval_ood_sets[sid],
                        eval_batch_size=int(eval_batch_size),
                        vote_augs=int(eval_vote_augs),
                        vote_spec=vote_spec,
                        save_unsolved_dir=unsolved_dir,
                        save_unsolved_max=int(plot_unsolved_n_i),
                        save_unsolved_step=int(step),
                        save_unsolved_tag="ood",
                        save_solved_dir=solved_dir,
                        save_solved_max=int(plot_solved_n_i),
                        save_solved_step=int(step),
                        save_solved_tag="ood",
                        save_augmented_dir=augmented_dir,
                        save_augmented_max=int(plot_augmented_n_i),
                        save_augmented_step=int(step),
                        save_augmented_tag="ood",
                        save_augmented_spec=aug_spec if bool(aug_spec.enabled) else None,
                        show_progress=bool(progress),
                    )

            # Optional strict OOD probe (held-out OOD test).
            acc_probe_ood = float("nan")
            if eval_probe_ood is not None and eval_probe_ood.n > 0:
                acc_probe_ood = evaluate_accuracy(
                    model=model,
                    rng=rng,
                    n_tasks=int(eval_tasks),
                    device=device,
                    grid_tokens=grid_tokens,
                    dataset=eval_probe_ood,
                    eval_batch_size=int(eval_batch_size),
                    vote_augs=int(eval_vote_augs),
                    vote_spec=vote_spec,
                    show_progress=bool(progress),
                )

            acc_probe_train = float("nan")
            if probe_ood_train is not None and probe_ood_train.n > 0:
                acc_probe_train = evaluate_accuracy(
                    model=model,
                    rng=rng,
                    n_tasks=int(eval_tasks),
                    device=device,
                    grid_tokens=grid_tokens,
                    dataset=probe_ood_train,
                    eval_batch_size=int(eval_batch_size),
                    vote_augs=int(eval_vote_augs),
                    vote_spec=vote_spec,
                    show_progress=bool(progress),
                )

            def fmt(acc: dict[int, float], skills: list[int]) -> str:
                return " ".join(f"s{sid}={acc.get(sid, float('nan')):.3f}" for sid in skills)

            print(f"step={step:5d} loss={loss.item():.4f}")
            print(f"  trn: {fmt(acc_train, eval_ids)}")
            if external_mode:
                print(f"  tst: {fmt(acc_id, eval_ids)}")
            else:
                print(f"  id : {fmt(acc_id, eval_ids)}")
            if len(eval_ood_sets) > 0:
                ood_line = f"  ood: {fmt(acc_ood, eval_ids)}"
                if eval_probe_ood is not None:
                    ood_line += (
                        f"  (probe: s{probe_skill} train-ood={acc_probe_train:.3f} fully-heldout-ood={acc_probe_ood:.3f})"
                    )
                print(ood_line)
            if (lr_decay != "none") or (warmup_steps_i > 0) or use_gradual_unfreeze:
                base_lr = float(opt.param_groups[0]["lr"])
                if use_gradual_unfreeze:
                    new_lr = float(opt.param_groups[2]["lr"])
                    print(f"  lr : base={base_lr:.6g} new={new_lr:.6g}")
                else:
                    print(f"  lr : {base_lr:.6g}")

            # Track and plot learning curves
            curves.steps.append(int(step))
            curves.loss.append(float(loss.item()))
            curves.probe_train_ood.append(float(acc_probe_train))
            curves.probe_fully_heldout_ood.append(float(acc_probe_ood))
            for sid in eval_ids:
                curves.ensure_skill(sid)
                curves.acc_train[sid].append(float(acc_train[sid]))
                curves.acc_id[sid].append(float(acc_id[sid]))
                curves.acc_ood[sid].append(float(acc_ood.get(sid, float("nan"))))

            # Save metrics CSV next to the plot output (even if --no_plots).
            metrics_csv = out_dir / "plots" / "learning_curves_latest.csv"
            write_learning_curves_csv(curves=curves, skills=sorted(eval_ids), out_path=metrics_csv)

            # "Val" model selection: mean ID accuracy across eval_id splits.
            if len(eval_ids) > 0:
                val_score = float(sum(float(acc_id[sid]) for sid in eval_ids)) / float(len(eval_ids))
                if val_score > best_val:
                    best_val = val_score
                    save_best_val_checkpoint(step=int(step), val_score=float(val_score))

            if plots_enabled:
                delay_s = (
                    "none"
                    if len(delay_until_by_skill) == 0
                    else " ".join(f"s{sid}@{until}" for sid, until in sorted(delay_until_by_skill.items()))
                )
                title = (
                    "ARC skill learning curves (exact-match acc)\n"
                    f"train_skills={train_skills} | ood_in_train={sorted(train_with_ood)} | "
                    f"probe_skill={probe_skill} | cap_skill={cap_skill}:{cap_n} | "
                    f"delay_skills={delay_s} | eval_tasks={int(eval_tasks)}"
                )
                latest = out_dir / "plots" / "learning_curves_latest.png"
                plot_learning_curves(curves=curves, skills=sorted(eval_ids), out_path=latest, title=title)

            model.train()

    # Qualitative examples
    model.eval()
    if len(eval_id_sets) > 0:
        sid0 = sorted(eval_id_sets.keys())[0]
        show_one_example(model=model, dataset=eval_id_sets[sid0], device=device, grid_size=grid_size)
    if eval_probe_ood is not None:
        show_one_example(model=model, dataset=eval_probe_ood, device=device, grid_size=grid_size)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a simple Transformer on synthetic ARC skills.")
    p.add_argument(
        "--data_dir",
        type=Path,
        nargs="*",
        default=[Path("tmp")],
        help=(
            "One or more dataset roots. "
            "Synthetic format: contains skill_<id>/{train,ood}.json. "
            "ARC-AGI format: contains {training,evaluation}/*.json (ARC task JSONs). "
            "If subfolders exist but are not named training/evaluation, all jsons are loaded and split into train/evaluation."
        ),
    )
    p.add_argument(
        "--grid_size",
        type=int,
        default=0,
        help="Grid size. Use 0 to infer from the loaded dataset/puzzles.",
    )
    p.add_argument(
        "--num_demos",
        type=int,
        default=0,
        help="Number of demonstrations. Use 0 to infer from the loaded dataset/puzzles.",
    )
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=2000,
        help=(
            "Maximum allowed tokenized prompt length T. Any training/eval examples whose prompt would exceed this are dropped. "
            "Also enforces that the final (grid_size,num_demos) token budget fits within this limit. "
            "Use 0 to disable."
        ),
    )
    p.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Optional path to pretrained weights (.pt checkpoint with `model` or a raw state_dict). "
        "Weights are loaded permissively (matching names+shapes); new layers stay randomly initialized.",
    )
    p.add_argument(
        "--gradual_unfreeze_new_layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set AND --pretrained is provided, parameters not loaded from the pretrained checkpoint "
            "(i.e., new/random-init layers) will use lr/100 for the first --gradual_unfreeze_steps steps."
        ),
    )
    p.add_argument(
        "--gradual_unfreeze_steps",
        type=int,
        default=1000,
        help="Number of initial training steps to apply lr/100 to new/random-init parameters (only if --gradual_unfreeze_new_layers).",
    )
    p.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=None,
        help="Alias for --train_skills (skill IDs to load for training/eval). Example: --tasks 14 15",
    )
    p.add_argument(
        "--train_skills",
        type=int,
        nargs="*",
        default=None,
        help=f"Skill IDs to load for training/eval. Default: {list(DEFAULT_TRAIN_SKILLS)}",
    )
    p.add_argument(
        "--phase1_skills",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional 2/3-phase curriculum. Phase 1 trains ONLY on these skills. "
            "Phase 2 adds skills (see --phase2_skills) at --phase2_start_step / --phase2_start_frac. "
            "Phase 3 (optional) adds skills (see --phase3_skills) at --phase3_start_step / --phase3_start_frac. "
            "Mutually exclusive with --delay_train_skills/--delay_train_until_steps."
        ),
    )
    p.add_argument(
        "--phase2_skills",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional curriculum. Phase 2 skill set (joint training pool after the first switch). "
            "Must include all phase1_skills. If omitted, defaults to phase1_skills (no-op)."
        ),
    )
    p.add_argument(
        "--phase2_start_step",
        type=int,
        default=None,
        help="Optional curriculum: step at which to start Phase 2 (adding phase2-only skills).",
    )
    p.add_argument(
        "--phase2_start_frac",
        type=float,
        default=None,
        help=(
            "Optional curriculum: fraction of total --steps at which to start Phase 2. "
            "Example: 0.5 means switch at half the steps. Ignored if --phase2_start_step is set."
        ),
    )
    p.add_argument(
        "--phase3_skills",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional 3-phase curriculum. Phase 3 skill set (joint training pool after the second switch). "
            "Must include all phase2_skills. If omitted, Phase 3 is disabled."
        ),
    )
    p.add_argument(
        "--phase3_start_step",
        type=int,
        default=None,
        help="Optional 3-phase curriculum: step at which to start Phase 3 (adding phase3-only skills).",
    )
    p.add_argument(
        "--phase3_start_frac",
        type=float,
        default=None,
        help=(
            "Optional 3-phase curriculum: fraction of total --steps at which to start Phase 3. "
            "Example: 0.75 means switch at 75% of the steps. Ignored if --phase3_start_step is set."
        ),
    )
    p.add_argument(
        "--delay_train_skill",
        "--delay_train_skills",
        dest="delay_train_skills",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Exclude these skills from the mixed training pool initially (hard switch). "
            "Must be paired 1:1 with --delay_train_until_step/--delay_train_until_steps. "
            "Example: --delay_train_skills 13 14 --delay_train_until_steps 1000 5000"
        ),
    )
    p.add_argument(
        "--delay_train_until_step",
        "--delay_train_until_steps",
        dest="delay_train_until_steps",
        type=int,
        nargs="*",
        default=None,
        help=(
            "For each skill in --delay_train_skill/--delay_train_skills, the step at which that skill is introduced. "
            "Example: --delay_train_skills 13 14 --delay_train_until_steps 1000 5000"
        ),
    )
    p.add_argument(
        "--probe_skill",
        type=int,
        default=16,
        help=(
            "A *strict* OOD generalization probe: this skill's OOD split is never used for training, "
            "even if you mix OOD data into training for other skills via --train_with_ood_skills. "
            "Probe accuracy is reported on the held-out OOD test portion (controlled by --test_frac)."
        ),
    )
    p.add_argument(
        "--cap_train_skill3",
        type=int,
        default=None,
        help="If set, limits the number of Skill 3 training tasks (from skill_3/train.json) used in the mixed pool.",
    )
    p.add_argument(
        "--cap_train_skill",
        type=int,
        default=None,
        help="If set, limits the number of training tasks for this skill in the training pool (applies after any OOD mixing).",
    )
    p.add_argument(
        "--cap_train_n",
        type=int,
        default=None,
        help="Number of training tasks to keep for --cap_train_skill.",
    )
    p.add_argument(
        "--train_with_ood_skills",
        type=int,
        nargs="*",
        default=list(DEFAULT_TRAIN_WITH_OOD_SKILLS),
        help=(
            "Skills whose training pool should also include (a subset of) their OOD split. "
            "OOD examples are taken only from the OOD-train portion (OOD-test is always held out via --test_frac). "
            "The --probe_skill OOD split is excluded from this mixing."
        ),
    )
    p.add_argument(
        "--ood_train_frac",
        type=float,
        default=0.0,
        help=(
            "Fraction of the OOD-train portion to include in training when that skill is in --train_with_ood_skills "
            "(the remainder of OOD-train is unused; OOD-test remains held out via --test_frac)."
        ),
    )
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help=(
            "Held-out test fraction for every loaded dataset split (both ID 'train' and OOD 'ood'). "
            "Reported accuracies (id/ood/probe) are computed on these held-out test portions."
        ),
    )
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument(
        "--lr_decay",
        type=str,
        default="cosine",
        choices=["cosine", "none"],
        help="LR schedule. Default is cosine decay over --steps; set 'none' for constant LR.",
    )
    p.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="Minimum LR for cosine decay (eta_min). Ignored when --lr_decay=none.",
    )
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Number of initial steps for linear LR warmup (default on). Set 0 to disable.",
    )
    p.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay (L2).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--pos_encoding",
        type=str,
        default="2d",
        choices=["2d", "1d"],
        help="Positional encoding scheme. '2d' (default) uses row+col embeddings per grid; '1d' uses the old absolute learned positions.",
    )
    p.add_argument(
        "--rel_pos_bias_2d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use learned 2D relative position bias in self-attention (enabled by default). "
            "Disable with --no-rel_pos_bias_2d."
        ),
    )
    p.add_argument(
        "--demo_rel_pos_bias_2d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add an additional learned 2D relative position bias that is relative *within each demonstration* "
            "(treating each demo as an x|gap|y 2D layout). Disable with --no-demo_rel_pos_bias_2d. "
            "Only used when --rel_pos_bias_2d is enabled."
        ),
    )
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="How often to update checkpoints/latest.pt (in steps). Set 0 to disable.",
    )
    p.add_argument("--eval_tasks", type=int, default=128)
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation (bigger is faster; uses more VRAM).",
    )
    p.add_argument(
        "--plot_unsolved_n",
        type=int,
        default=3,
        help="Per-skill number of unsolved test examples to render as PNG during eval (0 disables).",
    )
    p.add_argument(
        "--plot_solved_n",
        type=int,
        default=3,
        help="Per-skill number of solved test examples to render as PNG during eval (0 disables).",
    )
    p.add_argument(
        "--plot_augmented_n",
        type=int,
        default=3,
        help=(
            "Per-skill number of eval examples to render as PNG after applying the *train-time* augmentation pipeline "
            "(0 disables). Saved under plots/augmented_examples/."
        ),
    )
    p.add_argument(
        "--print_solved_n",
        type=int,
        default=0,
        help="Number of solved ID test examples to print (stdout) at each eval (0 disables). Printed for the first eval skill only.",
    )
    p.add_argument("--progress", action="store_true", help="Show tqdm progress if installed")
    p.add_argument("--out_dir", type=Path, default=Path("arc_train_runs"), help="Where to write plots/metrics")
    p.add_argument("--no_plots", action="store_true", help="Disable saving learning-curve PNGs")
    p.add_argument(
        "--dataset_device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Where to keep the training pool tensors. 'gpu' avoids CPU bottlenecks; 'cpu' pins memory for async H2D.",
    )
    p.add_argument(
        "--aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable train-time augmentation (D4 flips/rotations + global color remap). Disable with --no-aug.",
    )
    p.add_argument(
        "--aug_geom_prob",
        type=float,
        default=1.0,
        help="Probability of applying a random D4 geometric transform per sample in a training batch.",
    )
    p.add_argument(
        "--aug_color_prob",
        type=float,
        default=1.0,
        help="Probability of applying a random global color permutation per sample in a training batch.",
    )
    p.add_argument(
        "--aug_translate_prob",
        type=float,
        default=1.0,
        help="Probability of applying a random translation per sample in a training batch.",
    )
    p.add_argument(
        "--aug_translate_max",
        type=int,
        default=-1,
        help="Max absolute translation (cells). -1=auto (max in-bounds based on non-zero bbox).",
    )
    p.add_argument(
        "--aug_keep_background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep background color 0 fixed during color permutation (default). Disable with --no-aug_keep_background.",
    )
    p.add_argument(
        "--eval_vote_augs",
        type=int,
        default=0,
        help="If >0, perform test-time voting over this many random augmentations (slow).",
    )
    p.add_argument(
        "--run_tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run unit tests (pytest) before training starts (default). Disable with --no-run_tests.",
    )
    return p


def cli_main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.tasks is not None and args.train_skills is not None:
        raise ValueError("Use either --tasks or --train_skills (alias), not both.")

    # Resolve the "base" training skill list (before curriculum/delay tweaks).
    train_skills = args.tasks if args.tasks is not None else args.train_skills

    # Optional 2-phase curriculum, compiled down to the existing delay mechanism.
    if args.phase1_skills is not None:
        if (args.delay_train_skills is not None) or (args.delay_train_until_steps is not None):
            raise ValueError("--phase1_skills cannot be combined with --delay_train_skills/--delay_train_until_steps.")
        phase1 = [int(s) for s in args.phase1_skills]
        phase2 = phase1 if args.phase2_skills is None else [int(s) for s in args.phase2_skills]

        if args.phase3_skills is None:
            if args.phase2_start_step is not None:
                phase2_start_step = int(args.phase2_start_step)
            else:
                frac = 0.5 if args.phase2_start_frac is None else float(args.phase2_start_frac)
                if not (0.0 <= frac <= 1.0):
                    raise ValueError(f"--phase2_start_frac must be in [0,1], got {frac}")
                phase2_start_step = int(round(frac * float(int(args.steps))))

            train_skills, delay_skills, delay_steps = _curriculum_delay_from_phases(
                phase1_skills=phase1,
                phase2_skills=phase2,
                phase2_start_step=int(phase2_start_step),
            )
        else:
            phase3 = [int(s) for s in args.phase3_skills]

            if args.phase2_start_step is not None:
                phase2_start_step = int(args.phase2_start_step)
            else:
                frac2 = 0.5 if args.phase2_start_frac is None else float(args.phase2_start_frac)
                if not (0.0 <= frac2 <= 1.0):
                    raise ValueError(f"--phase2_start_frac must be in [0,1], got {frac2}")
                phase2_start_step = int(round(frac2 * float(int(args.steps))))

            if args.phase3_start_step is not None:
                phase3_start_step = int(args.phase3_start_step)
            else:
                frac3 = 0.75 if args.phase3_start_frac is None else float(args.phase3_start_frac)
                if not (0.0 <= frac3 <= 1.0):
                    raise ValueError(f"--phase3_start_frac must be in [0,1], got {frac3}")
                phase3_start_step = int(round(frac3 * float(int(args.steps))))

            train_skills, delay_skills, delay_steps = _curriculum_delay_from_3phases(
                phase1_skills=phase1,
                phase2_skills=phase2,
                phase2_start_step=int(phase2_start_step),
                phase3_skills=phase3,
                phase3_start_step=int(phase3_start_step),
            )
        args.delay_train_skills = delay_skills
        args.delay_train_until_steps = delay_steps
    main(
        data_dir=list(args.data_dir),
        grid_size=int(args.grid_size),
        num_demos=int(args.num_demos),
        max_seq_len=int(args.max_seq_len),
        pos_encoding=str(args.pos_encoding),
        rel_pos_bias_2d=bool(args.rel_pos_bias_2d),
        demo_rel_pos_bias_2d=bool(args.demo_rel_pos_bias_2d),
        pretrained=Path(args.pretrained) if args.pretrained is not None else None,
        gradual_unfreeze_new_layers=bool(args.gradual_unfreeze_new_layers),
        gradual_unfreeze_steps=int(args.gradual_unfreeze_steps),
        train_skills=[int(s) for s in train_skills] if train_skills is not None else None,
        delay_train_skills=[int(s) for s in args.delay_train_skills] if args.delay_train_skills is not None else None,
        delay_train_until_steps=[int(s) for s in args.delay_train_until_steps] if args.delay_train_until_steps is not None else None,
        probe_skill=int(args.probe_skill),
        cap_train_skill3=int(args.cap_train_skill3) if args.cap_train_skill3 is not None else None,
        cap_train_skill=int(args.cap_train_skill) if args.cap_train_skill is not None else None,
        cap_train_n=int(args.cap_train_n) if args.cap_train_n is not None else None,
        train_with_ood_skills=[int(s) for s in args.train_with_ood_skills] if args.train_with_ood_skills is not None else None,
        ood_train_frac=float(args.ood_train_frac),
        test_frac=float(args.test_frac),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lr_decay=str(args.lr_decay),
        min_lr=float(args.min_lr),
        warmup_steps=int(args.warmup_steps),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(args.device),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        eval_every=int(args.eval_every),
        save_every=int(args.save_every),
        eval_tasks=int(args.eval_tasks),
        eval_batch_size=int(args.eval_batch_size),
        plot_unsolved_n=int(args.plot_unsolved_n),
        plot_solved_n=int(args.plot_solved_n),
        plot_augmented_n=int(args.plot_augmented_n),
        print_solved_n=int(args.print_solved_n),
        progress=bool(args.progress),
        out_dir=Path(args.out_dir),
        no_plots=bool(args.no_plots),
        dataset_device=str(args.dataset_device),
        aug=bool(args.aug),
        aug_geom_prob=float(args.aug_geom_prob),
        aug_color_prob=float(args.aug_color_prob),
        aug_translate_prob=float(args.aug_translate_prob),
        aug_translate_max=int(args.aug_translate_max),
        aug_keep_background=bool(args.aug_keep_background),
        eval_vote_augs=int(args.eval_vote_augs),
        run_tests=bool(args.run_tests),
    )


if __name__ == "__main__":
    cli_main()


