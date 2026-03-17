import torch
import torch.nn as nn
from typing import Optional
from .models_utils import EncoderRelPos2D, TRMEncoderRelPos2D, HRMEncoderRelPos2D, TRMEncoderStandard, HRMEncoderStandard
from arc_train_utils import VOCAB_SIZE
from .utils import _prompt_rows_cols, _prompt_demo_rows_cols, _prompt_token_types


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
        model_type: str = "standard",
        recurrence_steps: int = 12,  # <--- Add this (for TRM)
        hrm_h_cycles: int = 3,  # <--- Add this (for HRM)
        hrm_l_steps: int = 4,  # <--- Add this (for HRM)
    ) -> None:
        super().__init__()
        self.model_type = model_type
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
            if self.model_type == "standard":
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
            elif self.model_type == "trm":
                self.transformer_rel = TRMEncoderRelPos2D(
                    embed_dim=int(embed_dim),
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    dropout=float(dropout),
                    recurrence_steps=int(recurrence_steps),
                    input_grid_size=self.grid_size,
                    output_grid_size=self.output_grid_size,
                    demo_rel_pos_bias_2d=bool(demo_rel_pos_bias_2d),
                )
            elif self.model_type == "hrm":
                self.transformer_rel = HRMEncoderRelPos2D(
                    embed_dim=int(embed_dim),
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    dropout=float(dropout),
                    num_H_cycles=int(hrm_h_cycles),
                    num_L_steps=int(hrm_l_steps),
                    input_grid_size=self.grid_size,
                    output_grid_size=self.output_grid_size,
                    demo_rel_pos_bias_2d=bool(demo_rel_pos_bias_2d),
                )
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type!r}")
        else:
            if self.model_type == "standard":
                layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    batch_first=True,
                    dropout=dropout,
                )
                self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            elif self.model_type == "trm":
                self.transformer = TRMEncoderStandard(
                    embed_dim=int(embed_dim),
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    dropout=float(dropout),
                    recurrence_steps=int(recurrence_steps),
                )
            elif self.model_type == "hrm":
                self.transformer = HRMEncoderStandard(
                    embed_dim=int(embed_dim),
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    dropout=float(dropout),
                    num_H_cycles=int(hrm_h_cycles),
                    num_L_steps=int(hrm_l_steps),
                )
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type!r}")
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
