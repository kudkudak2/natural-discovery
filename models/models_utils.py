import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# --- Token vocabulary ---
# ARC colors are 0..9. We reserve two special tokens:
# - SEP_TOKEN: separator between grids in the prompt
# - PAD_TOKEN: reserved for potential variable-length prompts (kept fixed by augmentations)
N_COLORS = 10
SEP_TOKEN = N_COLORS  # 10
PAD_TOKEN = N_COLORS + 1  # 11
VOCAB_SIZE = N_COLORS + 2  # 12


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

#########################################
#########################################
#########################################

class RecursiveEncoderRelPos2D(nn.Module):
    """
    Recursive Reasoning: Reuses the SAME layer weights 'num_layers' times.
    This creates a deep computation graph with fewer parameters.
    """

    def __init__(
            self,
            *,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout: float,
            num_recurrence: int,  # How many times to loop
            rel_pos: RelPosBias2D,
            rel_pos_demo: Optional[RelPosBias2DWithinDemo] = None,
    ) -> None:
        super().__init__()
        self.num_recurrence = int(num_recurrence)
        # We only create ONE layer instance
        self.shared_layer = EncoderLayerRelPos2D(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            rel_pos=rel_pos,
            rel_pos_demo=rel_pos_demo,
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
        # Loop over the SAME layer
        for _ in range(self.num_recurrence):
            x = self.shared_layer(
                x,
                row=row,
                col=col,
                demo_row=demo_row,
                demo_col=demo_col,
                demo_id=demo_id,
                is_sep=is_sep,
            )
        return x


class HierarchicalEncoderRelPos2D(nn.Module):
    """
    Hierarchical Reasoning: Processes at different 'resolutions' or groupings.
    (Simplified example: A stack of layers where some might have restricted attention).
    """

    def __init__(
            self,
            *,
            num_layers: int,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout: float,
            rel_pos: RelPosBias2D,
            rel_pos_demo: Optional[RelPosBias2DWithinDemo] = None,
    ) -> None:
        super().__init__()
        # Example: First half of layers are "local" (standard), second half are "global"
        # For this codebase, we will just chain them, but you can add pooling/unpooling logic here.
        self.bottom_layers = nn.ModuleList([
            EncoderLayerRelPos2D(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout,
                                 rel_pos=rel_pos, rel_pos_demo=rel_pos_demo)
            for _ in range(num_layers // 2)
        ])
        self.top_layers = nn.ModuleList([
            EncoderLayerRelPos2D(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout,
                                 rel_pos=rel_pos, rel_pos_demo=rel_pos_demo)
            for _ in range(num_layers - (num_layers // 2))
        ])

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # 1. Process local details
        for layer in self.bottom_layers:
            x = layer(x, **kwargs)

        # 2. (Optional) Insert Hierarchical Pooling here if desired
        # x_pooled = pool(x)

        # 3. Process global concepts
        for layer in self.top_layers:
            x = layer(x, **kwargs)

        return x



#############################
#############################
#############################

class TRMEncoderRelPos2D(nn.Module):
    """
    Tiny Recursive Model (Samsung SAIL):
    A single 'body' layer that recurses 'recurrence_steps' times.
    Crucially, it injects the original input 'x' at every step (Input Injection).
    """

    def __init__(
            self,
            *,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout: float,
            recurrence_steps: int,  # The 'N' in the paper
            input_grid_size: int,
            output_grid_size: int,
            demo_rel_pos_bias_2d: bool = True,
    ) -> None:
        super().__init__()
        self.recurrence_steps = int(recurrence_steps)

        # Instantiate the shared 2D biases internally, just like EncoderRelPos2D
        g_max = max(int(input_grid_size), int(output_grid_size))
        self.rel = RelPosBias2D(grid_size=g_max, num_heads=int(num_heads))
        self.demo_rel: Optional[RelPosBias2DWithinDemo] = (
            RelPosBias2DWithinDemo(
                input_grid_size=int(input_grid_size),
                output_grid_size=int(output_grid_size),
                num_heads=int(num_heads),
            )
            if bool(demo_rel_pos_bias_2d) else None
        )

        # The single "Tiny" network body
        self.body = EncoderLayerRelPos2D(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            rel_pos=self.rel,
            rel_pos_demo=self.demo_rel,
        )

        # Gate to mix previous state with original input (optional but recommended for TRM)
        # Allows the model to say "how much of the original problem do I need to look at again?"
        self.input_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

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
        # x is the original input (Problem Statement + Context)
        z = x.clone()  # z is the "Latent State"

        for _ in range(self.recurrence_steps):
            # 1. Input Injection: Combine current thought (z) with original facts (x)
            combined = torch.cat([z, x], dim=-1)
            gate = torch.sigmoid(self.input_gate(combined))
            z_in = (1 - gate) * x + gate * z

            # 2. Recursive Step
            z = self.body(
                z_in,
                row=row,
                col=col,
                demo_row=demo_row,
                demo_col=demo_col,
                demo_id=demo_id,
                is_sep=is_sep,
            )

        return self.norm(z)


class HRMEncoderRelPos2D(nn.Module):
    """
    Hierarchical Reasoning Model (Sapient):
    Splits reasoning into 'High-level' (Abstract/Slow) and 'Low-level' (Concrete/Fast) modules.
    Runs for 'num_H_cycles', where each cycle contains 'num_L_steps'.
    """

    def __init__(
            self,
            *,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout: float,
            num_H_cycles: int = 3,  # Cycles of high-level planning
            num_L_steps: int = 4,  # Steps of low-level execution per cycle
            input_grid_size: int,
            output_grid_size: int,
            demo_rel_pos_bias_2d: bool = True,
    ) -> None:
        super().__init__()
        self.num_H_cycles = num_H_cycles
        self.num_L_steps = num_L_steps

        g_max = max(int(input_grid_size), int(output_grid_size))
        self.rel = RelPosBias2D(grid_size=g_max, num_heads=int(num_heads))
        self.demo_rel: Optional[RelPosBias2DWithinDemo] = (
            RelPosBias2DWithinDemo(
                input_grid_size=int(input_grid_size),
                output_grid_size=int(output_grid_size),
                num_heads=int(num_heads),
            )
            if bool(demo_rel_pos_bias_2d) else None
        )

        # High-Level Module (The "Planner")
        self.H_module = EncoderLayerRelPos2D(
            embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout, rel_pos=self.rel, rel_pos_demo=self.demo_rel
        )

        # Low-Level Module (The "Executor")
        self.L_module = EncoderLayerRelPos2D(
            embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout, rel_pos=self.rel, rel_pos_demo=self.demo_rel
        )

        # Projectors to communicate between H and L
        self.L_to_H = nn.Linear(embed_dim, embed_dim)
        self.H_to_L = nn.Linear(embed_dim, embed_dim)

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
        # z_L initialized from input
        z_L = x
        # z_H initialized as zero (or learned parameter)
        z_H = torch.zeros_like(x)

        kwargs = {
            'row': row, 'col': col,
            'demo_row': demo_row, 'demo_col': demo_col,
            'demo_id': demo_id, 'is_sep': is_sep
        }

        for h in range(self.num_H_cycles):
            # --- Fast Loop (Low Level) ---
            # L "thinks" for several steps, guided by current H
            guidance = self.H_to_L(z_H)
            for l in range(self.num_L_steps):
                # L input is combination of its previous thought + H guidance + Original Fact (x)
                # (Simple addition fusion used here, can be gated)
                current_input = z_L + guidance + x
                z_L = self.L_module(current_input, **kwargs)

            # --- Slow Update (High Level) ---
            # H updates its plan based on what L accomplished
            summary = self.L_to_H(z_L)
            z_H = self.H_module(z_H + summary, **kwargs)

        # We typically return the Low-Level detailed representation for the final prediction
        return z_L


class TRMEncoderStandard(nn.Module):
    """
    Tiny Recursive Model (TRM) using standard absolute positional embeddings.
    Loops over a single native PyTorch Transformer layer.
    """

    def __init__(
            self,
            *,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout: float,
            recurrence_steps: int,
    ) -> None:
        super().__init__()
        self.recurrence_steps = int(recurrence_steps)

        # Standard PyTorch layer (batch_first=True is required for this codebase's dimensions)
        self.body = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
        )

        self.input_gate = nn.Linear(int(embed_dim) * 2, int(embed_dim))
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = x.clone()

        for _ in range(self.recurrence_steps):
            # Input Injection
            combined = torch.cat([z, x], dim=-1)
            gate = torch.sigmoid(self.input_gate(combined))
            z_in = (1 - gate) * x + gate * z

            # Forward pass through the single layer
            z = self.body(z_in, src_key_padding_mask=src_key_padding_mask)

        return self.norm(z)


class HRMEncoderStandard(nn.Module):
    """
    Hierarchical Reasoning Model (HRM) using standard absolute positional embeddings.
    """

    def __init__(
            self,
            *,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout: float,
            num_H_cycles: int = 3,
            num_L_steps: int = 4,
    ) -> None:
        super().__init__()
        self.num_H_cycles = int(num_H_cycles)
        self.num_L_steps = int(num_L_steps)

        self.H_module = nn.TransformerEncoderLayer(
            d_model=int(embed_dim), nhead=int(num_heads), dim_feedforward=int(ff_dim),
            dropout=float(dropout), batch_first=True
        )

        self.L_module = nn.TransformerEncoderLayer(
            d_model=int(embed_dim), nhead=int(num_heads), dim_feedforward=int(ff_dim),
            dropout=float(dropout), batch_first=True
        )

        self.L_to_H = nn.Linear(int(embed_dim), int(embed_dim))
        self.H_to_L = nn.Linear(int(embed_dim), int(embed_dim))

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_L = x
        z_H = torch.zeros_like(x)

        for h in range(self.num_H_cycles):
            # Fast Loop
            guidance = self.H_to_L(z_H)
            for l in range(self.num_L_steps):
                current_input = z_L + guidance + x
                z_L = self.L_module(current_input, src_key_padding_mask=src_key_padding_mask)

            # Slow Update
            summary = self.L_to_H(z_L)
            z_H = self.H_module(z_H + summary, src_key_padding_mask=src_key_padding_mask)

        return z_L