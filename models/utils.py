import torch


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
