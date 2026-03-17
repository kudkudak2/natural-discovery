import pytest
param = pytest.mark.parametrize

import torch

@param('causal', (False, True))
def test_hrm(causal):
    from HRM.hrm import HRM
    from x_transformers import Encoder

    hrm = HRM(
        networks = [
            dict(
                dim = 32,
                depth = 2,
                attn_dim_head = 8,
                heads = 1,
                use_rmsnorm = True,
                rotary_pos_emb = True,
                pre_norm = False
            ),
            dict(
                dim = 32,
                depth = 4,
                attn_dim_head = 8,
                heads = 1,
                use_rmsnorm = True,
                rotary_pos_emb = True,
                pre_norm = False
            ),
            Encoder(
                dim = 32,
                depth = 8,
                attn_dim_head = 8,
                heads = 1,
                use_rmsnorm = True,
                rotary_pos_emb = True,
                pre_norm = False
            )
        ],
        causal = causal,
        num_tokens = 256,
        dim = 32,
        reasoning_steps = 10
    )

    seq = torch.randint(0, 256, (3, 1024))
    labels = torch.randint(0, 256, (3, 1024))

    loss, hiddens, _ = hrm(seq, labels = labels)
    loss.backward()

    loss, hiddens, _ = hrm(seq, hiddens = hiddens, labels = labels)
    loss.backward()

    # after much training

    pred = hrm(seq, reasoning_steps = 5)

@param('compute_loss_across_reasoning_steps', (False, True))
@param('causal', (False, True))
def test_hrm_with_act(
    compute_loss_across_reasoning_steps,
    causal
):
    from HRM.hrm_with_act import HRM

    hrm = HRM(
        networks = [
            dict(
                dim = 32,
                depth = 2,
                attn_dim_head = 8,
                heads = 1,
                use_rmsnorm = True,
                rotary_pos_emb = True,
                pre_norm = False
            )
        ],
        num_tokens = 256,
        dim = 32,
        max_reasoning_steps = 10,
        causal = causal
    )

    seq = torch.randint(0, 256, (3, 1024))
    labels = torch.randint(0, 256, (3, 1024))

    loss, *_ = hrm(seq, labels = labels)
    loss.backward()

    # after much training

    seq_inference = torch.randint(0, 256, (32, 1024))

    pred, exit_hiddens, exited_indices_order = hrm(seq_inference, max_reasoning_steps = 5, compute_loss_across_reasoning_steps = compute_loss_across_reasoning_steps)

    assert len(exit_hiddens) == 1
    assert exited_indices_order.shape == (32,)
