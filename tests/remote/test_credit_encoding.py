import pytest


torch = pytest.importorskip("torch")

from agent_rl.trainer.credit_encoding import (  # noqa: E402
    decode_centered_evidence,
    encode_evidence_in_token_rewards,
)


@pytest.mark.parametrize("score", [0.0, 0.85, -0.3])
def test_evidence_transport_preserves_reward_and_sign(score):
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    evidence = torch.tensor([-1.0, -1.0, 0.0, 0.5, 0.5])
    encoded = encode_evidence_in_token_rewards(
        score=score,
        raw_evidence=evidence,
        response_mask=mask,
    )

    assert torch.isclose(encoded.sum(), torch.tensor(score), atol=1e-6)

    decoded = decode_centered_evidence(
        encoded.unsqueeze(0),
        mask.unsqueeze(0),
    )[0]
    assert decoded[0] < 0
    assert decoded[1] < 0
    assert decoded[2] == 0
    assert decoded[3] > 0
    assert decoded[4] > 0
    assert torch.isclose((decoded * mask).sum(), torch.tensor(0.0), atol=1e-5)
