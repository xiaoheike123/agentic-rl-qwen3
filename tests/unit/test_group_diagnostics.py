from agent_rl.eval.group_diagnostics import build_group_report


def test_group_diagnostics_recommends_g4_at_half_mixed_groups() -> None:
    rows = []
    for group in range(8):
        for sample in range(4):
            success = group < 4 and sample == 0
            rows.append(
                {
                    "tau_group_id": f"group-{group}",
                    "tau_outcome_reward": float(success),
                    "response_ids": [group, sample],
                }
            )
    report = build_group_report(rows)
    assert report["nonzero_advantage_groups"] == 4
    assert report["nonzero_advantage_fraction"] == 0.5
    assert report["recommendation"] == "KEEP_G4"
