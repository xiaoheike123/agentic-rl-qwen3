"""Audit one exported LoRA adapter and its smoke-run synchronization log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit_lora_adapter(
    adapter_dir: str | Path,
    *,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    root = Path(adapter_dir)
    config = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    tensors = load_file(root / "adapter_model.safetensors", device="cpu")
    lora_tensors = {
        name: tensor for name, tensor in tensors.items() if "lora_" in name
    }
    if not lora_tensors or len(lora_tensors) != len(tensors):
        raise RuntimeError("exported adapter contains missing or non-LoRA tensors")
    changed_b = [
        name
        for name, tensor in lora_tensors.items()
        if "lora_B" in name and bool(tensor.count_nonzero().item())
    ]
    if not changed_b:
        raise RuntimeError("no LoRA-B tensor changed from zero initialization")

    rank = int(config.get("r", 0))
    alpha = int(config.get("lora_alpha", 0))
    if (rank, alpha) != (64, 64):
        raise RuntimeError(f"expected LoRA rank/alpha 64/64, got {rank}/{alpha}")

    markers: dict[str, bool] = {}
    if log_path is not None:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        markers = {
            "optimizer_trainable_only": "LORA_OPTIMIZER_AUDIT" in text,
            "vllm_adapter_refreshed": "vLLM load weights, loaded_params" in text,
            "cache_reset_enabled": "free_cache_engine=true" in text,
            "bypass_disabled": "bypass_mode=false" in text,
        }
        missing = [name for name, present in markers.items() if not present]
        if missing:
            raise RuntimeError("smoke log is missing closure markers: " + ", ".join(missing))

    return {
        "rank": rank,
        "alpha": alpha,
        "adapter_tensors": len(lora_tensors),
        "changed_lora_b_tensors": len(changed_b),
        "log_markers": markers,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dir")
    parser.add_argument("--log")
    args = parser.parse_args()
    print(
        json.dumps(
            audit_lora_adapter(args.adapter_dir, log_path=args.log),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
