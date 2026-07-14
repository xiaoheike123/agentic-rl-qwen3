from types import SimpleNamespace

from agent_rl.rollout.vllm_policy import VLLMPolicy, VLLMPolicyConfig


class _FakeCompletions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **request):
        self.request = request
        message = SimpleNamespace(content="OK", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            usage=None,
            model_dump=lambda **_: {"choices": []},
        )


def test_vllm_policy_forwards_locked_sampling_parameters() -> None:
    completions = _FakeCompletions()
    policy = object.__new__(VLLMPolicy)
    policy.config = VLLMPolicyConfig(temperature=0.8, top_p=0.95)
    policy._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    output = policy.generate([{"role": "user", "content": "hello"}])

    assert output.action == "OK"
    assert completions.request["temperature"] == 0.8
    assert completions.request["top_p"] == 0.95
