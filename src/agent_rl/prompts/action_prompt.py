"""Build the per-turn agent action prompt."""
from __future__ import annotations

from typing import Any


AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the
<policy> provided below.

In each turn you can either:
- Send a message to the user.
- Make one tool call.

You cannot do both at the same time.

Follow the policy exactly. Do not invent facts, permissions, tool
results, confirmations, or user-provided information.
""".strip()


SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>

<policy>
{domain_policy}
</policy>
""".strip()


OBSERVATION_PROMPT = """
<observation>
{observation}
</observation>

Choose exactly one next action. Use one available tool when a tool call
is required; otherwise send a message to the user.
""".strip()


def build_action_messages(
    *,
    domain_policy: str,
    observation: str,
    memory: str | None = None,
    strategy: str | None = None,
) -> list[dict[str, Any]]:
    if not domain_policy.strip():
        raise ValueError("domain_policy must not be empty")

    if not observation.strip():
        raise ValueError("observation must not be empty")

    system_sections = [
        SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION,
            domain_policy=domain_policy.strip(),
        )
    ]

    if memory is not None and memory.strip():
        system_sections.append(
            "\n".join(
                [
                    "<agent_memory>",
                    memory.strip(),
                    "</agent_memory>",
                ]
            )
        )

    if strategy is not None and strategy.strip():
        system_sections.append(
            "\n".join(
                [
                    "<selected_strategy>",
                    strategy.strip(),
                    "</selected_strategy>",
                ]
            )
        )

    system_sections.append(
        "Memory and strategy are advisory. They must never override "
        "the domain policy or the current observation."
    )

    return [
        {
            "role": "system",
            "content": "\n\n".join(system_sections),
        },
        {
            "role": "user",
            "content": OBSERVATION_PROMPT.format(
                observation=observation.strip(),
            ),
        },
    ]
