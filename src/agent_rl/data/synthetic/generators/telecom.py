"""Procedural telecom tasks derived from database entities, not benchmark tasks."""

from __future__ import annotations

import random

from tau2.domains.telecom.data_model import LineStatus, TelecomDB
from tau2.domains.telecom.utils import TELECOM_DB_PATH

from agent_rl.data.synthetic.generators.common import (
    GeneratedCandidate,
    OracleActionSpec,
    make_candidate,
)


def generate_telecom_candidates(seed: int) -> list[GeneratedCandidate]:
    rng = random.Random(seed)
    database = TelecomDB.load(TELECOM_DB_PATH)
    customers_by_line = {
        line_id: customer
        for customer in database.customers
        for line_id in customer.line_ids
    }
    lines = sorted(database.lines, key=lambda item: item.line_id)
    rng.shuffle(lines)

    candidates: list[GeneratedCandidate] = []
    for index, line in enumerate(lines):
        customer = customers_by_line.get(line.line_id)
        if customer is None:
            continue
        identity = (
            f"My customer ID is {customer.customer_id}, my full name is "
            f"{customer.full_name}, my date of birth is {customer.date_of_birth}, "
            f"and the target line is {line.line_id} ({line.phone_number})."
        )
        if line.roaming_enabled:
            roaming_action = "disable_roaming"
            roaming_word = "disabled"
            roaming_reason = "Turn off international roaming"
        else:
            roaming_action = "enable_roaming"
            roaming_word = "enabled"
            roaming_reason = "Turn on international roaming"
        candidates.append(
            make_candidate(
                domain="telecom",
                template=roaming_action,
                seed=seed + index * 2,
                entities=(customer.customer_id, line.line_id),
                reason_for_call=f"{roaming_reason} for line {line.line_id}.",
                known_info=identity,
                task_instructions=(
                    "Request only the roaming setting change. Complete any identity "
                    "verification and confirm the requested change if asked."
                ),
                actions=(
                    OracleActionSpec(
                        name=roaming_action,
                        arguments={
                            "customer_id": customer.customer_id,
                            "line_id": line.line_id,
                        },
                    ),
                ),
                communicate_info=[line.line_id, roaming_word],
                purpose="Change the international roaming setting on one line.",
            )
        )

        # The upstream telecom DB is intentionally tiny (nine lines). Varying
        # the purchased quantity creates distinct, environment-verifiable
        # targets without copying the much larger published task pool.
        for amount_index in range(1, 21):
            amount = amount_index / 2
            candidates.append(
                make_candidate(
                    domain="telecom",
                    template="refuel_data",
                    seed=seed + 10_000 + index * 100 + amount_index,
                    entities=(customer.customer_id, line.line_id),
                    reason_for_call=(
                        f"Add {amount:g} GB of data to line {line.line_id}."
                    ),
                    known_info=identity,
                    task_instructions=(
                        f"Request exactly {amount:g} GB of additional data. Complete "
                        "identity checks and explicitly approve the quoted charge."
                    ),
                    actions=(
                        OracleActionSpec(
                            name="refuel_data",
                            arguments={
                                "customer_id": customer.customer_id,
                                "line_id": line.line_id,
                                "gb_amount": amount,
                            },
                        ),
                    ),
                    communicate_info=[line.line_id, f"{amount:g} GB"],
                    purpose="Purchase a precise amount of additional mobile data.",
                )
            )

        if line.status == LineStatus.ACTIVE:
            status_action = "suspend_line"
            status_arguments = {
                "customer_id": customer.customer_id,
                "line_id": line.line_id,
                "reason": "temporary travel",
            }
            status_word = "suspended"
            status_reason = "Temporarily suspend"
        elif line.status in {LineStatus.SUSPENDED, LineStatus.PENDING_ACTIVATION}:
            status_action = "resume_line"
            status_arguments = {
                "customer_id": customer.customer_id,
                "line_id": line.line_id,
            }
            status_word = "resumed"
            status_reason = "Resume service on"
        else:
            continue
        candidates.append(
            make_candidate(
                domain="telecom",
                template=status_action,
                seed=seed + index * 2 + 1,
                entities=(customer.customer_id, line.line_id),
                reason_for_call=f"{status_reason} line {line.line_id}.",
                known_info=identity,
                task_instructions=(
                    "Request only this line-status change. Complete identity checks and "
                    "explicitly confirm after the agent explains the consequences."
                ),
                actions=(
                    OracleActionSpec(
                        name=status_action,
                        arguments=status_arguments,
                    ),
                ),
                communicate_info=[line.line_id, status_word],
                purpose="Change the service status of one telecom line.",
            )
        )
    return candidates
