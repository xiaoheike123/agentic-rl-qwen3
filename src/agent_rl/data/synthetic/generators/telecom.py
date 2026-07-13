"""Clean-room telecom tasks derived only from policy, tools, and database state."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from tau2.data_model.tasks import EnvFunctionCall
from tau2.domains.telecom.data_model import BillStatus, LineStatus, TelecomDB
from tau2.domains.telecom.utils import TELECOM_DB_PATH

from agent_rl.data.synthetic.generators.common import (
    GeneratedCandidate,
    OracleActionSpec,
    make_candidate,
)
from agent_rl.data.synthetic.telecom_policy import active_line_subsets, resume_allowed


@dataclass(frozen=True, slots=True)
class SupportCase:
    name: str
    symptom: str
    initialization: tuple[tuple[str, dict[str, Any]], ...]
    fixes: tuple[tuple[str, dict[str, Any]], ...]
    diagnostic_tools: tuple[str, ...]
    difficulty: str


SINGLE_SUPPORT_CASES = (
    SupportCase("airplane_mode_on", "the phone shows no service", (("turn_airplane_mode_on", {}),), (("toggle_airplane_mode", {}),), ("check_status_bar", "check_network_status"), "simple"),
    SupportCase("sim_card_missing", "the phone says no SIM is detected", (("unseat_sim_card", {}),), (("reseat_sim_card", {}),), ("check_network_status", "check_sim_status"), "medium"),
    SupportCase("mobile_data_off", "cellular data does not work away from Wi-Fi", (("turn_data_off", {}),), (("toggle_data", {}),), ("check_status_bar", "check_network_status"), "simple"),
    SupportCase("broken_apn", "the phone has signal but no mobile internet", (("break_apn_settings", {}),), (("reset_apn_settings", {}),), ("check_network_status", "check_apn_settings"), "complex"),
    SupportCase("data_saver_on", "mobile internet is unexpectedly very slow", (("turn_data_saver_mode_on", {}),), (("toggle_data_saver_mode", {}),), ("check_data_restriction_status", "run_speed_test"), "medium"),
    SupportCase("slow_vpn", "mobile internet became slow after connecting a VPN", (("break_vpn", {}),), (("disconnect_vpn", {}),), ("check_vpn_status", "run_speed_test"), "complex"),
    SupportCase("missing_sms_permission", "picture messages cannot be sent", (("remove_app_permission", {"app_name": "messaging", "permission": "sms"}),), (("grant_app_permission", {"app_name": "messaging", "permission": "sms"}),), ("can_send_mms", "check_app_permissions"), "complex"),
    SupportCase("missing_storage_permission", "picture messages fail when attaching media", (("remove_app_permission", {"app_name": "messaging", "permission": "storage"}),), (("grant_app_permission", {"app_name": "messaging", "permission": "storage"}),), ("can_send_mms", "check_app_permissions"), "complex"),
    SupportCase("broken_mms_apn", "ordinary data works but picture messages fail", (("break_apn_mms_setting", {}),), (("reset_apn_settings", {}),), ("can_send_mms", "check_apn_settings"), "hard"),
    SupportCase("slow_network_mode", "the phone remains on a legacy slow network", (("set_network_mode_preference", {"mode": "2g_only"}),), (("set_network_mode_preference", {"mode": "4g_5g_preferred"}),), ("check_network_mode_preference", "run_speed_test"), "medium"),
)

SINGLE_BY_NAME = {case.name: case for case in SINGLE_SUPPORT_CASES}
FAMILY_LIMIT = 96
COMBINED_CASE_NAMES = (
    ("airplane_mode_on", "mobile_data_off"),
    ("airplane_mode_on", "broken_apn"),
    ("sim_card_missing", "mobile_data_off"),
    ("mobile_data_off", "data_saver_on"),
    ("mobile_data_off", "slow_vpn"),
    ("broken_apn", "data_saver_on"),
    ("broken_apn", "slow_vpn"),
    ("data_saver_on", "slow_network_mode"),
    ("slow_vpn", "slow_network_mode"),
    ("missing_sms_permission", "missing_storage_permission"),
)


def _line_map(db: TelecomDB) -> dict[str, Any]:
    return {line.line_id: line for line in db.lines}


def _policy_metadata(
    *,
    case: str,
    difficulty: str,
    reads: tuple[str, ...],
    actions: tuple[OracleActionSpec, ...],
    customer_id: str,
    line_ids: tuple[str, ...],
    support_cases: tuple[str, ...] = (),
    **extra: Any,
) -> dict[str, Any]:
    return {
        "policy": {
            "case": case,
            "difficulty": difficulty,
            "required_reads": list(dict.fromkeys(reads)),
            "required_confirmation": True,
            "expected_writes": [action.name for action in actions],
            "action_requestors": [action.requestor for action in actions],
            "customer_id": customer_id,
            "affected_line_ids": list(line_ids),
            "support_cases": list(support_cases),
            **extra,
        }
    }


def _identity_initialization(customer: Any, phone_number: str) -> EnvFunctionCall:
    return EnvFunctionCall(
        env_type="user",
        func_name="set_user_info",
        arguments={"name": customer.full_name, "phone_number": phone_number},
    )


def generate_telecom_candidates(
    seed: int,
    database: TelecomDB | None = None,
) -> list[GeneratedCandidate]:
    rng = random.Random(seed)
    db = database if database is not None else TelecomDB.load(TELECOM_DB_PATH)
    customers = sorted(db.customers, key=lambda item: item.customer_id)
    rng.shuffle(customers)
    lines = _line_map(db)
    families: dict[str, list[GeneratedCandidate]] = {}

    def add(
        *,
        template: str,
        customer: Any,
        line_ids: tuple[str, ...],
        reason_for_call: str,
        known_info: str,
        task_instructions: str,
        actions: tuple[OracleActionSpec, ...],
        communicate_info: list[str],
        purpose: str,
        difficulty: str,
        reads: tuple[str, ...],
        initialization_actions: tuple[EnvFunctionCall, ...],
        user_tools: list[str] | None = None,
        support_cases: tuple[str, ...] = (),
        policy_extra: dict[str, Any] | None = None,
    ) -> None:
        pool = families.setdefault(template, [])
        if len(pool) >= FAMILY_LIMIT:
            return
        pool.append(
            make_candidate(
                domain="telecom",
                template=template,
                seed=seed + len(pool) + len(families) * 10_003,
                entities=(customer.customer_id, *line_ids),
                reason_for_call=reason_for_call,
                known_info=known_info,
                task_instructions=task_instructions,
                actions=actions,
                communicate_info=communicate_info,
                purpose=purpose,
                metadata=_policy_metadata(
                    case=template,
                    difficulty=difficulty,
                    reads=reads,
                    actions=actions,
                    customer_id=customer.customer_id,
                    line_ids=line_ids,
                    support_cases=support_cases,
                    **(policy_extra or {}),
                ),
                initialization_actions=initialization_actions,
                user_tools=user_tools,
            )
        )

    for customer in customers:
        customer_lines = [lines[line_id] for line_id in customer.line_ids]
        active_lines = [line for line in customer_lines if line.status == LineStatus.ACTIVE]

        for line in active_lines:
            identity = _identity_initialization(customer, line.phone_number)
            known = (
                f"My name is {customer.full_name}, date of birth is "
                f"{customer.date_of_birth}, and the affected number is {line.phone_number}."
            )
            support_specs: list[tuple[str, tuple[SupportCase, ...]]] = [
                (case.name, (case,)) for case in SINGLE_SUPPORT_CASES
            ]
            support_specs.extend(
                ("and".join(names), tuple(SINGLE_BY_NAME[name] for name in names))
                for names in COMBINED_CASE_NAMES
            )

            for support_name, cases in support_specs:
                init_calls = [identity]
                fixes: list[OracleActionSpec] = []
                diagnostics: list[str] = []
                symptoms: list[str] = []
                for case in cases:
                    symptoms.append(case.symptom)
                    diagnostics.extend(case.diagnostic_tools)
                    init_calls.extend(
                        EnvFunctionCall(env_type="user", func_name=name, arguments=args)
                        for name, args in case.initialization
                    )
                    fixes.extend(
                        OracleActionSpec(name=name, arguments=args, requestor="user")
                        for name, args in case.fixes
                    )
                write_tools = [action.name for action in fixes]
                add(
                    template=f"support_{support_name}",
                    customer=customer,
                    line_ids=(line.line_id,),
                    reason_for_call="Troubleshoot " + " and ".join(symptoms) + ".",
                    known_info=known,
                    task_instructions=(
                        "Follow diagnostic instructions one step at a time, report each "
                        "result accurately, perform only requested device changes, and "
                        "confirm whether the problem is resolved."
                    ),
                    actions=tuple(fixes),
                    communicate_info=[line.phone_number, "issue resolved"],
                    purpose="Diagnose and repair a clean-room mobile-device fault state.",
                    difficulty=("hard" if len(cases) > 1 else cases[0].difficulty),
                    reads=("get_customer_by_phone", *diagnostics),
                    initialization_actions=tuple(init_calls),
                    user_tools=list(dict.fromkeys([*diagnostics, *write_tools])),
                    support_cases=tuple(case.name for case in cases),
                )

        for subset in active_line_subsets([line.line_id for line in active_lines]):
            actions = tuple(
                OracleActionSpec(
                    "suspend_line",
                    {
                        "customer_id": customer.customer_id,
                        "line_id": line_id,
                        "reason": "customer requested temporary suspension",
                    },
                )
                for line_id in subset
            )
            phone = lines[subset[0]].phone_number
            add(
                template="suspend_lines",
                customer=customer,
                line_ids=tuple(subset),
                reason_for_call=f"Temporarily suspend lines {', '.join(subset)}.",
                known_info=f"My name is {customer.full_name}, DOB {customer.date_of_birth}; the line IDs are {', '.join(subset)}.",
                task_instructions="Confirm each selected line only after the loss of service and monthly holding fee are explained.",
                actions=actions,
                communicate_info=[*subset, "$5/month holding fee"],
                purpose="Suspend an explicitly selected set of active lines.",
                difficulty=("medium" if len(subset) == 1 else "hard"),
                reads=("get_customer_by_phone", "get_details_by_id"),
                initialization_actions=(_identity_initialization(customer, phone),),
            )

        for line in customer_lines:
            phone = line.phone_number
            initialization = (_identity_initialization(customer, phone),)
            if resume_allowed(db, customer.customer_id, line.line_id):
                add(
                    template="resume_eligible_line",
                    customer=customer,
                    line_ids=(line.line_id,),
                    reason_for_call=f"Restore service on suspended line {line.line_id}.",
                    known_info=f"My name is {customer.full_name}, DOB {customer.date_of_birth}, and the number is {phone}.",
                    task_instructions="Ask the agent to verify billing and contract eligibility, confirm restoration, then acknowledge the reboot requirement.",
                    actions=(OracleActionSpec("resume_line", {"customer_id": customer.customer_id, "line_id": line.line_id}),),
                    communicate_info=[line.line_id, "reboot"],
                    purpose="Resume a suspended line only after all policy conditions pass.",
                    difficulty="complex",
                    reads=("get_customer_by_phone", "get_details_by_id", "get_bills_for_customer"),
                    initialization_actions=initialization,
                )

            if line.status == LineStatus.ACTIVE and not line.roaming_enabled:
                add(
                    template="enable_roaming_for_travel",
                    customer=customer,
                    line_ids=(line.line_id,),
                    reason_for_call=f"Enable data roaming on {line.line_id} for an upcoming international trip.",
                    known_info=f"My name is {customer.full_name}, DOB {customer.date_of_birth}, and the number is {phone}. I am traveling abroad.",
                    task_instructions="Confirm after the agent checks the current setting and explains that roaming is being enabled.",
                    actions=(OracleActionSpec("enable_roaming", {"customer_id": customer.customer_id, "line_id": line.line_id}),),
                    communicate_info=[line.line_id, "roaming enabled", "no cost"],
                    purpose="Enable roaming for a customer who explicitly reports international travel.",
                    difficulty="medium",
                    reads=("get_customer_by_phone", "get_details_by_id"),
                    initialization_actions=initialization,
                    policy_extra={"traveling_abroad": True},
                )

            has_stable_draft = any(
                bill.customer_id == customer.customer_id and bill.status == BillStatus.DRAFT
                for bill in db.bills
            )
            if line.status == LineStatus.ACTIVE and has_stable_draft:
                amount = (0.5, 1.0, 1.5, 2.0)[int(line.line_id[-1]) % 4]
                plan = next(plan for plan in db.plans if plan.plan_id == line.plan_id)
                charge = round(amount * plan.data_refueling_price_per_gb, 2)
                add(
                    template="refuel_data_with_quote",
                    customer=customer,
                    line_ids=(line.line_id,),
                    reason_for_call=f"Add {amount:g} GB of data to {line.line_id}.",
                    known_info=f"My name is {customer.full_name}, DOB {customer.date_of_birth}, and the number is {phone}.",
                    task_instructions=f"Approve only after the agent quotes the exact ${charge:.2f} charge.",
                    actions=(OracleActionSpec("refuel_data", {"customer_id": customer.customer_id, "line_id": line.line_id, "gb_amount": amount}),),
                    communicate_info=[line.line_id, f"{amount:g} GB", f"${charge:.2f}"],
                    purpose="Purchase no more than two gigabytes at the plan's verified rate.",
                    difficulty="medium",
                    reads=("get_customer_by_phone", "get_details_by_id", "get_data_usage"),
                    initialization_actions=initialization,
                    policy_extra={"quoted_charges": {line.line_id: charge}},
                )

        stable_active = [
            line
            for line in active_lines
            if any(
                bill.customer_id == customer.customer_id and bill.status == BillStatus.DRAFT
                for bill in db.bills
            )
        ]
        for first, second in combinations(stable_active, 2):
            amount = 1.0
            charges = {
                line.line_id: round(
                    amount
                    * next(plan for plan in db.plans if plan.plan_id == line.plan_id).data_refueling_price_per_gb,
                    2,
                )
                for line in (first, second)
            }
            actions = tuple(
                OracleActionSpec("refuel_data", {"customer_id": customer.customer_id, "line_id": line.line_id, "gb_amount": amount})
                for line in (first, second)
            )
            add(
                template="refuel_two_lines_with_separate_quotes",
                customer=customer,
                line_ids=(first.line_id, second.line_id),
                reason_for_call=f"Add 1 GB to both {first.line_id} and {second.line_id}.",
                known_info=f"My name is {customer.full_name}, DOB {customer.date_of_birth}; the numbers are {first.phone_number} and {second.phone_number}.",
                task_instructions="Confirm each line only after its own plan rate and charge are stated.",
                actions=actions,
                communicate_info=[first.line_id, f"${charges[first.line_id]:.2f}", second.line_id, f"${charges[second.line_id]:.2f}"],
                purpose="Refuel two lines while keeping their plan-specific charges separate.",
                difficulty="hard",
                reads=("get_customer_by_phone", "get_details_by_id", "get_data_usage"),
                initialization_actions=(_identity_initialization(customer, first.phone_number),),
                policy_extra={"quoted_charges": charges},
            )

    return [candidate for template in sorted(families) for candidate in families[template]]
