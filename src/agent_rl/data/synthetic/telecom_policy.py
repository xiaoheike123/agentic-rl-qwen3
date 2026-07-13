"""Policy validation for clean-room telecom task generation."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import TYPE_CHECKING

from tau2.domains.telecom.data_model import BillStatus, LineStatus, TelecomDB
from tau2.domains.telecom.utils import TELECOM_DB_PATH, get_today

if TYPE_CHECKING:
    from agent_rl.data.synthetic.generators.common import GeneratedCandidate


CUSTOMER_READ = "get_customer_by_phone"
DETAIL_READ = "get_details_by_id"
USAGE_READ = "get_data_usage"

SUPPORT_CASES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "airplane_mode_on": (("turn_airplane_mode_on",), ("toggle_airplane_mode",)),
    "sim_card_missing": (("unseat_sim_card",), ("reseat_sim_card",)),
    "mobile_data_off": (("turn_data_off",), ("toggle_data",)),
    "broken_apn": (("break_apn_settings",), ("reset_apn_settings",)),
    "data_saver_on": (("turn_data_saver_mode_on",), ("toggle_data_saver_mode",)),
    "slow_vpn": (("break_vpn",), ("disconnect_vpn",)),
    "missing_sms_permission": (("remove_app_permission",), ("grant_app_permission",)),
    "missing_storage_permission": (("remove_app_permission",), ("grant_app_permission",)),
    "broken_mms_apn": (("break_apn_mms_setting",), ("reset_apn_settings",)),
    "slow_network_mode": (("set_network_mode_preference",), ("set_network_mode_preference",)),
}


@lru_cache(maxsize=1)
def load_telecom_policy_db() -> TelecomDB:
    return TelecomDB.load(TELECOM_DB_PATH)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _customer(db: TelecomDB, customer_id: str):
    return next(item for item in db.customers if item.customer_id == customer_id)


def _line(db: TelecomDB, line_id: str):
    return next(item for item in db.lines if item.line_id == line_id)


def _plan(db: TelecomDB, plan_id: str):
    return next(item for item in db.plans if item.plan_id == plan_id)


def _overdue_bills(db: TelecomDB, customer_id: str):
    return [
        bill
        for bill in db.bills
        if bill.customer_id == customer_id and bill.status == BillStatus.OVERDUE
    ]


def resume_allowed(db: TelecomDB, customer_id: str, line_id: str) -> bool:
    line = _line(db, line_id)
    contract_valid = line.contract_end_date is None or line.contract_end_date >= get_today()
    return bool(
        line.status in {LineStatus.SUSPENDED, LineStatus.PENDING_ACTIVATION}
        and not _overdue_bills(db, customer_id)
        and contract_valid
    )


def active_line_subsets(line_ids: list[str], maximum: int = 5):
    for size in range(1, min(maximum, len(line_ids)) + 1):
        yield from combinations(sorted(line_ids), size)


def validate_telecom_candidate(
    candidate: GeneratedCandidate,
    database: TelecomDB | None = None,
) -> None:
    policy = candidate.metadata.get("policy")
    _require(isinstance(policy, dict), "telecom candidate requires policy metadata")
    _require(policy.get("required_confirmation") is True, "writes require confirmation")
    _require(
        policy.get("difficulty") in {"simple", "medium", "complex", "hard"},
        "invalid task difficulty",
    )
    required_reads = policy.get("required_reads")
    _require(isinstance(required_reads, list), "required_reads must be a list")
    _require(CUSTOMER_READ in required_reads, "telecom tasks must identify the customer")

    criteria = candidate.task.evaluation_criteria
    actions = list(criteria.actions or []) if criteria else []
    _require(actions, "telecom candidate requires at least one state-changing action")
    _require(
        [action.name for action in actions] == policy.get("expected_writes"),
        "expected_writes mismatch",
    )
    _require(
        [action.requestor for action in actions] == policy.get("action_requestors"),
        "action_requestors mismatch",
    )

    db = database if database is not None else load_telecom_policy_db()
    customer_id = policy.get("customer_id")
    customer = _customer(db, customer_id)
    affected_line_ids = policy.get("affected_line_ids")
    _require(isinstance(affected_line_ids, list), "affected_line_ids must be a list")
    _require(
        all(line_id in customer.line_ids for line_id in affected_line_ids),
        "all affected lines must belong to one customer",
    )

    support_cases = policy.get("support_cases", [])
    initial_state = candidate.task.initial_state
    initial_calls = list(initial_state.initialization_actions or []) if initial_state else []
    initial_functions = [call.func_name for call in initial_calls]
    _require(initial_functions and initial_functions[0] == "set_user_info", "user identity initialization missing")

    if support_cases:
        _require(all(action.requestor == "user" for action in actions), "support fixes must be user actions")
        expected_initial: list[str] = ["set_user_info"]
        expected_actions: list[str] = []
        for case in support_cases:
            _require(case in SUPPORT_CASES, f"unknown support case: {case}")
            initial, fixes = SUPPORT_CASES[case]
            expected_initial.extend(initial)
            expected_actions.extend(fixes)
        _require(initial_functions == expected_initial, "support initialization mismatch")
        _require([action.name for action in actions] == expected_actions, "support fix sequence mismatch")
        return

    _require(all(action.requestor == "assistant" for action in actions), "account writes must be assistant actions")
    seen_line_effects: set[tuple[str, str]] = set()
    for action in actions:
        arguments = action.arguments
        _require(arguments.get("customer_id") == customer_id, "customer mismatch")
        line_id = arguments.get("line_id")
        _require(line_id in customer.line_ids, "line does not belong to customer")
        line = _line(db, line_id)

        if action.name == "enable_roaming":
            _require(not line.roaming_enabled, "roaming is already enabled")
            _require(policy.get("traveling_abroad") is True, "roaming enablement requires travel")
            effect = (line_id, "roaming")
        elif action.name == "refuel_data":
            amount = float(arguments["gb_amount"])
            _require(line.status == LineStatus.ACTIVE, "data refueling requires an active line")
            _require(0.0 < amount <= 2.0, "data refueling must be in (0, 2] GB")
            _require(USAGE_READ in required_reads, "refueling requires usage and plan lookup")
            plan = _plan(db, line.plan_id)
            expected_charge = round(amount * plan.data_refueling_price_per_gb, 2)
            _require(policy.get("quoted_charges", {}).get(line_id) == expected_charge, "refuel quote mismatch")
            _require(
                any(
                    bill.customer_id == customer_id and bill.status == BillStatus.DRAFT
                    for bill in db.bills
                ),
                "refuel tasks require a stable existing draft bill",
            )
            effect = (line_id, "refuel")
        elif action.name == "suspend_line":
            _require(line.status == LineStatus.ACTIVE, "only active lines can be suspended")
            _require(bool(str(arguments.get("reason", "")).strip()), "suspension reason required")
            effect = (line_id, "status")
        elif action.name == "resume_line":
            _require(resume_allowed(db, customer_id, line_id), "line is not eligible to resume")
            effect = (line_id, "status")
        else:
            raise ValueError(f"unsupported telecom Oracle action: {action.name}")

        _require(effect not in seen_line_effects, "duplicate state effect in one task")
        seen_line_effects.add(effect)

    _require(DETAIL_READ in required_reads, "account writes require line details")
