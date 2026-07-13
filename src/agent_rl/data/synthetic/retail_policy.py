"""Policy validation for clean-room retail task generation."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from tau2.domains.retail.data_model import GiftCard, Order, RetailDB, User
from tau2.domains.retail.utils import RETAIL_DB_PATH

if TYPE_CHECKING:
    from agent_rl.data.synthetic.generators.common import GeneratedCandidate


AUTH_TOOL = "find_user_id_by_email"
ORDER_READ_TOOL = "get_order_details"
PRODUCT_READ_TOOL = "get_product_details"
USER_READ_TOOL = "get_user_details"


@lru_cache(maxsize=1)
def load_retail_policy_db() -> RetailDB:
    """Load a read-only retail snapshot for independent policy checks."""

    return RetailDB.load(RETAIL_DB_PATH)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _item_counts_are_valid(order: Order, item_ids: list[str]) -> bool:
    requested = Counter(item_ids)
    available = Counter(item.item_id for item in order.items)
    return bool(item_ids) and all(
        count <= available[item_id] for item_id, count in requested.items()
    )


def _payment_for_difference(user: User, payment_id: str, difference: float) -> None:
    _require(payment_id in user.payment_methods, "payment method does not belong to user")
    payment = user.payment_methods[payment_id]
    if isinstance(payment, GiftCard):
        _require(
            payment.balance >= max(0.0, difference),
            "gift card cannot cover the positive price difference",
        )


def _validate_cancel(order: Order, arguments: dict[str, Any]) -> None:
    _require(order.status == "pending", "only pending orders can be cancelled")
    _require(
        arguments["reason"] in {"no longer needed", "ordered by mistake"},
        "invalid cancellation reason",
    )


def _validate_address_change(before: Any, arguments: dict[str, Any]) -> None:
    fields = ("address1", "address2", "city", "state", "country", "zip")
    _require(all(isinstance(arguments.get(field), str) for field in fields), "invalid address")
    after = {field: arguments[field] for field in fields}
    _require(any(after[field] != getattr(before, field) for field in fields), "address is unchanged")


def _validate_payment_change(order: Order, user: User, arguments: dict[str, Any]) -> None:
    _require("pending" in order.status, "only pending orders can change payment")
    _require(
        len(order.payment_history) == 1
        and order.payment_history[0].transaction_type == "payment",
        "pending payment change requires exactly one original payment",
    )
    payment_id = arguments["payment_method_id"]
    _require(payment_id in user.payment_methods, "new payment method does not belong to user")
    _require(
        payment_id != order.payment_history[0].payment_method_id,
        "new payment method must differ from the original",
    )
    payment = user.payment_methods[payment_id]
    if isinstance(payment, GiftCard):
        _require(
            payment.balance >= order.payment_history[0].amount,
            "gift card cannot cover the order total",
        )


def _validate_item_change(
    db: RetailDB,
    order: Order,
    user: User,
    arguments: dict[str, Any],
    *,
    delivered: bool,
) -> None:
    expected_status = "delivered" if delivered else "pending"
    _require(order.status == expected_status, f"order must be {expected_status}")
    item_ids = list(arguments["item_ids"])
    new_item_ids = list(arguments["new_item_ids"])
    _require(len(item_ids) == len(new_item_ids) and item_ids, "item lists must align")
    _require(_item_counts_are_valid(order, item_ids), "requested item is not in the order")

    difference = 0.0
    for old_id, new_id in zip(item_ids, new_item_ids, strict=True):
        _require(old_id != new_id, "replacement item must differ from original")
        old_item = next(item for item in order.items if item.item_id == old_id)
        product = db.products[old_item.product_id]
        _require(new_id in product.variants, "replacement must be from the same product")
        replacement = product.variants[new_id]
        _require(replacement.available, "replacement item is unavailable")
        difference += replacement.price - old_item.price

    _payment_for_difference(user, arguments["payment_method_id"], round(difference, 2))


def _validate_return(order: Order, user: User, arguments: dict[str, Any]) -> None:
    _require(order.status == "delivered", "only delivered orders can be returned")
    item_ids = list(arguments["item_ids"])
    _require(_item_counts_are_valid(order, item_ids), "returned item is not in the order")
    payment_id = arguments["payment_method_id"]
    _require(payment_id in user.payment_methods, "refund method does not belong to user")
    payment = user.payment_methods[payment_id]
    original_id = order.payment_history[0].payment_method_id
    _require(
        isinstance(payment, GiftCard) or payment_id == original_id,
        "refund must use the original payment method or an existing gift card",
    )


def validate_retail_candidate(
    candidate: GeneratedCandidate,
    database: RetailDB | None = None,
) -> None:
    """Recompute every policy-sensitive Retail Oracle argument from the DB."""

    policy = candidate.metadata.get("policy")
    _require(isinstance(policy, dict), "retail candidate requires policy metadata")
    _require(policy.get("required_confirmation") is True, "writes require confirmation")
    _require(
        policy.get("difficulty") in {"simple", "medium", "complex", "hard"},
        "invalid task difficulty",
    )
    required_reads = policy.get("required_reads")
    _require(isinstance(required_reads, list), "required_reads must be a list")
    _require(AUTH_TOOL in required_reads, "retail tasks must authenticate by email")

    criteria = candidate.task.evaluation_criteria
    actions = list(criteria.actions or []) if criteria else []
    names = [action.name for action in actions]
    _require(names == policy.get("expected_writes"), "expected_writes mismatch")
    _require(actions, "retail candidate requires at least one write")

    db = database if database is not None else load_retail_policy_db()
    expected_user_id = policy.get("user_id")
    _require(expected_user_id in db.users, "policy user does not exist")
    seen_order_ids: set[str] = set()
    action_user_ids: set[str] = set()

    for action in actions:
        arguments = action.arguments
        if action.name == "modify_user_address":
            user = db.users[arguments["user_id"]]
            action_user_ids.add(user.user_id)
            _require(USER_READ_TOOL in required_reads, "profile update requires a user read")
            _validate_address_change(user.address, arguments)
            continue

        order_id = arguments.get("order_id")
        _require(order_id in db.orders, "order does not exist")
        _require(order_id not in seen_order_ids, "only one write is allowed per order")
        seen_order_ids.add(order_id)
        order = db.orders[order_id]
        user = db.users[order.user_id]
        action_user_ids.add(user.user_id)
        _require(ORDER_READ_TOOL in required_reads, "order writes require an order read")

        if action.name == "cancel_pending_order":
            _validate_cancel(order, arguments)
        elif action.name == "modify_pending_order_address":
            _require("pending" in order.status, "only pending orders can change address")
            _validate_address_change(order.address, arguments)
        elif action.name == "modify_pending_order_payment":
            _validate_payment_change(order, user, arguments)
        elif action.name == "modify_pending_order_items":
            _require(PRODUCT_READ_TOOL in required_reads, "item changes require product reads")
            _validate_item_change(db, order, user, arguments, delivered=False)
        elif action.name == "return_delivered_order_items":
            _validate_return(order, user, arguments)
        elif action.name == "exchange_delivered_order_items":
            _require(PRODUCT_READ_TOOL in required_reads, "exchanges require product reads")
            _validate_item_change(db, order, user, arguments, delivered=True)
        else:
            raise ValueError(f"unsupported retail Oracle action: {action.name}")

    _require(action_user_ids == {expected_user_id}, "all requests must belong to one user")
    _require(
        sorted(seen_order_ids) == sorted(policy.get("affected_order_ids", [])),
        "affected_order_ids mismatch",
    )
