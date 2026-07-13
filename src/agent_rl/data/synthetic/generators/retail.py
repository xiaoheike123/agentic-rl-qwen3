"""Clean-room retail tasks derived only from policy, tools, and database state."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from tau2.domains.retail.data_model import GiftCard, Order, RetailDB, User
from tau2.domains.retail.utils import RETAIL_DB_PATH

from agent_rl.data.synthetic.generators.common import (
    GeneratedCandidate,
    OracleActionSpec,
    make_candidate,
)


FAMILY_LIMIT = 96


def _address(index: int, seed: int) -> dict[str, str]:
    number = 100 + ((seed * 97 + index * 53) % 9800)
    return {
        "address1": f"{number} Research Avenue",
        "address2": f"Unit {(index % 90) + 1}",
        "city": "Denver",
        "state": "CO",
        "country": "USA",
        "zip": f"{80000 + ((seed + index * 17) % 999):05d}",
    }


def _identity(user: User, order_ids: tuple[str, ...] = ()) -> str:
    orders = ""
    if order_ids:
        orders = " The relevant order numbers are " + ", ".join(order_ids) + "."
    return f"My account email is {user.email}.{orders}"


def _valid_difference_payment(user: User, difference: float) -> str | None:
    methods = sorted(user.payment_methods.items())
    for payment_id, payment in methods:
        if not isinstance(payment, GiftCard):
            return payment_id
    for payment_id, payment in methods:
        if isinstance(payment, GiftCard) and payment.balance >= max(0.0, difference):
            return payment_id
    return None


def _alternative_payment(order: Order, user: User) -> str | None:
    if len(order.payment_history) != 1:
        return None
    original = order.payment_history[0]
    if original.transaction_type != "payment":
        return None
    for payment_id, payment in sorted(user.payment_methods.items()):
        if payment_id == original.payment_method_id:
            continue
        if isinstance(payment, GiftCard) and payment.balance < original.amount:
            continue
        return payment_id
    return None


def _refund_payment(order: Order, user: User) -> str | None:
    if not order.payment_history:
        return None
    original_id = order.payment_history[0].payment_method_id
    if original_id in user.payment_methods:
        return original_id
    for payment_id, payment in sorted(user.payment_methods.items()):
        if isinstance(payment, GiftCard):
            return payment_id
    return None


def _replacement_pairs(
    db: RetailDB,
    order: Order,
    *,
    maximum: int,
) -> tuple[list[str], list[str], float] | None:
    old_ids: list[str] = []
    new_ids: list[str] = []
    difference = 0.0
    for item in order.items:
        alternatives = [
            variant
            for variant in db.products[item.product_id].variants.values()
            if variant.available and variant.item_id != item.item_id
        ]
        if not alternatives:
            continue
        replacement = sorted(alternatives, key=lambda value: value.item_id)[0]
        old_ids.append(item.item_id)
        new_ids.append(replacement.item_id)
        difference += replacement.price - item.price
        if len(old_ids) == maximum:
            break
    if not old_ids:
        return None
    return old_ids, new_ids, round(difference, 2)


def _replacement_difference(
    db: RetailDB,
    order: Order,
    old_item_id: str,
    new_item_id: str,
) -> float:
    old_item = next(item for item in order.items if item.item_id == old_item_id)
    replacement = db.products[old_item.product_id].variants[new_item_id]
    return round(replacement.price - old_item.price, 2)


def _policy_metadata(
    *,
    case: str,
    difficulty: str,
    reads: tuple[str, ...],
    writes: tuple[str, ...],
    user: User,
    order_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "policy": {
            "case": case,
            "difficulty": difficulty,
            "required_reads": list(dict.fromkeys(reads)),
            "required_confirmation": True,
            "expected_writes": list(writes),
            "user_id": user.user_id,
            "affected_order_ids": list(order_ids),
        }
    }


def generate_retail_candidates(
    seed: int,
    database: RetailDB | None = None,
) -> list[GeneratedCandidate]:
    rng = random.Random(seed)
    db = database if database is not None else RetailDB.load(RETAIL_DB_PATH)
    orders = sorted(db.orders.values(), key=lambda item: item.order_id)
    users = sorted(db.users.values(), key=lambda item: item.user_id)
    rng.shuffle(orders)
    rng.shuffle(users)
    families: dict[str, list[GeneratedCandidate]] = {}

    def add(
        *,
        template: str,
        user: User,
        order_ids: tuple[str, ...],
        reason_for_call: str,
        known_info: str,
        task_instructions: str,
        actions: tuple[OracleActionSpec, ...],
        communicate_info: list[str],
        purpose: str,
        difficulty: str,
        reads: tuple[str, ...],
    ) -> None:
        pool = families.setdefault(template, [])
        if len(pool) >= FAMILY_LIMIT:
            return
        pool.append(
            make_candidate(
                domain="retail",
                template=template,
                seed=seed + len(pool) + len(families) * 10_003,
                entities=(user.user_id, *order_ids),
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
                    writes=tuple(action.name for action in actions),
                    user=user,
                    order_ids=order_ids,
                ),
            )
        )

    pending_by_user: dict[str, list[Order]] = defaultdict(list)
    delivered_by_user: dict[str, list[Order]] = defaultdict(list)

    for index, order in enumerate(orders):
        user = db.users[order.user_id]
        if order.status == "pending":
            pending_by_user[user.user_id].append(order)
            reason = "ordered by mistake" if index % 2 else "no longer needed"
            add(
                template="cancel_pending_order",
                user=user,
                order_ids=(order.order_id,),
                reason_for_call=f"Cancel order {order.order_id} because it was {reason}.",
                known_info=_identity(user, (order.order_id,)),
                task_instructions="Confirm only after the agent states the order, reason, and refund timing.",
                actions=(OracleActionSpec("cancel_pending_order", {"order_id": order.order_id, "reason": reason}),),
                communicate_info=[order.order_id, "cancelled", "5 to 7 business days"],
                purpose="Authenticate and cancel an eligible pending order.",
                difficulty="simple",
                reads=("find_user_id_by_email", "get_order_details"),
            )

            address = _address(index, seed)
            add(
                template="modify_pending_order_address",
                user=user,
                order_ids=(order.order_id,),
                reason_for_call=f"Change the shipping address for pending order {order.order_id}.",
                known_info=_identity(user, (order.order_id,)) + f" The complete new address is {address}.",
                task_instructions="Confirm after the agent repeats the complete address and makes clear this affects only the order.",
                actions=(OracleActionSpec("modify_pending_order_address", {"order_id": order.order_id, **address}),),
                communicate_info=[order.order_id, address["zip"]],
                purpose="Modify only the shipping address of a pending order.",
                difficulty="medium",
                reads=("find_user_id_by_email", "get_order_details"),
            )

            payment_id = _alternative_payment(order, user)
            if payment_id is not None:
                add(
                    template="modify_pending_order_payment",
                    user=user,
                    order_ids=(order.order_id,),
                    reason_for_call=f"Move payment for order {order.order_id} to {payment_id}.",
                    known_info=_identity(user, (order.order_id,)) + f" Use payment method {payment_id}.",
                    task_instructions="Confirm after the agent identifies both the new charge and original-method refund behavior.",
                    actions=(OracleActionSpec("modify_pending_order_payment", {"order_id": order.order_id, "payment_method_id": payment_id}),),
                    communicate_info=[order.order_id, payment_id, "5 to 7 business days"],
                    purpose="Change a pending order to a different valid payment method.",
                    difficulty="medium",
                    reads=("find_user_id_by_email", "get_order_details", "get_user_details"),
                )

            replacements = _replacement_pairs(db, order, maximum=2)
            if replacements is not None:
                old_ids, new_ids, difference = replacements
                payment_id = _valid_difference_payment(user, difference)
                if payment_id is not None:
                    single_old, single_new = old_ids[:1], new_ids[:1]
                    single_difference = _replacement_difference(
                        db,
                        order,
                        single_old[0],
                        single_new[0],
                    )
                    single_payment = _valid_difference_payment(user, single_difference)
                    if single_payment is not None:
                        add(
                            template="modify_single_pending_item",
                            user=user,
                            order_ids=(order.order_id,),
                            reason_for_call=f"Change one item option in pending order {order.order_id}.",
                            known_info=_identity(user, (order.order_id,)) + f" Replace item {single_old[0]} with {single_new[0]} and use {single_payment} for the difference.",
                            task_instructions="Confirm only after the agent checks the product match and reminds you to include every desired item change.",
                            actions=(OracleActionSpec("modify_pending_order_items", {"order_id": order.order_id, "item_ids": single_old, "new_item_ids": single_new, "payment_method_id": single_payment}),),
                            communicate_info=[order.order_id, single_new[0]],
                            purpose="Modify one pending-order item to an available variant of the same product.",
                            difficulty="complex",
                            reads=("find_user_id_by_email", "get_order_details", "get_product_details"),
                        )
                    if len(old_ids) >= 2:
                        add(
                            template="modify_multiple_pending_items",
                            user=user,
                            order_ids=(order.order_id,),
                            reason_for_call=f"Change two item options together in pending order {order.order_id}.",
                            known_info=_identity(user, (order.order_id,)) + f" Replace {old_ids} with {new_ids} in that order and use {payment_id} for the net difference.",
                            task_instructions="Confirm only after every old-to-new pair and the one-call finality have been summarized.",
                            actions=(OracleActionSpec("modify_pending_order_items", {"order_id": order.order_id, "item_ids": old_ids, "new_item_ids": new_ids, "payment_method_id": payment_id}),),
                            communicate_info=[order.order_id, *new_ids],
                            purpose="Modify multiple pending-order variants in one atomic tool call.",
                            difficulty="hard",
                            reads=("find_user_id_by_email", "get_order_details", "get_product_details"),
                        )

        if order.status == "delivered":
            delivered_by_user[user.user_id].append(order)
            payment_id = _refund_payment(order, user)
            if payment_id is not None:
                add(
                    template="return_single_delivered_item",
                    user=user,
                    order_ids=(order.order_id,),
                    reason_for_call=f"Return one item from delivered order {order.order_id}.",
                    known_info=_identity(user, (order.order_id,)) + f" Return item {order.items[0].item_id} and refund {payment_id}.",
                    task_instructions="Confirm after the agent states the item, refund destination, and follow-up email process.",
                    actions=(OracleActionSpec("return_delivered_order_items", {"order_id": order.order_id, "item_ids": [order.items[0].item_id], "payment_method_id": payment_id}),),
                    communicate_info=[order.order_id, order.items[0].item_id, "email"],
                    purpose="Return one item from a delivered order to a permitted refund method.",
                    difficulty="medium",
                    reads=("find_user_id_by_email", "get_order_details"),
                )
                if len(order.items) >= 2:
                    item_ids = [item.item_id for item in order.items]
                    add(
                        template="return_multiple_delivered_items",
                        user=user,
                        order_ids=(order.order_id,),
                        reason_for_call=f"Return every item from delivered order {order.order_id}.",
                        known_info=_identity(user, (order.order_id,)) + f" Return item IDs {item_ids} and refund {payment_id}.",
                        task_instructions="Confirm only after the complete item list and refund destination are repeated.",
                        actions=(OracleActionSpec("return_delivered_order_items", {"order_id": order.order_id, "item_ids": item_ids, "payment_method_id": payment_id}),),
                        communicate_info=[order.order_id, *item_ids, "email"],
                        purpose="Return multiple delivered items together without dropping duplicates.",
                        difficulty="complex",
                        reads=("find_user_id_by_email", "get_order_details"),
                    )

            replacements = _replacement_pairs(db, order, maximum=2)
            if replacements is not None:
                old_ids, new_ids, difference = replacements
                single_difference = _replacement_difference(
                    db,
                    order,
                    old_ids[0],
                    new_ids[0],
                )
                single_payment_id = _valid_difference_payment(user, single_difference)
                payment_id = _valid_difference_payment(user, difference)
                if single_payment_id is not None:
                    add(
                        template="exchange_single_delivered_item",
                        user=user,
                        order_ids=(order.order_id,),
                        reason_for_call=f"Exchange one item from delivered order {order.order_id}.",
                        known_info=_identity(user, (order.order_id,)) + f" Exchange {old_ids[0]} for {new_ids[0]} and use {single_payment_id} for the difference.",
                        task_instructions="Confirm after the agent validates the same-product option and explains the return email.",
                        actions=(OracleActionSpec("exchange_delivered_order_items", {"order_id": order.order_id, "item_ids": old_ids[:1], "new_item_ids": new_ids[:1], "payment_method_id": single_payment_id}),),
                        communicate_info=[order.order_id, new_ids[0], "email"],
                        purpose="Exchange one delivered item for an available same-product variant.",
                        difficulty="complex",
                        reads=("find_user_id_by_email", "get_order_details", "get_product_details"),
                    )
                if payment_id is not None and len(old_ids) >= 2:
                        add(
                            template="exchange_multiple_delivered_items",
                            user=user,
                            order_ids=(order.order_id,),
                            reason_for_call=f"Exchange two delivered items together in order {order.order_id}.",
                            known_info=_identity(user, (order.order_id,)) + f" Exchange {old_ids} for {new_ids} in that order and use {payment_id} for the net difference.",
                            task_instructions="Confirm only after all pairs, payment handling, and the one-call requirement are clear.",
                            actions=(OracleActionSpec("exchange_delivered_order_items", {"order_id": order.order_id, "item_ids": old_ids, "new_item_ids": new_ids, "payment_method_id": payment_id}),),
                            communicate_info=[order.order_id, *new_ids, "email"],
                            purpose="Exchange multiple delivered items in one atomic request.",
                            difficulty="hard",
                            reads=("find_user_id_by_email", "get_order_details", "get_product_details"),
                        )

    for index, user in enumerate(users):
        address = _address(index, seed + 100_000)
        add(
            template="modify_default_address",
            user=user,
            order_ids=(),
            reason_for_call="Update the default shipping address saved on my account.",
            known_info=_identity(user) + f" The complete new default address is {address}.",
            task_instructions="Confirm after the agent repeats the address and clarifies that existing orders are unaffected.",
            actions=(OracleActionSpec("modify_user_address", {"user_id": user.user_id, **address}),),
            communicate_info=[user.user_id, address["zip"]],
            purpose="Update the authenticated user's default address.",
            difficulty="simple",
            reads=("find_user_id_by_email", "get_user_details"),
        )

        pending = pending_by_user.get(user.user_id, [])
        delivered = delivered_by_user.get(user.user_id, [])
        if pending:
            order = pending[0]
            reason = "no longer needed"
            address = _address(index, seed + 200_000)
            add(
                template="cancel_order_and_modify_default_address",
                user=user,
                order_ids=(order.order_id,),
                reason_for_call=f"Cancel {order.order_id} and then update my account's default address.",
                known_info=_identity(user, (order.order_id,)) + f" The cancellation reason is {reason}; the new default address is {address}.",
                task_instructions="Give a separate explicit confirmation when the agent summarizes each database-changing action.",
                actions=(OracleActionSpec("cancel_pending_order", {"order_id": order.order_id, "reason": reason}), OracleActionSpec("modify_user_address", {"user_id": user.user_id, **address})),
                communicate_info=[order.order_id, "cancelled", address["zip"]],
                purpose="Complete two independent requests for the same authenticated user.",
                difficulty="complex",
                reads=("find_user_id_by_email", "get_order_details", "get_user_details"),
            )

            payment_id = _alternative_payment(order, user)
            if payment_id is not None:
                address = _address(index, seed + 300_000)
                add(
                    template="change_payment_and_modify_default_address",
                    user=user,
                    order_ids=(order.order_id,),
                    reason_for_call=f"Change payment for {order.order_id} and update my default address.",
                    known_info=_identity(user, (order.order_id,)) + f" Use {payment_id}; the new default address is {address}.",
                    task_instructions="Confirm each change only after the agent separates order payment effects from the account address update.",
                    actions=(OracleActionSpec("modify_pending_order_payment", {"order_id": order.order_id, "payment_method_id": payment_id}), OracleActionSpec("modify_user_address", {"user_id": user.user_id, **address})),
                    communicate_info=[order.order_id, payment_id, address["zip"]],
                    purpose="Handle an order payment change and an account-level address update.",
                    difficulty="hard",
                    reads=("find_user_id_by_email", "get_order_details", "get_user_details"),
                )

        if delivered:
            order = delivered[0]
            payment_id = _refund_payment(order, user)
            if payment_id is not None:
                address = _address(index, seed + 400_000)
                add(
                    template="return_item_and_modify_default_address",
                    user=user,
                    order_ids=(order.order_id,),
                    reason_for_call=f"Return an item from {order.order_id} and update my default address.",
                    known_info=_identity(user, (order.order_id,)) + f" Return {order.items[0].item_id} to {payment_id}; the new default address is {address}.",
                    task_instructions="Confirm the return and address update separately after both consequences are explained.",
                    actions=(OracleActionSpec("return_delivered_order_items", {"order_id": order.order_id, "item_ids": [order.items[0].item_id], "payment_method_id": payment_id}), OracleActionSpec("modify_user_address", {"user_id": user.user_id, **address})),
                    communicate_info=[order.order_id, order.items[0].item_id, address["zip"]],
                    purpose="Complete a delivered-item return and an independent profile update.",
                    difficulty="complex",
                    reads=("find_user_id_by_email", "get_order_details", "get_user_details"),
                )

            replacements = _replacement_pairs(db, order, maximum=1)
            if replacements is not None:
                old_ids, new_ids, difference = replacements
                payment_id = _valid_difference_payment(user, difference)
                if payment_id is not None:
                    address = _address(index, seed + 500_000)
                    add(
                        template="exchange_item_and_modify_default_address",
                        user=user,
                        order_ids=(order.order_id,),
                        reason_for_call=f"Exchange an item from {order.order_id} and update my default address.",
                        known_info=_identity(user, (order.order_id,)) + f" Exchange {old_ids[0]} for {new_ids[0]} using {payment_id}; the new default address is {address}.",
                        task_instructions="Confirm the exchange and profile update separately after all option and payment details are checked.",
                        actions=(OracleActionSpec("exchange_delivered_order_items", {"order_id": order.order_id, "item_ids": old_ids, "new_item_ids": new_ids, "payment_method_id": payment_id}), OracleActionSpec("modify_user_address", {"user_id": user.user_id, **address})),
                        communicate_info=[order.order_id, new_ids[0], address["zip"]],
                        purpose="Handle an exchange plus an unrelated profile update for one user.",
                        difficulty="hard",
                        reads=("find_user_id_by_email", "get_order_details", "get_product_details", "get_user_details"),
                    )

        if pending and delivered:
            pending_order = pending[0]
            delivered_order = delivered[0]
            payment_id = _refund_payment(delivered_order, user)
            if payment_id is not None:
                add(
                    template="cancel_and_return_two_orders",
                    user=user,
                    order_ids=(pending_order.order_id, delivered_order.order_id),
                    reason_for_call=f"Cancel {pending_order.order_id} and return one item from {delivered_order.order_id}.",
                    known_info=_identity(user, (pending_order.order_id, delivered_order.order_id)) + f" The cancellation reason is ordered by mistake. Return {delivered_order.items[0].item_id} to {payment_id}.",
                    task_instructions="Confirm each order action only after the agent keeps the two orders and their consequences distinct.",
                    actions=(OracleActionSpec("cancel_pending_order", {"order_id": pending_order.order_id, "reason": "ordered by mistake"}), OracleActionSpec("return_delivered_order_items", {"order_id": delivered_order.order_id, "item_ids": [delivered_order.items[0].item_id], "payment_method_id": payment_id})),
                    communicate_info=[pending_order.order_id, "cancelled", delivered_order.order_id, "email"],
                    purpose="Resolve two different order requests for the same authenticated user.",
                    difficulty="hard",
                    reads=("find_user_id_by_email", "get_order_details"),
                )

        if len(pending) >= 2:
            first, second = pending[:2]
            add(
                template="cancel_two_pending_orders",
                user=user,
                order_ids=(first.order_id, second.order_id),
                reason_for_call=f"Cancel both pending orders {first.order_id} and {second.order_id}.",
                known_info=_identity(user, (first.order_id, second.order_id)) + " Both were ordered by mistake.",
                task_instructions="Confirm each cancellation after the agent verifies both order statuses and keeps their refunds separate.",
                actions=(OracleActionSpec("cancel_pending_order", {"order_id": first.order_id, "reason": "ordered by mistake"}), OracleActionSpec("cancel_pending_order", {"order_id": second.order_id, "reason": "ordered by mistake"})),
                communicate_info=[first.order_id, second.order_id, "cancelled"],
                purpose="Cancel two independently eligible orders for one user.",
                difficulty="hard",
                reads=("find_user_id_by_email", "get_order_details"),
            )

    return [candidate for template in sorted(families) for candidate in families[template]]
