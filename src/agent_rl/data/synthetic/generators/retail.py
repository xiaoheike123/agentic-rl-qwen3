"""Procedural retail tasks derived from database entities, not benchmark tasks."""

from __future__ import annotations

import random

from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH

from agent_rl.data.synthetic.generators.common import (
    GeneratedCandidate,
    make_candidate,
)


def _address(index: int, seed: int) -> dict[str, str]:
    number = 100 + ((seed * 97 + index * 53) % 9800)
    return {
        "address1": f"{number} Synthetic Avenue",
        "address2": f"Unit {(index % 90) + 1}",
        "city": "Denver",
        "state": "CO",
        "country": "USA",
        "zip": f"{80000 + ((seed + index * 17) % 999):05d}",
    }


def generate_retail_candidates(seed: int) -> list[GeneratedCandidate]:
    rng = random.Random(seed)
    database = RetailDB.load(RETAIL_DB_PATH)
    orders = sorted(database.orders.values(), key=lambda item: item.order_id)
    users = sorted(database.users.values(), key=lambda item: item.user_id)
    rng.shuffle(orders)
    rng.shuffle(users)

    candidates: list[GeneratedCandidate] = []
    for index, order in enumerate(orders):
        if order.status != "pending":
            continue
        user = database.users[order.user_id]
        identity = (
            f"My user ID is {user.user_id}, my email is {user.email}, and the "
            f"order number is {order.order_id}."
        )
        reason = "ordered by mistake" if index % 2 else "no longer needed"
        candidates.append(
            make_candidate(
                domain="retail",
                template="cancel_pending_order",
                seed=seed + index * 2,
                entities=(user.user_id, order.order_id),
                reason_for_call=f"Cancel pending order {order.order_id}.",
                known_info=identity,
                task_instructions=(
                    f"Explain that the order was {reason}, request cancellation, "
                    "and explicitly confirm after the agent explains the refund."
                ),
                action_name="cancel_pending_order",
                action_arguments={"order_id": order.order_id, "reason": reason},
                communicate_info=[order.order_id, "cancelled"],
                purpose="Cancel a pending retail order with a valid reason.",
            )
        )

        new_address = _address(index, seed)
        candidates.append(
            make_candidate(
                domain="retail",
                template="modify_pending_order_address",
                seed=seed + index * 2 + 1,
                entities=(user.user_id, order.order_id),
                reason_for_call=f"Change the delivery address for {order.order_id}.",
                known_info=identity + f" The new address is {new_address}.",
                task_instructions=(
                    "Request only the shipping-address change and explicitly confirm "
                    "after the agent repeats the new address."
                ),
                action_name="modify_pending_order_address",
                action_arguments={"order_id": order.order_id, **new_address},
                communicate_info=[order.order_id, new_address["zip"]],
                purpose="Modify the shipping address of a pending order.",
            )
        )

    for index, user in enumerate(users):
        new_address = _address(index, seed + 10_000)
        candidates.append(
            make_candidate(
                domain="retail",
                template="modify_user_address",
                seed=seed + 100_000 + index,
                entities=(user.user_id,),
                reason_for_call="Update the default address saved on the account.",
                known_info=(
                    f"My user ID is {user.user_id}, my email is {user.email}, and "
                    f"the new default address is {new_address}."
                ),
                task_instructions=(
                    "Request the account-level address update, not an individual order "
                    "change, and explicitly confirm after the agent repeats it."
                ),
                action_name="modify_user_address",
                action_arguments={"user_id": user.user_id, **new_address},
                communicate_info=[user.user_id, new_address["zip"]],
                purpose="Update a customer's default retail address.",
            )
        )
    return candidates
