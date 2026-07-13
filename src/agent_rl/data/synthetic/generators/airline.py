"""Procedural airline tasks derived from database entities, not benchmark tasks."""

from __future__ import annotations

import random

from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH

from agent_rl.data.synthetic.generators.common import (
    GeneratedCandidate,
    make_candidate,
)


def generate_airline_candidates(seed: int) -> list[GeneratedCandidate]:
    rng = random.Random(seed)
    database = FlightDB.load(AIRLINE_DB_PATH)
    reservations = [
        reservation
        for reservation in database.reservations.values()
        if reservation.status != "cancelled"
    ]
    reservations.sort(key=lambda item: item.reservation_id)
    rng.shuffle(reservations)

    candidates: list[GeneratedCandidate] = []
    for index, reservation in enumerate(reservations):
        user = database.users[reservation.user_id]
        identity = (
            f"My user ID is {user.user_id}, my email is {user.email}, and my date "
            f"of birth is {user.dob}. The reservation is {reservation.reservation_id}."
        )
        candidates.append(
            make_candidate(
                domain="airline",
                template="cancel_reservation",
                seed=seed + index * 2,
                entities=(user.user_id, reservation.reservation_id),
                reason_for_call=(
                    f"Cancel reservation {reservation.reservation_id} because the trip "
                    "is no longer needed."
                ),
                known_info=identity,
                task_instructions=(
                    "Ask the agent to cancel the entire reservation. Provide identity "
                    "details when requested and explicitly confirm the cancellation "
                    "after the agent explains it. Do not request any other change."
                ),
                action_name="cancel_reservation",
                action_arguments={"reservation_id": reservation.reservation_id},
                communicate_info=[reservation.reservation_id, "cancelled"],
                purpose="Cancel one existing reservation after normal confirmation.",
            )
        )

        payment_ids = sorted(user.payment_methods)
        if payment_ids:
            new_total = reservation.total_baggages + 1
            candidates.append(
                make_candidate(
                    domain="airline",
                    template="add_included_baggage",
                    seed=seed + index * 2 + 1,
                    entities=(
                        user.user_id,
                        reservation.reservation_id,
                        payment_ids[0],
                    ),
                    reason_for_call=(
                        f"Add one bag to reservation {reservation.reservation_id} "
                        "without changing its current paid-bag count."
                    ),
                    known_info=identity + f" Use saved payment method {payment_ids[0]}.",
                    task_instructions=(
                        f"Request {new_total} total bags while keeping the number of "
                        f"non-free bags at {reservation.nonfree_baggages}. Confirm the "
                        "change if asked and do not modify flights or passengers."
                    ),
                    action_name="update_reservation_baggages",
                    action_arguments={
                        "reservation_id": reservation.reservation_id,
                        "total_baggages": new_total,
                        "nonfree_baggages": reservation.nonfree_baggages,
                        "payment_id": payment_ids[0],
                    },
                    communicate_info=[reservation.reservation_id, "baggage"],
                    purpose="Update baggage quantity on an existing reservation.",
                )
            )
    return candidates
