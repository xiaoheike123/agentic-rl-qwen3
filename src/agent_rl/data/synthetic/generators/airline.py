"""Clean-room airline tasks derived only from policy, tools, and database state."""

from __future__ import annotations

import random
from copy import deepcopy
from datetime import datetime
from typing import Any

from tau2.domains.airline.data_model import (
    FlightDB,
    FlightDateStatusAvailable,
    Passenger,
    Reservation,
    User,
)
from tau2.domains.airline.utils import AIRLINE_DB_PATH

from agent_rl.data.synthetic.airline_policy import (
    CURRENT_TIME,
    cancellation_allowed,
    expected_nonfree_baggages,
    has_flown_segment,
    reservation_statuses,
    valid_update_payment_ids,
)
from agent_rl.data.synthetic.generators.common import (
    GeneratedCandidate,
    OracleActionSpec,
    make_candidate,
)


FAMILY_LIMIT = 96
REPLACEMENT_LAST_NAMES = ("Bennett", "Chandra", "Morales", "Okafor", "Sato")


def _identity(user: User, reservation: Reservation) -> str:
    return (
        f"My user ID is {user.user_id}, my email is {user.email}, and my date "
        f"of birth is {user.dob}. The reservation is {reservation.reservation_id}."
    )


def _passenger_update(reservation: Reservation, index: int) -> list[dict[str, str]]:
    passengers = [item.model_dump(mode="json") for item in reservation.passengers]
    replacement = REPLACEMENT_LAST_NAMES[index % len(REPLACEMENT_LAST_NAMES)]
    if replacement == passengers[0]["last_name"]:
        replacement = REPLACEMENT_LAST_NAMES[(index + 1) % len(REPLACEMENT_LAST_NAMES)]
    passengers[0]["last_name"] = replacement
    return passengers


def _credit_card_ids(user: User) -> tuple[str, ...]:
    return tuple(
        payment_id
        for payment_id, method in sorted(user.payment_methods.items())
        if method.source == "credit_card"
    )


def _replacement_flights(
    db: FlightDB,
    reservation: Reservation,
) -> tuple[list[dict[str, str]], int] | None:
    if reservation.cabin == "basic_economy" or has_flown_segment(db, reservation):
        return None

    for segment_index, old in enumerate(reservation.flights):
        alternatives = []
        for flight in db.flights.values():
            if flight.flight_number == old.flight_number:
                continue
            if (flight.origin, flight.destination) != (old.origin, old.destination):
                continue
            state = flight.dates.get(old.date)
            if not isinstance(state, FlightDateStatusAvailable):
                continue
            if state.available_seats[reservation.cabin] < len(reservation.passengers):
                continue
            alternatives.append(flight)
        if not alternatives:
            continue
        selected = sorted(alternatives, key=lambda item: item.flight_number)[0]
        flights = [
            {"flight_number": item.flight_number, "date": item.date}
            for item in reservation.flights
        ]
        flights[segment_index] = {
            "flight_number": selected.flight_number,
            "date": old.date,
        }
        state = selected.dates[old.date]
        assert isinstance(state, FlightDateStatusAvailable)
        price_delta = (
            state.prices[reservation.cabin] - old.price
        ) * len(reservation.passengers)
        return flights, price_delta
    return None


def _cabin_upgrade(
    db: FlightDB,
    reservation: Reservation,
) -> tuple[str, list[dict[str, str]], int] | None:
    if has_flown_segment(db, reservation) or reservation.cabin == "business":
        return None
    new_cabin = "economy" if reservation.cabin == "basic_economy" else "business"
    new_total = 0
    flights: list[dict[str, str]] = []
    for segment in reservation.flights:
        state = db.flights[segment.flight_number].dates[segment.date]
        if not isinstance(state, FlightDateStatusAvailable):
            return None
        if state.available_seats[new_cabin] < len(reservation.passengers):
            return None
        new_total += state.prices[new_cabin] * len(reservation.passengers)
        flights.append({"flight_number": segment.flight_number, "date": segment.date})
    old_total = sum(item.price for item in reservation.flights) * len(
        reservation.passengers
    )
    return new_cabin, flights, new_total - old_total


def _policy_metadata(
    *,
    case: str,
    difficulty: str,
    reads: tuple[str, ...],
    writes: tuple[str, ...],
    reservation: Reservation,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "policy": {
            "case": case,
            "difficulty": difficulty,
            "required_reads": list(reads),
            "required_confirmation": True,
            "expected_writes": list(writes),
            "reservation_id": reservation.reservation_id,
            **extra,
        }
    }


def generate_airline_candidates(
    seed: int,
    database: FlightDB | None = None,
) -> list[GeneratedCandidate]:
    rng = random.Random(seed)
    db = database if database is not None else FlightDB.load(AIRLINE_DB_PATH)
    reservations = sorted(db.reservations.values(), key=lambda item: item.reservation_id)
    rng.shuffle(reservations)
    families: dict[str, list[GeneratedCandidate]] = {}

    def add(
        *,
        template: str,
        reservation: Reservation,
        user: User,
        reason_for_call: str,
        known_info: str,
        task_instructions: str,
        actions: tuple[OracleActionSpec, ...],
        communicate_info: list[str],
        purpose: str,
        difficulty: str,
        reads: tuple[str, ...],
        policy_extra: dict[str, Any] | None = None,
        entities: tuple[str, ...] = (),
    ) -> None:
        pool = families.setdefault(template, [])
        if len(pool) >= FAMILY_LIMIT:
            return
        pool.append(
            make_candidate(
                domain="airline",
                template=template,
                seed=seed + len(pool) + len(families) * 10_003,
                entities=(user.user_id, reservation.reservation_id, *entities),
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
                    reservation=reservation,
                    **(policy_extra or {}),
                ),
            )
        )

    for index, reservation in enumerate(reservations):
        if reservation.status == "cancelled":
            continue
        user = db.users[reservation.user_id]
        identity = _identity(user, reservation)
        statuses = reservation_statuses(db, reservation)
        not_flown = not has_flown_segment(db, reservation)
        created_within_24h = (
            CURRENT_TIME - datetime.fromisoformat(reservation.created_at)
        ).total_seconds() <= 24 * 60 * 60

        if created_within_24h and cancellation_allowed(
            db, reservation, "change_of_plan"
        ):
            add(
                template="cancel_within_24h",
                reservation=reservation,
                user=user,
                reason_for_call=f"Cancel reservation {reservation.reservation_id}; my plans changed.",
                known_info=identity,
                task_instructions="Request cancellation, state the change-of-plan reason, and confirm after the agent summarizes the refund.",
                actions=(OracleActionSpec("cancel_reservation", {"reservation_id": reservation.reservation_id}),),
                communicate_info=[reservation.reservation_id, "cancelled", "5 to 7 business days"],
                purpose="Cancel an eligible reservation created within 24 hours.",
                difficulty="simple",
                reads=("get_reservation_details",),
                policy_extra={"cancellation_reason": "change_of_plan"},
            )

        if reservation.cabin == "business" and not_flown:
            add(
                template="cancel_business_cabin",
                reservation=reservation,
                user=user,
                reason_for_call=f"Cancel my business-cabin reservation {reservation.reservation_id} because my plans changed.",
                known_info=identity,
                task_instructions="Request cancellation and explicitly confirm only after the agent explains the action.",
                actions=(OracleActionSpec("cancel_reservation", {"reservation_id": reservation.reservation_id}),),
                communicate_info=[reservation.reservation_id, "cancelled"],
                purpose="Cancel an unflown business-cabin reservation.",
                difficulty="simple",
                reads=("get_reservation_details",),
                policy_extra={"cancellation_reason": "change_of_plan"},
            )

        if reservation.insurance == "yes" and not_flown:
            add(
                template="cancel_insured_health",
                reservation=reservation,
                user=user,
                reason_for_call=f"Cancel reservation {reservation.reservation_id} for a health reason covered by my insurance.",
                known_info=identity,
                task_instructions="Explain that a health issue prevents travel, request cancellation, and explicitly confirm it.",
                actions=(OracleActionSpec("cancel_reservation", {"reservation_id": reservation.reservation_id}),),
                communicate_info=[reservation.reservation_id, "cancelled", "refund"],
                purpose="Cancel an insured, unflown reservation for a covered health reason.",
                difficulty="medium",
                reads=("get_reservation_details",),
                policy_extra={"cancellation_reason": "health"},
            )

        compensation_eligible = (
            user.membership in {"silver", "gold"}
            or reservation.insurance == "yes"
            or reservation.cabin == "business"
        )
        if "cancelled" in statuses and not_flown and compensation_eligible:
            amount = 100 * len(reservation.passengers)
            add(
                template="cancel_and_compensate_cancelled_flight",
                reservation=reservation,
                user=user,
                reason_for_call=f"Flight disruption affected reservation {reservation.reservation_id}; cancel it and provide the allowed compensation.",
                known_info=identity,
                task_instructions="Explicitly request both cancellation and compensation for the airline-cancelled flight, then confirm the database-changing actions.",
                actions=(
                    OracleActionSpec("cancel_reservation", {"reservation_id": reservation.reservation_id}),
                    OracleActionSpec("send_certificate", {"user_id": user.user_id, "amount": amount}),
                ),
                communicate_info=[reservation.reservation_id, "cancelled", f"${amount}", "certificate"],
                purpose="Cancel an airline-cancelled itinerary and issue policy-compliant compensation.",
                difficulty="hard",
                reads=("get_reservation_details", "get_user_details", "get_flight_status"),
                policy_extra={"cancellation_reason": "airline_cancelled", "compensation_requested": True},
            )

        new_total = reservation.total_baggages + 1
        new_nonfree = expected_nonfree_baggages(user, reservation, new_total)
        baggage_charge = 50 * max(0, new_nonfree - reservation.nonfree_baggages)
        payment_ids = valid_update_payment_ids(user, baggage_charge)
        baggage_action: OracleActionSpec | None = None
        payment_id: str | None = payment_ids[0] if payment_ids else None
        if payment_id and new_nonfree in {
            reservation.nonfree_baggages,
            reservation.nonfree_baggages + 1,
        }:
            baggage_action = OracleActionSpec(
                "update_reservation_baggages",
                {
                    "reservation_id": reservation.reservation_id,
                    "total_baggages": new_total,
                    "nonfree_baggages": new_nonfree,
                    "payment_id": payment_id,
                },
            )
            bag_kind = "free" if baggage_charge == 0 else "paid"
            add(
                template=f"add_{bag_kind}_baggage",
                reservation=reservation,
                user=user,
                reason_for_call=f"Add one checked bag to reservation {reservation.reservation_id}.",
                known_info=identity + f" Use saved payment method {payment_id} if payment is required.",
                task_instructions="Request exactly one additional checked bag. Do not tell the agent whether it is free; let the agent calculate the allowance, then confirm the change.",
                actions=(baggage_action,),
                communicate_info=[reservation.reservation_id, f"{new_total} total", f"{new_nonfree} non-free"],
                purpose=f"Add one {bag_kind} checked bag using the correct allowance calculation.",
                difficulty="medium",
                reads=("get_reservation_details", "get_user_details"),
                entities=(payment_id,),
            )

        passengers = _passenger_update(reservation, index)
        passenger_action = OracleActionSpec(
            "update_reservation_passengers",
            {"reservation_id": reservation.reservation_id, "passengers": passengers},
        )
        corrected_name = passengers[0]["last_name"]
        add(
            template="correct_passenger_name",
            reservation=reservation,
            user=user,
            reason_for_call=f"Correct the first passenger's last name on reservation {reservation.reservation_id} to {corrected_name}.",
            known_info=identity,
            task_instructions=f"Request only the last-name correction to {corrected_name}, preserve every passenger and other field, and confirm the update.",
            actions=(passenger_action,),
            communicate_info=[reservation.reservation_id, corrected_name],
            purpose="Correct one passenger field without changing passenger count.",
            difficulty="medium",
            reads=("get_reservation_details",),
        )

        if baggage_action is not None:
            add(
                template="correct_passenger_and_add_baggage",
                reservation=reservation,
                user=user,
                reason_for_call=f"On reservation {reservation.reservation_id}, correct the first passenger's last name to {corrected_name} and add one checked bag.",
                known_info=identity + f" Use saved payment method {payment_id} if needed.",
                task_instructions="Request both changes, preserve all unspecified fields, and explicitly confirm both after the agent summarizes them.",
                actions=(passenger_action, baggage_action),
                communicate_info=[reservation.reservation_id, corrected_name, f"{new_total} total bags"],
                purpose="Perform two independent, policy-compliant reservation updates.",
                difficulty="complex",
                reads=("get_reservation_details", "get_user_details"),
                entities=(payment_id or "",),
            )

        replacement = _replacement_flights(db, reservation)
        if replacement is not None:
            flights, price_delta = replacement
            flight_payment_ids = valid_update_payment_ids(user, max(0, price_delta))
            if flight_payment_ids:
                flight_payment = flight_payment_ids[0]
                flight_action = OracleActionSpec(
                    "update_reservation_flights",
                    {
                        "reservation_id": reservation.reservation_id,
                        "cabin": reservation.cabin,
                        "flights": flights,
                        "payment_id": flight_payment,
                    },
                )
                changed = next(
                    new
                    for old, new in zip(reservation.flights, flights, strict=True)
                    if (old.flight_number, old.date)
                    != (new["flight_number"], new["date"])
                )
                target = f"{changed['flight_number']} on {changed['date']}"
                add(
                    template="change_one_flight_segment",
                    reservation=reservation,
                    user=user,
                    reason_for_call=f"Change one segment of reservation {reservation.reservation_id} to {target}.",
                    known_info=identity + f" Use saved payment method {flight_payment}.",
                    task_instructions=f"Request {target}, keep the remaining itinerary and cabin unchanged, and confirm after the agent verifies availability and price.",
                    actions=(flight_action,),
                    communicate_info=[reservation.reservation_id, changed["flight_number"], changed["date"]],
                    purpose="Change one eligible flight segment while preserving the itinerary.",
                    difficulty="complex",
                    reads=("get_reservation_details", "search_direct_flight"),
                    entities=(flight_payment,),
                )
                add(
                    template="change_flight_and_correct_passenger",
                    reservation=reservation,
                    user=user,
                    reason_for_call=f"Move one segment of {reservation.reservation_id} to {target} and correct the first passenger's last name to {corrected_name}.",
                    known_info=identity + f" Use saved payment method {flight_payment}.",
                    task_instructions="Request both updates, keep every unspecified itinerary and passenger field unchanged, and confirm the complete plan.",
                    actions=(flight_action, passenger_action),
                    communicate_info=[reservation.reservation_id, changed["flight_number"], corrected_name],
                    purpose="Combine a constrained flight change with a passenger correction.",
                    difficulty="hard",
                    reads=("get_reservation_details", "search_direct_flight"),
                    entities=(flight_payment,),
                )
                if baggage_action is not None and flight_payment in _credit_card_ids(user):
                    baggage_with_same_payment = deepcopy(baggage_action)
                    baggage_with_same_payment.arguments["payment_id"] = flight_payment
                    add(
                        template="change_flight_and_add_baggage",
                        reservation=reservation,
                        user=user,
                        reason_for_call=f"Move one segment of {reservation.reservation_id} to {target} and add one checked bag.",
                        known_info=identity + f" Use saved credit card {flight_payment}.",
                        task_instructions="Request both changes, let the agent calculate all price effects, preserve the rest of the itinerary, and confirm the complete plan.",
                        actions=(flight_action, baggage_with_same_payment),
                        communicate_info=[reservation.reservation_id, changed["flight_number"], f"{new_total} total bags"],
                        purpose="Combine a flight change with a correctly priced baggage update.",
                        difficulty="hard",
                        reads=("get_reservation_details", "get_user_details", "search_direct_flight"),
                        entities=(flight_payment,),
                    )

        upgrade = _cabin_upgrade(db, reservation)
        if upgrade is not None:
            new_cabin, flights, price_delta = upgrade
            upgrade_payments = valid_update_payment_ids(user, max(0, price_delta))
            if upgrade_payments:
                upgrade_payment = upgrade_payments[0]
                upgrade_action = OracleActionSpec(
                    "update_reservation_flights",
                    {
                        "reservation_id": reservation.reservation_id,
                        "cabin": new_cabin,
                        "flights": flights,
                        "payment_id": upgrade_payment,
                    },
                )
                add(
                    template="upgrade_cabin_and_correct_passenger",
                    reservation=reservation,
                    user=user,
                    reason_for_call=f"Upgrade reservation {reservation.reservation_id} to {new_cabin} and correct the first passenger's last name to {corrected_name}.",
                    known_info=identity + f" Use saved payment method {upgrade_payment}.",
                    task_instructions="Request both updates, retain every flight and passenger count, and confirm after availability and price are explained.",
                    actions=(upgrade_action, passenger_action),
                    communicate_info=[reservation.reservation_id, new_cabin, corrected_name],
                    purpose="Combine a whole-itinerary cabin upgrade with a passenger correction.",
                    difficulty="hard",
                    reads=("get_reservation_details",),
                    entities=(upgrade_payment,),
                )

    candidates = [candidate for pool in families.values() for candidate in pool]
    rng.shuffle(candidates)
    return candidates
