"""Policy calculations and validation for clean-room airline task generation."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from tau2.domains.airline.data_model import (
    FlightDB,
    FlightDateStatusAvailable,
    Reservation,
    User,
)
from tau2.domains.airline.utils import AIRLINE_DB_PATH

if TYPE_CHECKING:
    from agent_rl.data.synthetic.generators.common import GeneratedCandidate


CURRENT_TIME = datetime.fromisoformat("2024-05-15T15:00:00")
CancellationReason = Literal["change_of_plan", "health", "weather", "airline_cancelled"]

FREE_BAGS_PER_PASSENGER = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 4},
}


@lru_cache(maxsize=1)
def load_airline_policy_db() -> FlightDB:
    """Load one read-only DB snapshot for pure policy validation."""

    return FlightDB.load(AIRLINE_DB_PATH)


def reservation_statuses(db: FlightDB, reservation: Reservation) -> tuple[str, ...]:
    return tuple(
        db.flights[segment.flight_number].dates[segment.date].status
        for segment in reservation.flights
    )


def has_flown_segment(db: FlightDB, reservation: Reservation) -> bool:
    return any(
        status in {"flying", "landed"}
        for status in reservation_statuses(db, reservation)
    )


def cancellation_allowed(
    db: FlightDB,
    reservation: Reservation,
    reason: CancellationReason,
) -> bool:
    if reservation.status == "cancelled" or has_flown_segment(db, reservation):
        return False

    statuses = reservation_statuses(db, reservation)
    within_24_hours = (
        CURRENT_TIME - datetime.fromisoformat(reservation.created_at)
    ).total_seconds() <= 24 * 60 * 60
    covered_by_insurance = (
        reservation.insurance == "yes" and reason in {"health", "weather"}
    )
    return bool(
        within_24_hours
        or "cancelled" in statuses
        or reservation.cabin == "business"
        or covered_by_insurance
    )


def free_baggage_allowance(user: User, reservation: Reservation) -> int:
    return (
        FREE_BAGS_PER_PASSENGER[user.membership][reservation.cabin]
        * len(reservation.passengers)
    )


def expected_nonfree_baggages(
    user: User,
    reservation: Reservation,
    total_baggages: int,
) -> int:
    if total_baggages < 0:
        raise ValueError("total_baggages must be non-negative")
    return max(0, total_baggages - free_baggage_allowance(user, reservation))


def valid_update_payment_ids(user: User, amount: int = 0) -> tuple[str, ...]:
    valid: list[str] = []
    for payment_id, method in sorted(user.payment_methods.items()):
        if method.source == "credit_card":
            valid.append(payment_id)
        elif method.source == "gift_card" and method.amount >= amount:
            valid.append(payment_id)
    return tuple(valid)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_payment(user: User, payment_id: str, amount: int) -> None:
    _require(
        payment_id in valid_update_payment_ids(user, max(0, amount)),
        f"invalid update payment {payment_id!r} for amount {amount}",
    )


def _validate_baggage_action(
    db: FlightDB,
    arguments: dict[str, Any],
) -> None:
    reservation = db.reservations[arguments["reservation_id"]]
    user = db.users[reservation.user_id]
    new_total = int(arguments["total_baggages"])
    new_nonfree = int(arguments["nonfree_baggages"])
    _require(reservation.status != "cancelled", "cannot update a cancelled reservation")
    _require(new_total > reservation.total_baggages, "baggage tasks must only add bags")
    expected = expected_nonfree_baggages(user, reservation, new_total)
    _require(
        new_nonfree == expected,
        f"nonfree_baggages must be {expected}, got {new_nonfree}",
    )
    charge = 50 * max(0, new_nonfree - reservation.nonfree_baggages)
    _validate_payment(user, arguments["payment_id"], charge)


def _validate_passenger_action(
    db: FlightDB,
    arguments: dict[str, Any],
) -> None:
    reservation = db.reservations[arguments["reservation_id"]]
    passengers = list(arguments["passengers"])
    _require(reservation.status != "cancelled", "cannot update a cancelled reservation")
    _require(
        len(passengers) == len(reservation.passengers),
        "passenger count cannot change",
    )
    before = [passenger.model_dump(mode="json") for passenger in reservation.passengers]
    _require(passengers != before, "passenger update must change at least one field")


def _validate_flight_action(
    db: FlightDB,
    arguments: dict[str, Any],
) -> None:
    reservation = db.reservations[arguments["reservation_id"]]
    user = db.users[reservation.user_id]
    cabin = arguments["cabin"]
    flights = list(arguments["flights"])
    _require(reservation.status != "cancelled", "cannot update a cancelled reservation")
    _require(
        len(flights) == len(reservation.flights),
        "generated flight changes must preserve itinerary segment count",
    )
    cabin_change = cabin != reservation.cabin
    if cabin_change:
        _require(
            not has_flown_segment(db, reservation),
            "cabin cannot change after any segment has flown",
        )
    else:
        _require(
            reservation.cabin != "basic_economy",
            "basic economy flights cannot be modified",
        )

    changed = cabin_change
    new_total = 0
    for old, new in zip(reservation.flights, flights, strict=True):
        flight = db.flights[new["flight_number"]]
        _require(
            (flight.origin, flight.destination) == (old.origin, old.destination),
            "flight change must preserve every segment's endpoints",
        )
        same_segment = (
            new["flight_number"] == old.flight_number and new["date"] == old.date
        )
        if same_segment and not cabin_change:
            price = old.price
        else:
            state = flight.dates[new["date"]]
            _require(
                isinstance(state, FlightDateStatusAvailable),
                "new flight segment must be available",
            )
            _require(
                state.available_seats[cabin] >= len(reservation.passengers),
                "new flight segment lacks enough seats",
            )
            price = state.prices[cabin]
            changed = True
        new_total += price * len(reservation.passengers)

    _require(changed, "flight update must change a segment or cabin")
    old_total = sum(item.price for item in reservation.flights) * len(
        reservation.passengers
    )
    _validate_payment(user, arguments["payment_id"], new_total - old_total)


def validate_airline_candidate(
    candidate: GeneratedCandidate,
    database: FlightDB | None = None,
) -> None:
    """Independently recompute all policy-sensitive airline action arguments."""

    policy = candidate.metadata.get("policy")
    _require(isinstance(policy, dict), "airline candidate requires policy metadata")
    _require(policy.get("required_confirmation") is True, "writes require confirmation")
    required_reads = policy.get("required_reads")
    _require(isinstance(required_reads, list) and required_reads, "required_reads missing")
    _require(
        policy.get("difficulty") in {"simple", "medium", "complex", "hard"},
        "invalid task difficulty",
    )

    criteria = candidate.task.evaluation_criteria
    actions = list(criteria.actions or []) if criteria else []
    names = [action.name for action in actions]
    _require(names == policy.get("expected_writes"), "expected_writes mismatch")

    db = database if database is not None else load_airline_policy_db()
    for action in actions:
        arguments = action.arguments
        if action.name == "cancel_reservation":
            reservation = db.reservations[arguments["reservation_id"]]
            reason = policy.get("cancellation_reason")
            _require(
                reason in {"change_of_plan", "health", "weather", "airline_cancelled"},
                "valid cancellation_reason required",
            )
            _require(
                cancellation_allowed(db, reservation, reason),
                "reservation is not eligible for cancellation",
            )
        elif action.name == "update_reservation_baggages":
            _validate_baggage_action(db, arguments)
        elif action.name == "update_reservation_passengers":
            _validate_passenger_action(db, arguments)
        elif action.name == "update_reservation_flights":
            _validate_flight_action(db, arguments)
        elif action.name == "send_certificate":
            _require(policy.get("compensation_requested") is True, "compensation must be requested")
            reservation_id = policy.get("reservation_id")
            reservation = db.reservations[reservation_id]
            user = db.users[reservation.user_id]
            _require(arguments["user_id"] == user.user_id, "certificate user mismatch")
            eligible = (
                user.membership in {"silver", "gold"}
                or reservation.insurance == "yes"
                or reservation.cabin == "business"
            )
            _require(eligible, "user is not eligible for compensation")
            expected_amount = 100 * len(reservation.passengers)
            _require(
                arguments["amount"] == expected_amount,
                f"cancelled-flight certificate must be {expected_amount}",
            )
            _require(
                "cancelled" in reservation_statuses(db, reservation),
                "cancelled-flight compensation requires a cancelled segment",
            )
        else:
            raise ValueError(f"unsupported airline Oracle action: {action.name}")
