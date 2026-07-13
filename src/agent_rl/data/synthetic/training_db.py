"""Build isolated, pseudonymized databases for synthetic RL training."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from tau2.domains.airline.data_model import FlightDB, Name, Passenger
from tau2.domains.airline.utils import AIRLINE_DB_PATH
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.utils import TELECOM_DB_PATH


TRAINING_DB_VERSION = "1.0.0"
SUPPORTED_TRAINING_DOMAINS = ("airline", "retail", "telecom")


@dataclass(frozen=True, slots=True)
class TrainingDatabaseConfig:
    output_root: Path
    seed: int = 43
    telecom_clone_factor: int = 16

    def __post_init__(self) -> None:
        if self.telecom_clone_factor <= 0:
            raise ValueError("telecom_clone_factor must be positive")


def _stable_digits(seed: int, namespace: str, index: int, width: int) -> str:
    digest = hashlib.sha256(
        f"{seed}:{namespace}:{index}".encode("utf-8")
    ).hexdigest()
    value = int(digest[:16], 16) % (10**width)
    return f"{value:0{width}d}"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _synthetic_person(index: int) -> tuple[str, str]:
    first_names = (
        "Avery",
        "Blake",
        "Cameron",
        "Dakota",
        "Emerson",
        "Finley",
        "Harper",
        "Jordan",
        "Morgan",
        "Quinn",
        "Reese",
        "Rowan",
    )
    first = first_names[index % len(first_names)]
    return first, f"Northfield{index:05d}"


def _synthetic_airline_db(source: FlightDB, *, seed: int) -> FlightDB:
    flight_map = {
        old_id: f"SYN{index:05d}"
        for index, old_id in enumerate(sorted(source.flights), start=1)
    }
    user_map = {
        old_id: f"synthetic_air_user_{index:05d}"
        for index, old_id in enumerate(sorted(source.users), start=1)
    }
    reservation_map = {
        old_id: f"SYNRES{index:06d}"
        for index, old_id in enumerate(sorted(source.reservations), start=1)
    }

    flights = {
        flight_map[old_id]: flight.model_copy(
            deep=True,
            update={"flight_number": flight_map[old_id]},
        )
        for old_id, flight in source.flights.items()
    }
    payment_maps: dict[str, dict[str, str]] = {}
    users = {}
    for index, old_id in enumerate(sorted(source.users), start=1):
        user = source.users[old_id]
        payment_map: dict[str, str] = {}
        payment_methods = {}
        for payment_index, (old_payment_id, method) in enumerate(
            sorted(user.payment_methods.items()), start=1
        ):
            new_payment_id = (
                f"{method.source}_synthetic_"
                f"{_stable_digits(seed, old_id, payment_index, 8)}"
            )
            payment_map[old_payment_id] = new_payment_id
            payment_methods[new_payment_id] = method.model_copy(
                deep=True,
                update={"id": new_payment_id},
            )
        payment_maps[old_id] = payment_map
        first, last = _synthetic_person(index)
        saved_passengers = [
            Passenger(
                first_name=_synthetic_person(index * 10 + passenger_index)[0],
                last_name=_synthetic_person(index * 10 + passenger_index)[1],
                dob=f"{1960 + ((index + passenger_index) % 40):04d}-"
                f"{1 + ((index + passenger_index) % 12):02d}-"
                f"{1 + ((index * 3 + passenger_index) % 27):02d}",
            )
            for passenger_index, _ in enumerate(user.saved_passengers, start=1)
        ]
        users[user_map[old_id]] = user.model_copy(
            deep=True,
            update={
                "user_id": user_map[old_id],
                "name": Name(first_name=first, last_name=last),
                "email": f"air-user-{index:05d}@example.invalid",
                "dob": f"{1955 + (index % 45):04d}-{1 + (index % 12):02d}-"
                f"{1 + ((index * 7) % 27):02d}",
                "address": user.address.model_copy(
                    deep=True,
                    update={
                        "address1": f"{100 + index} Synthetic Aviation Way",
                        "address2": f"Unit {1 + (index % 80)}",
                        "city": "Denver",
                        "state": "CO",
                        "zip": f"{80000 + (index % 999):05d}",
                    },
                ),
                "payment_methods": payment_methods,
                "saved_passengers": saved_passengers,
                "reservations": [
                    reservation_map[item]
                    for item in user.reservations
                    if item in reservation_map
                ],
            },
        )

    reservations = {}
    for index, old_id in enumerate(sorted(source.reservations), start=1):
        reservation = source.reservations[old_id]
        payment_map = payment_maps[reservation.user_id]
        passengers = [
            Passenger(
                first_name=_synthetic_person(index * 10 + passenger_index)[0],
                last_name=_synthetic_person(index * 10 + passenger_index)[1],
                dob=f"{1958 + ((index + passenger_index) % 43):04d}-"
                f"{1 + ((index + passenger_index) % 12):02d}-"
                f"{1 + ((index * 5 + passenger_index) % 27):02d}",
            )
            for passenger_index, _ in enumerate(reservation.passengers, start=1)
        ]
        reservations[reservation_map[old_id]] = reservation.model_copy(
            deep=True,
            update={
                "reservation_id": reservation_map[old_id],
                "user_id": user_map[reservation.user_id],
                "flights": [
                    segment.model_copy(
                        deep=True,
                        update={"flight_number": flight_map[segment.flight_number]},
                    )
                    for segment in reservation.flights
                ],
                "passengers": passengers,
                "payment_history": [
                    payment.model_copy(
                        deep=True,
                        update={
                            "payment_id": payment_map.get(
                                payment.payment_id,
                                "certificate_synthetic_"
                                + _stable_digits(seed, old_id, payment_index, 8),
                            )
                        },
                    )
                    for payment_index, payment in enumerate(
                        reservation.payment_history, start=1
                    )
                ],
            },
        )

    database = FlightDB(flights=flights, users=users, reservations=reservations)
    _validate_airline_links(database)
    return database


def _retail_address(address: Any, index: int) -> Any:
    return address.model_copy(
        deep=True,
        update={
            "address1": f"{200 + index} Synthetic Commerce Street",
            "address2": f"Suite {1 + (index % 90)}",
            "city": "Austin",
            "state": "TX",
            "zip": f"{73301 + (index % 899):05d}",
        },
    )


def _synthetic_retail_db(source: RetailDB, *, seed: int) -> RetailDB:
    product_map = {
        old_id: f"90{index:010d}"
        for index, old_id in enumerate(sorted(source.products), start=1)
    }
    item_map: dict[str, str] = {}
    item_index = 0
    for product in source.products.values():
        for old_item_id in product.variants:
            item_index += 1
            item_map[old_item_id] = f"91{item_index:010d}"
    user_map = {
        old_id: f"synthetic_retail_user_{index:05d}"
        for index, old_id in enumerate(sorted(source.users), start=1)
    }
    order_map = {
        old_id: f"SYN-ORDER-{index:06d}"
        for index, old_id in enumerate(sorted(source.orders), start=1)
    }

    products = {}
    for index, old_id in enumerate(sorted(source.products), start=1):
        product = source.products[old_id]
        new_product_id = product_map[old_id]
        variants = {
            item_map[old_item_id]: variant.model_copy(
                deep=True,
                update={"item_id": item_map[old_item_id]},
            )
            for old_item_id, variant in product.variants.items()
        }
        products[new_product_id] = product.model_copy(
            deep=True,
            update={
                "name": f"Training Product {index:05d}",
                "product_id": new_product_id,
                "variants": variants,
            },
        )

    payment_maps: dict[str, dict[str, str]] = {}
    users = {}
    for index, old_id in enumerate(sorted(source.users), start=1):
        user = source.users[old_id]
        payment_map: dict[str, str] = {}
        payment_methods = {}
        for payment_index, (old_payment_id, method) in enumerate(
            sorted(user.payment_methods.items()), start=1
        ):
            new_payment_id = (
                f"{method.source}_synthetic_"
                f"{_stable_digits(seed, old_id, payment_index, 8)}"
            )
            payment_map[old_payment_id] = new_payment_id
            payment_methods[new_payment_id] = method.model_copy(
                deep=True,
                update={"id": new_payment_id},
            )
        payment_maps[old_id] = payment_map
        first, last = _synthetic_person(index + 20_000)
        users[user_map[old_id]] = user.model_copy(
            deep=True,
            update={
                "user_id": user_map[old_id],
                "name": user.name.model_copy(
                    deep=True,
                    update={"first_name": first, "last_name": last},
                ),
                "email": f"retail-user-{index:05d}@example.invalid",
                "address": _retail_address(user.address, index),
                "payment_methods": payment_methods,
                "orders": [order_map[item] for item in user.orders if item in order_map],
            },
        )

    orders = {}
    for index, old_id in enumerate(sorted(source.orders), start=1):
        order = source.orders[old_id]
        payment_map = payment_maps[order.user_id]
        items = [
            item.model_copy(
                deep=True,
                update={
                    "name": products[product_map[item.product_id]].name,
                    "product_id": product_map[item.product_id],
                    "item_id": item_map[item.item_id],
                },
            )
            for item in order.items
        ]
        fulfillments = [
            fulfillment.model_copy(
                deep=True,
                update={
                    "tracking_id": [
                        f"SYNTRACK{index:06d}{tracking_index:02d}"
                        for tracking_index, _ in enumerate(
                            fulfillment.tracking_id, start=1
                        )
                    ],
                    "item_ids": [item_map[item] for item in fulfillment.item_ids],
                },
            )
            for fulfillment in order.fulfillments
        ]
        orders[order_map[old_id]] = order.model_copy(
            deep=True,
            update={
                "order_id": order_map[old_id],
                "user_id": user_map[order.user_id],
                "address": _retail_address(order.address, 10_000 + index),
                "items": items,
                "fulfillments": fulfillments,
                "payment_history": [
                    payment.model_copy(
                        deep=True,
                        update={
                            "payment_method_id": payment_map[
                                payment.payment_method_id
                            ]
                        },
                    )
                    for payment in order.payment_history
                ],
                "exchange_items": (
                    [item_map[item] for item in order.exchange_items]
                    if order.exchange_items is not None
                    else None
                ),
                "exchange_new_items": (
                    [item_map[item] for item in order.exchange_new_items]
                    if order.exchange_new_items is not None
                    else None
                ),
                "exchange_payment_method_id": (
                    payment_map[order.exchange_payment_method_id]
                    if order.exchange_payment_method_id is not None
                    else None
                ),
                "return_items": (
                    [item_map[item] for item in order.return_items]
                    if order.return_items is not None
                    else None
                ),
                "return_payment_method_id": (
                    payment_map[order.return_payment_method_id]
                    if order.return_payment_method_id is not None
                    else None
                ),
            },
        )

    database = RetailDB(products=products, users=users, orders=orders)
    _validate_retail_links(database)
    return database


def _replace_known_ids(text: str, mapping: dict[str, str]) -> str:
    output = text
    for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        output = output.replace(old, new)
    return output


def _synthetic_telecom_db(
    source: TelecomDB,
    *,
    seed: int,
    clone_factor: int,
) -> TelecomDB:
    plan_map = {
        plan.plan_id: f"SYN-PLAN-{index:03d}"
        for index, plan in enumerate(source.plans, start=1)
    }
    plans = [
        plan.model_copy(
            deep=True,
            update={
                "plan_id": plan_map[plan.plan_id],
                "name": f"Training Plan {index:03d}",
            },
        )
        for index, plan in enumerate(source.plans, start=1)
    ]
    source_lines = {line.line_id: line for line in source.lines}
    source_bills = {bill.bill_id: bill for bill in source.bills}
    source_devices = {device.device_id: device for device in source.devices}
    customers = []
    lines = []
    bills = []
    devices = []
    customer_index = 0
    line_index = 0
    bill_index = 0
    device_index = 0

    for replica in range(1, clone_factor + 1):
        for source_customer in sorted(
            source.customers, key=lambda item: item.customer_id
        ):
            customer_index += 1
            customer_id = f"SYN-CUSTOMER-{customer_index:05d}"
            line_map: dict[str, str] = {}
            phone_map: dict[str, str] = {}
            device_map: dict[str, str] = {}
            for source_line_id in source_customer.line_ids:
                source_line = source_lines[source_line_id]
                line_index += 1
                line_map[source_line_id] = f"SYN-LINE-{line_index:06d}"
                phone_map[source_line.phone_number] = f"+1555{line_index:07d}"
                if source_line.device_id is not None:
                    device_index += 1
                    device_map[source_line.device_id] = (
                        f"SYN-DEVICE-{device_index:06d}"
                    )
            bill_map: dict[str, str] = {}
            for source_bill_id in source_customer.bill_ids:
                bill_index += 1
                bill_map[source_bill_id] = f"SYN-BILL-{bill_index:06d}"

            for source_device_id, new_device_id in device_map.items():
                source_device = source_devices[source_device_id]
                devices.append(
                    source_device.model_copy(
                        deep=True,
                        update={
                            "device_id": new_device_id,
                            "imei": (
                                "99"
                                + _stable_digits(
                                    seed + replica,
                                    source_device_id,
                                    device_index,
                                    13,
                                )
                                if source_device.imei is not None
                                else None
                            ),
                        },
                    )
                )

            for source_line_id, new_line_id in line_map.items():
                source_line = source_lines[source_line_id]
                lines.append(
                    source_line.model_copy(
                        deep=True,
                        update={
                            "line_id": new_line_id,
                            "phone_number": phone_map[source_line.phone_number],
                            "plan_id": plan_map[source_line.plan_id],
                            "device_id": (
                                device_map[source_line.device_id]
                                if source_line.device_id is not None
                                else None
                            ),
                        },
                    )
                )

            replacement_map = {**line_map, **phone_map, **bill_map}
            for source_bill_id, new_bill_id in bill_map.items():
                source_bill = source_bills[source_bill_id]
                bills.append(
                    source_bill.model_copy(
                        deep=True,
                        update={
                            "bill_id": new_bill_id,
                            "customer_id": customer_id,
                            "line_items": [
                                item.model_copy(
                                    deep=True,
                                    update={
                                        "description": _replace_known_ids(
                                            item.description, replacement_map
                                        )
                                    },
                                )
                                for item in source_bill.line_items
                            ],
                        },
                    )
                )

            first, last = _synthetic_person(customer_index + 40_000)
            primary_phone = (
                phone_map[source_lines[source_customer.line_ids[0]].phone_number]
                if source_customer.line_ids
                else f"+1555{customer_index:07d}"
            )
            customers.append(
                source_customer.model_copy(
                    deep=True,
                    update={
                        "customer_id": customer_id,
                        "full_name": f"{first} {last}",
                        "date_of_birth": (
                            f"{1955 + (customer_index % 45):04d}-"
                            f"{1 + (customer_index % 12):02d}-"
                            f"{1 + ((customer_index * 11) % 27):02d}"
                        ),
                        "email": (
                            f"telecom-user-{customer_index:05d}@example.invalid"
                        ),
                        "phone_number": primary_phone,
                        "address": source_customer.address.model_copy(
                            deep=True,
                            update={
                                "street": (
                                    f"{300 + customer_index} Synthetic Network Lane"
                                ),
                                "city": "Seattle",
                                "state": "WA",
                                "zip_code": f"{98100 + (customer_index % 899):05d}",
                            },
                        ),
                        "payment_methods": [
                            method.model_copy(
                                deep=True,
                                update={
                                    "account_number_last_4": _stable_digits(
                                        seed + replica,
                                        customer_id,
                                        payment_index,
                                        4,
                                    )
                                },
                            )
                            for payment_index, method in enumerate(
                                source_customer.payment_methods, start=1
                            )
                        ],
                        "line_ids": [line_map[item] for item in source_customer.line_ids],
                        "bill_ids": [bill_map[item] for item in source_customer.bill_ids],
                    },
                )
            )

    database = TelecomDB(
        plans=plans,
        customers=customers,
        lines=lines,
        bills=bills,
        devices=devices,
    )
    _validate_telecom_links(database)
    return database


def _validate_airline_links(db: FlightDB) -> None:
    for key, flight in db.flights.items():
        if key != flight.flight_number:
            raise ValueError("airline flight key mismatch")
    for key, user in db.users.items():
        if key != user.user_id or not set(user.reservations) <= set(db.reservations):
            raise ValueError("airline user relationship is invalid")
    for key, reservation in db.reservations.items():
        if key != reservation.reservation_id or reservation.user_id not in db.users:
            raise ValueError("airline reservation relationship is invalid")
        if any(item.flight_number not in db.flights for item in reservation.flights):
            raise ValueError("airline reservation references an unknown flight")


def _validate_retail_links(db: RetailDB) -> None:
    item_ids = {
        item_id
        for product in db.products.values()
        for item_id in product.variants
    }
    for key, product in db.products.items():
        if key != product.product_id:
            raise ValueError("retail product key mismatch")
    for key, user in db.users.items():
        if key != user.user_id or not set(user.orders) <= set(db.orders):
            raise ValueError("retail user relationship is invalid")
    for key, order in db.orders.items():
        if key != order.order_id or order.user_id not in db.users:
            raise ValueError("retail order relationship is invalid")
        if any(
            item.product_id not in db.products or item.item_id not in item_ids
            for item in order.items
        ):
            raise ValueError("retail order references an unknown catalog item")
        payment_ids = set(db.users[order.user_id].payment_methods)
        if any(
            payment.payment_method_id not in payment_ids
            for payment in order.payment_history
        ):
            raise ValueError("retail order references an unknown payment method")


def _validate_telecom_links(db: TelecomDB) -> None:
    plans = {item.plan_id for item in db.plans}
    customers = {item.customer_id: item for item in db.customers}
    lines = {item.line_id: item for item in db.lines}
    bills = {item.bill_id: item for item in db.bills}
    devices = {item.device_id for item in db.devices}
    if len(customers) != len(db.customers) or len(lines) != len(db.lines):
        raise ValueError("telecom synthetic identifiers must be unique")
    for customer in customers.values():
        if not set(customer.line_ids) <= set(lines):
            raise ValueError("telecom customer references an unknown line")
        if not set(customer.bill_ids) <= set(bills):
            raise ValueError("telecom customer references an unknown bill")
        if any(bills[item].customer_id != customer.customer_id for item in customer.bill_ids):
            raise ValueError("telecom bill belongs to the wrong customer")
    for line in lines.values():
        if line.plan_id not in plans:
            raise ValueError("telecom line references an unknown plan")
        if line.device_id is not None and line.device_id not in devices:
            raise ValueError("telecom line references an unknown device")


def database_identifiers(domain: str, db: Any) -> set[str]:
    """Collect identifiers whose reuse would couple training to the source DB."""

    if domain == "airline":
        return {
            *db.flights.keys(),
            *db.users.keys(),
            *db.reservations.keys(),
            *(user.email for user in db.users.values()),
            *(item for user in db.users.values() for item in user.payment_methods),
        }
    if domain == "retail":
        return {
            *db.products.keys(),
            *db.users.keys(),
            *db.orders.keys(),
            *(user.email for user in db.users.values()),
            *(
                item
                for product in db.products.values()
                for item in product.variants
            ),
            *(item for user in db.users.values() for item in user.payment_methods),
            *(
                tracking_id
                for order in db.orders.values()
                for fulfillment in order.fulfillments
                for tracking_id in fulfillment.tracking_id
            ),
        }
    if domain == "telecom":
        return {
            *(item.plan_id for item in db.plans),
            *(item.customer_id for item in db.customers),
            *(item.line_id for item in db.lines),
            *(item.phone_number for item in db.lines),
            *(item.phone_number for item in db.customers),
            *(item.email for item in db.customers),
            *(item.bill_id for item in db.bills),
            *(item.device_id for item in db.devices),
            *(item.imei for item in db.devices if item.imei is not None),
        }
    raise ValueError(f"unsupported training database domain: {domain}")


def _assert_disjoint(domain: str, source: Any, synthetic: Any) -> None:
    overlap = database_identifiers(domain, source) & database_identifiers(
        domain, synthetic
    )
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(f"{domain} training DB retains official identifiers: {preview}")


def _source_databases() -> dict[str, Any]:
    return {
        "airline": FlightDB.load(AIRLINE_DB_PATH),
        "retail": RetailDB.load(RETAIL_DB_PATH),
        "telecom": TelecomDB.load(TELECOM_DB_PATH),
    }


def build_training_databases(config: TrainingDatabaseConfig) -> dict[str, Any]:
    sources = _source_databases()
    databases = {
        "airline": _synthetic_airline_db(sources["airline"], seed=config.seed),
        "retail": _synthetic_retail_db(sources["retail"], seed=config.seed),
        "telecom": _synthetic_telecom_db(
            sources["telecom"],
            seed=config.seed,
            clone_factor=config.telecom_clone_factor,
        ),
    }
    manifest_domains: dict[str, Any] = {}
    for domain, database in databases.items():
        _assert_disjoint(domain, sources[domain], database)
        path = config.output_root / domain / "db.json"
        _atomic_json(path, database.model_dump(mode="json"))
        manifest_domains[domain] = {
            "path": f"{domain}/db.json",
            "source_hash": sources[domain].get_hash(),
            "training_hash": database.get_hash(),
            "source_statistics": sources[domain].get_statistics(),
            "training_statistics": database.get_statistics(),
            "official_identifier_overlap": 0,
        }
    manifest = {
        "version": TRAINING_DB_VERSION,
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
        },
        "domains": manifest_domains,
    }
    _atomic_json(config.output_root / "manifest.json", manifest)
    _load_training_database_cached.cache_clear()
    return manifest


def training_database_fingerprint(manifest: dict[str, Any]) -> str:
    """Hash only portable identity fields, never host-specific output paths."""

    portable = {
        "version": manifest.get("version"),
        "seed": (manifest.get("config") or {}).get("seed"),
        "telecom_clone_factor": (manifest.get("config") or {}).get(
            "telecom_clone_factor"
        ),
        "domains": {
            domain: {
                "source_hash": values.get("source_hash"),
                "training_hash": values.get("training_hash"),
            }
            for domain, values in sorted((manifest.get("domains") or {}).items())
        },
    }
    payload = json.dumps(portable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_training_databases(config: TrainingDatabaseConfig) -> dict[str, Any]:
    manifest_path = config.output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_config = {
        "seed": config.seed,
        "telecom_clone_factor": config.telecom_clone_factor,
    }
    actual_config = manifest.get("config") or {}
    mismatches = {
        key: {"expected": value, "actual": actual_config.get(key)}
        for key, value in expected_config.items()
        if actual_config.get(key) != value
    }
    if manifest.get("version") != TRAINING_DB_VERSION:
        mismatches["version"] = {
            "expected": TRAINING_DB_VERSION,
            "actual": manifest.get("version"),
        }
    if mismatches:
        raise ValueError(f"training DB manifest mismatch: {mismatches}")

    sources = _source_databases()
    for domain in SUPPORTED_TRAINING_DOMAINS:
        database = load_training_database(config.output_root, domain)
        expected = (manifest.get("domains") or {}).get(domain) or {}
        if expected.get("source_hash") != sources[domain].get_hash():
            raise ValueError(f"{domain} official DB changed; rebuild training DB")
        if expected.get("training_hash") != database.get_hash():
            raise ValueError(f"{domain} training DB hash mismatch")
        _assert_disjoint(domain, sources[domain], database)
    return manifest


@lru_cache(maxsize=12)
def _load_training_database_cached(root: str, domain: str) -> Any:
    path = Path(root) / domain / "db.json"
    if domain == "airline":
        return FlightDB.load(path)
    if domain == "retail":
        return RetailDB.load(path)
    if domain == "telecom":
        return TelecomDB.load(path)
    raise ValueError(f"unsupported training database domain: {domain}")


def load_training_database(root: str | Path, domain: str) -> Any:
    """Return a deep copy so one episode can never mutate another."""

    resolved = str(Path(root).resolve())
    return deepcopy(_load_training_database_cached(resolved, domain))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--telecom-clone-factor", type=int, default=16)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = TrainingDatabaseConfig(
        output_root=Path(args.output_root),
        seed=args.seed,
        telecom_clone_factor=args.telecom_clone_factor,
    )
    manifest = (
        validate_training_databases(config)
        if args.validate_only
        else build_training_databases(config)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
