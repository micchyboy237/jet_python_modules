"""
Summary: Practical e-commerce demo showing decorator-based handler registration
and Pydantic event emission. Demonstrates concurrent processing of user signup
(email + analytics) and order creation (warehouse notification) with proper
payload serialization from frozen Pydantic models.
"""

import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from patterns.async_event_bus import Event, EventBus
from use_cases.events import UserSignupEvent

# Initialize the bus
bus = EventBus()

# --- Handlers using Decorators ---


@bus.on("user.signup")
async def send_welcome_email(event: Event):
    print(f"[Email Service] Sending welcome email to {event.payload['email']}")
    await asyncio.sleep(0.5)


@bus.on("user.signup")
def update_analytics(event: Event):
    print(f"[Analytics Service] Tracking user signup: {event.payload['user_id']}")


@bus.on("order.created")
async def notify_warehouse(event: Event):
    print(
        f"[Warehouse Service] Preparing shipment for order {event.payload['order_id']}"
    )


# --- Main Logic ---
async def main():
    print("--- Triggering User Signup ---")
    signup_data = UserSignupEvent(user_id=101, email="jet@example.com")

    signup_event = Event(event_type="user.signup", payload=signup_data.model_dump())
    await bus.publish(signup_event)

    print("\n--- Triggering Order Creation ---")
    order_event = Event(
        event_type="order.created",
        payload={"order_id": "ORD-999", "total": 150.00, "items": ["Laptop"]},
    )
    await bus.publish(order_event)


if __name__ == "__main__":
    asyncio.run(main())
