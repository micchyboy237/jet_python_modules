"""
Summary: Pydantic-based event payload definitions ensuring type safety, validation,
and immutability (frozen=True). Defines domain events like UserSignupEvent and
OrderCreatedEvent that serve as contracts between producers and consumers.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class UserSignupEvent(BaseModel):
    """Type-safe event payload using Pydantic"""

    user_id: int
    email: str
    signup_date: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True  # Makes the event immutable


class OrderCreatedEvent(BaseModel):
    order_id: str
    total_amount: float
    items: list[str]
