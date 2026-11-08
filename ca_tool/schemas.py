"""Strict, versioned contracts for fictional advisor requests and brand policy."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0.0"
Scenario = Literal[
    "greeting", "product_recommendation", "post_purchase_care", "reactivation",
    "event_invitation", "price_objection",
]
NonEmpty = Annotated[str, Field(min_length=1, max_length=1000)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Client(StrictModel):
    id: Annotated[str, Field(min_length=1, max_length=120)]
    preferred_name: Annotated[str, Field(min_length=1, max_length=80)] | None = None
    preferences: list[NonEmpty] = Field(default_factory=list, max_length=20)
    contact_permission: Literal["allowed", "unknown", "do_not_contact"] = "unknown"
    relationship_level: Annotated[str, Field(min_length=1, max_length=80)] | None = None
    notes: NonEmpty | None = None


class HistoryMessage(StrictModel):
    speaker: Literal["advisor", "client"]
    content: NonEmpty
    timestamp: AwareDatetime | None = None


class Product(StrictModel):
    id: Annotated[str, Field(min_length=1, max_length=120)]
    name: Annotated[str, Field(min_length=1, max_length=120)]
    price: Annotated[Decimal, Field(gt=0, max_digits=12, decimal_places=2)] | None = None
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None = None
    material: NonEmpty | None = None
    color: NonEmpty | None = None
    carry_options: list[NonEmpty] = Field(default_factory=list, max_length=10)
    dimensions: NonEmpty | None = None
    features: list[NonEmpty] = Field(default_factory=list, max_length=20)
    care_instructions: list[NonEmpty] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def paired_price(self) -> Product:
        if (self.price is None) != (self.currency is None):
            raise ValueError("price and currency must be supplied together")
        return self


class GreetingContext(StrictModel):
    occasion_type: Literal["birthday", "festival"] | None = None
    occasion_name: NonEmpty | None = None
    occasion_date: date | None = None


class RecommendationContext(StrictModel):
    target_product_id: NonEmpty | None = None
    usage_occasion: NonEmpty | None = None


class Purchase(StrictModel):
    product_id: NonEmpty | None = None
    purchased_at: AwareDatetime | None = None


class CareContext(StrictModel):
    purchase: Purchase | None = None


class ReactivationContext(StrictModel):
    last_interaction_at: AwareDatetime | None = None
    last_purchase_at: AwareDatetime | None = None


class Event(StrictModel):
    name: NonEmpty | None = None
    starts_at: AwareDatetime | None = None
    ends_at: AwareDatetime | None = None
    location: NonEmpty | None = None
    rsvp_method: NonEmpty | None = None
    description: NonEmpty | None = None


class EventContext(StrictModel):
    event: Event | None = None


class PriceContext(StrictModel):
    target_product_id: NonEmpty | None = None
    objection_type: Literal["too_expensive", "channel_comparison", "discount_request"] | None = None
    objection_text: NonEmpty | None = None


CONTEXT_TYPES = {
    "greeting": GreetingContext,
    "product_recommendation": RecommendationContext,
    "post_purchase_care": CareContext,
    "reactivation": ReactivationContext,
    "event_invitation": EventContext,
    "price_objection": PriceContext,
}


class Request(StrictModel):
    request_id: Annotated[str, Field(min_length=1, max_length=120)]
    brand_id: Annotated[str, Field(min_length=1, max_length=120)]
    scenario: Scenario
    reference_time: AwareDatetime
    client: Client
    conversation_history: list[HistoryMessage] = Field(default_factory=list, max_length=30)
    products: list[Product] = Field(default_factory=list, max_length=10)
    scenario_context: GreetingContext | RecommendationContext | CareContext | ReactivationContext | EventContext | PriceContext

    @model_validator(mode="before")
    @classmethod
    def select_context(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            context_type = CONTEXT_TYPES.get(value.get("scenario"))
            if context_type is not None:
                value["scenario_context"] = context_type.model_validate(value.get("scenario_context", {}))
        return value

    @model_validator(mode="after")
    def request_integrity(self) -> Request:
        ids = [p.id for p in self.products]
        if len(ids) != len(set(ids)):
            raise ValueError("product IDs must be unique")
        if not isinstance(self.scenario_context, CONTEXT_TYPES[self.scenario]):
            raise ValueError("scenario_context does not match scenario")
        times = [m.timestamp for m in self.conversation_history if m.timestamp is not None]
        if any(a > b for a, b in zip(times, times[1:])):
            raise ValueError("conversation_history timestamps must be chronological")
        return self


class Brand(StrictModel):
    brand_id: NonEmpty
    name: NonEmpty
    brand_rules_version: NonEmpty
    category: NonEmpty
    audience: NonEmpty
    voice: list[NonEmpty] = Field(min_length=1, max_length=20)
    max_message_chars: int = Field(default=160, ge=30, le=160)
    target_message_chars: tuple[int, int] = (30, 100)
    extra_discounts_allowed: Literal[False] = False
    allowed_services: list[NonEmpty] = Field(default_factory=lambda: ["预约试背"], max_length=20)
    forbidden_claims: list[NonEmpty] = Field(default_factory=list, max_length=50)
    scenario_rules: dict[Scenario, list[NonEmpty]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def message_lengths(self) -> Brand:
        low, high = self.target_message_chars
        if not 1 <= low <= high <= self.max_message_chars:
            raise ValueError("target_message_chars must fit max_message_chars")
        return self


class Candidate(StrictModel):
    id: Literal["option_1", "option_2", "option_3"]
    text: Annotated[str, Field(min_length=1, max_length=160)]


class CandidateSet(StrictModel):
    candidates: list[Candidate] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def unique_ids(self) -> CandidateSet:
        if {c.id for c in self.candidates} != {"option_1", "option_2", "option_3"}:
            raise ValueError("candidate IDs must be option_1, option_2, option_3 exactly once")
        return self
