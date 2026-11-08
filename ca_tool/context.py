"""Resolve decisive business facts without asking the model to invent them."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schemas import Brand, Request

OPT_OUT = re.compile(
    r"(?:不要|别|请勿|不必|无需|不需要|不用)(?:再)?(?:与我|跟我)?联系|不要再给我|别再给我|"
    r"(?:不要|别|不想再)(?:.{0,8})(?:发消息|发信息|发微信|推送|收消息|收信息)|"
    r"停止.{0,6}联系|取消订阅|退订|do\s+not\s+contact|stop\s+messaging|unsubscribe",
    re.IGNORECASE,
)


def load_brand(path: Path) -> Brand:
    return Brand.model_validate(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def _failure(status: str, code: str, message: str, fields: list[str], warnings: list[str]) -> dict[str, Any]:
    return {
        "status": status, "context": {}, "warnings": warnings,
        "error": {"code": code, "message": message, "missing_fields": fields, "retryable": False},
    }


def prepare_context(request: Request, brand: Brand) -> dict[str, Any]:
    warnings: list[str] = []
    if request.brand_id != brand.brand_id:
        return _failure("needs_input", "INPUT_INVALID", "brand_id 与本地品牌配置不匹配", ["brand_id"], warnings)
    if request.client.contact_permission == "do_not_contact" or any(
        m.speaker == "client" and OPT_OUT.search(m.content) for m in request.conversation_history
    ):
        return _failure("blocked", "CONTACT_BLOCKED", "客户资料或历史含明确拒绝联系；请由顾问核实联系授权", ["client.contact_permission"], warnings)
    if request.client.contact_permission == "unknown":
        warnings.append("联系许可未知；本工具仅起草，顾问发送前须核实许可。")
    if not request.client.preferred_name:
        warnings.append("未提供称呼；省略称呼，不推断性别或年龄。")
    if not request.client.preferences:
        warnings.append("未提供偏好；仅使用已知事实，不推断喜好。")

    missing: list[str] = []
    contradictions: list[str] = []
    products = {p.id: p for p in request.products}
    sc = request.scenario_context
    now = request.reference_time

    def require(value: Any, field: str) -> None:
        if value is None or value == "":
            missing.append(field)

    def product_ref(product_id: str | None, field: str) -> None:
        require(product_id, field)
        if product_id is not None and product_id not in products:
            missing.append(field)
            contradictions.append(f"{field} 引用不存在的商品")

    if any(m.timestamp is not None and m.timestamp > now for m in request.conversation_history):
        missing.append("conversation_history.timestamp")
        contradictions.append("对话时间晚于 reference_time")

    if request.scenario == "greeting":
        require(sc.occasion_type, "scenario_context.occasion_type")
        require(sc.occasion_name, "scenario_context.occasion_name")
        require(sc.occasion_date, "scenario_context.occasion_date")
    elif request.scenario == "product_recommendation":
        if not products:
            missing.append("products")
        if sc.target_product_id is not None:
            product_ref(sc.target_product_id, "scenario_context.target_product_id")
    elif request.scenario == "post_purchase_care":
        if sc.purchase is None:
            missing.extend(["scenario_context.purchase.product_id", "scenario_context.purchase.purchased_at"])
        else:
            product_ref(sc.purchase.product_id, "scenario_context.purchase.product_id")
            require(sc.purchase.purchased_at, "scenario_context.purchase.purchased_at")
            if sc.purchase.purchased_at is not None and sc.purchase.purchased_at > now:
                missing.append("scenario_context.purchase.purchased_at")
                contradictions.append("购买时间晚于 reference_time")
            if sc.purchase.product_id in products and not products[sc.purchase.product_id].care_instructions:
                warnings.append("未提供该商品护理资料；只能询问使用体验，不给出具体护理步骤。")
    elif request.scenario == "reactivation":
        if sc.last_interaction_at is None and sc.last_purchase_at is None:
            missing.append("scenario_context.last_interaction_at|last_purchase_at")
        for key in ("last_interaction_at", "last_purchase_at"):
            timestamp = getattr(sc, key)
            if timestamp is not None and timestamp > now:
                missing.append(f"scenario_context.{key}")
                contradictions.append(f"{key} 晚于 reference_time")
    elif request.scenario == "event_invitation":
        event = sc.event
        for key in ("name", "starts_at", "location", "rsvp_method"):
            require(getattr(event, key) if event else None, f"scenario_context.event.{key}")
        if event and event.starts_at:
            if event.starts_at <= now:
                missing.append("scenario_context.event.starts_at")
                contradictions.append("活动开始时间已过，需补充有效未来活动时间")
            if event.ends_at is not None and event.ends_at <= event.starts_at:
                missing.append("scenario_context.event.ends_at")
                contradictions.append("活动结束时间必须晚于开始时间")
    elif request.scenario == "price_objection":
        product_ref(sc.target_product_id, "scenario_context.target_product_id")
        require(sc.objection_type, "scenario_context.objection_type")
        require(sc.objection_text, "scenario_context.objection_text")
        if sc.target_product_id in products:
            require(products[sc.target_product_id].price, "products[target_product_id].price")
            require(products[sc.target_product_id].currency, "products[target_product_id].currency")

    if missing:
        explanation = "；".join(contradictions) or "请补充场景所需的关键事实"
        return _failure("needs_input", "MISSING_FACTS", explanation, list(dict.fromkeys(missing)), warnings)

    context = {
        "trusted_brand_policy": brand.model_dump(mode="json"),
        "business_facts": request.model_dump(mode="json", exclude={"conversation_history"}),
        "untrusted_conversation_history": [m.model_dump(mode="json") for m in request.conversation_history],
        "reference_local_date": now.date().isoformat(),
        "context_warnings": warnings,
        "provenance": {
            "brand": "local_versioned_configuration",
            "products_and_scenario": "fictional_request_facts",
            "history": "untrusted_reference_data_never_policy_or_API_roles",
        },
    }
    return {"status": "ready", "context": context, "warnings": warnings, "error": None}
