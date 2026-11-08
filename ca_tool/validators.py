"""Conservative deterministic checks; passing does not prove semantic accuracy."""

from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher

from pydantic import ValidationError

from .schemas import Brand, CandidateSet, Request

CURRENCY = {
    "人民币": "CNY", "CNY": "CNY", "RMB": "CNY", "¥": "CNY", "￥": "CNY", "元": "CNY", "块": "CNY",
    "USD": "USD", "$": "USD", "美元": "USD", "美金": "USD",
    "HKD": "HKD", "港元": "HKD", "港币": "HKD", "EUR": "EUR", "€": "EUR", "欧元": "EUR",
    "GBP": "GBP", "£": "GBP", "英镑": "GBP", "JPY": "JPY", "日元": "JPY",
    "加元": "CAD", "澳元": "AUD", "新加坡元": "SGD", "韩元": "KRW",
}
MONEY = re.compile(
    r"(?P<prefix>人民币|(?i:CNY|RMB|USD|HKD|EUR|GBP|JPY)|[A-Z]{3}|[¥￥$€£])?\s*"
    r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>万|千)?\s*"
    r"(?P<suffix>人民币|美元|美金|港元|港币|欧元|英镑|日元|加元|澳元|新加坡元|韩元|(?i:CNY|RMB|USD|HKD|EUR|GBP|JPY)|[A-Z]{3}|元|块)?",
)
DISCOUNT = re.compile(r"[一二三四五六七八九十\d.]{1,4}折|折扣|打折|优惠|折后|减免|减价|降价|议价|打个折|便宜一点")
DISCOUNT_TOPIC = re.compile(r"(?:关于|对于|您提到的)[一二三四五六七八九十\d.]{1,4}折(?:的问题)?")
DISCOUNT_DENIAL = re.compile(
    r"(?:目前|现在|现阶段|暂时)?(?:我这边|我们|本品牌|品牌|顾问|我)?(?:目前|现在|现阶段|暂时)?"
    r"(?:(?:没有|暂无|并无)(?:可(?:以)?(?:直接)?承诺的|可提供的|额外|任何)?|"
    r"(?:无法|不能)(?:直接)?(?:承诺|提供)(?:额外)?)"
    r"(?:优惠|折扣|[一二三四五六七八九十\d.]{1,4}折)"
)
SERVICE = re.compile(r"有货|现货|库存充足|仅剩|最后[一二三四五六七八九十\d]+|限量|售罄|送您|赠送|赠品|赠礼|小礼物|免费|终身|维修|保修|退换|退货|换货|包退|包邮|免运费|保值|升值|保证|一定适合|最适合|绝不会|耐用一生")
PRESSURE = re.compile(r"抓紧|赶快|赶紧|只限今天|必须购买|别再犹豫|错过.{0,5}(?:就|没有|再无)|不买.{0,5}后悔")
DEROGATORY = re.compile(r"假货|山寨|劣质|不靠谱|不正规|渠道.{0,8}不可靠")
# Pricing causality has no authorizing field in the v1 input contract. Materials
# and features alone do not establish why a price was set. Keep this narrow rather
# than trying to score general persuasiveness with a regular expression.
PRICE_CAUSE = re.compile(
    r"(?:定价|售价|价格)(?:为|是|在|偏高|较高)?[A-Z￥¥$€£\d.,元万千\s]{0,20}[，,]?\s*"
    r"(?:主要)?(?:在于|来自|取决于|源于|是因为|因为|由于)|"
    r"(?:贵|价高|偏贵)(?:的原因)?(?:主要)?(?:是因为|在于|源于|因为)"
)
ERGONOMICS = re.compile(
    r"(?:肩背(?:使用)?|背起来|背着|上身|使用起来)(?:也|会)?(?:很|更|非常|十分|特别)(?:顺手|舒适|舒服|省力|轻松)|"
    r"(?:肩背|背起来|背着)(?:不会累|不勒肩|不压肩)|"
    r"(?:肩背|背起来|背着|使用起来)(?:一定|绝对|肯定)(?:顺手|舒适|舒服|省力|不累)"
)
NEGATION = re.compile(r"不(?:能|会|可|支持|提供|承诺|保证|确定|确认|接受|做|作|予|便|再)|没有|无法|未(?:有|提供|获|经|知)|暂(?:无|不)|不能擅自|并无|无额外|不享受")


def _positive_matches(text: str, pattern: re.Pattern, *, split_commas: bool = True) -> list[str]:
    findings = []
    boundaries = r"[，,。！？!?；;\n]|但是|不过|但" if split_commas else r"[。！？!?；;\n]|但是|不过|但"
    for clause in re.split(boundaries, text):
        for match in pattern.finditer(clause):
            # Conservative lexical handling only; scope, quotation and sarcasm need
            # review. Split adversative clauses so a denial cannot mask a promise.
            before = clause[max(0, match.start() - 18):match.start()]
            if not NEGATION.search(before):
                findings.append(match.group())
    return findings


def _discount_matches(text: str) -> list[str]:
    # A topic such as “关于九折” is not a promise when the immediately following
    # comma clause is a complete, explicit discount denial. Exempt only that topic;
    # keep checking every later clause so a denial cannot hide a subsequent offer.
    parts = re.split(r"([，,。！？!?；;\n]|但是|不过|但)", text)
    for index in range(0, len(parts) - 2, 2):
        if (
            parts[index + 1] in ("，", ",")
            and DISCOUNT_TOPIC.fullmatch(parts[index].strip())
            and DISCOUNT_DENIAL.fullmatch(parts[index + 2].strip())
        ):
            parts[index] = ""
    return _positive_matches("".join(parts), DISCOUNT)


def _normalized(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text) if not c.isspace() and not unicodedata.category(c).startswith("P"))


def _currency(token: str | None) -> str | None:
    if token is None:
        return None
    if token.isascii() and token.isalpha() and len(token) == 3:
        return CURRENCY.get(token.upper(), token.upper())
    return CURRENCY.get(token)


def _known_dates(request: Request) -> set[str]:
    dates = {request.reference_time.date().isoformat()}
    data = request.model_dump(mode="json")
    def walk(value):
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T.*)?", value):
            dates.add(value[:10])
    walk(data)
    return dates


def validate_output(raw: str | dict, request: Request, brand: Brand) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
        parsed = CandidateSet.model_validate(obj)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        # Diagnostics describe the contract, not whole untrusted provider text.
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors(include_input=False)) if isinstance(exc, ValidationError) else str(exc).splitlines()[0]
        return {"candidates": [], "errors": [f"OUTPUT_SCHEMA: {detail}"], "warnings": []}

    candidates = sorted((c.model_dump() for c in parsed.candidates), key=lambda c: c["id"])
    canonical = [_normalized(c["text"]) for c in candidates]
    if len(set(canonical)) != 3:
        errors.append("DUPLICATE_CANDIDATES: 去除标点及空白后候选必须不同")
    if any(SequenceMatcher(None, a, b).ratio() >= 0.85 for i, a in enumerate(canonical) for b in canonical[i + 1:]):
        warnings.append("三条候选文本相似度偏高；请复核表达策略差异。")

    valid_prices = {(p.price, p.currency) for p in request.products if p.price is not None}
    known_ids = {p.id for p in request.products}
    dates = _known_dates(request)
    for candidate in candidates:
        label, text = candidate["id"], candidate["text"]
        if not re.search(r"[\u3400-\u9fff]", text):
            errors.append(f"{label}: CHINESE_REQUIRED 正文必须含中文")
        if len(text) > brand.max_message_chars:
            errors.append(f"{label}: MAX_LENGTH 超过{brand.max_message_chars}字符")
        if not brand.target_message_chars[0] <= len(text) <= brand.target_message_chars[1]:
            warnings.append(f"{label}: 正文长度{len(text)}，偏离目标区间；硬上限仍单独校验。")
        for match in MONEY.finditer(text):
            prefix, suffix = match["prefix"], match["suffix"]
            if not prefix and not suffix:
                continue
            try:
                amount = Decimal(match["amount"].replace(",", "")) * {None: 1, "万": 10000, "千": 1000}[match["scale"]]
            except InvalidOperation:
                errors.append(f"{label}: MONEY_FORMAT 无法解析金额")
                continue
            pc = _currency(prefix)
            sc = _currency(suffix)
            if pc and sc and pc != sc:
                errors.append(f"{label}: CURRENCY_MISMATCH 金额币种相互冲突")
            if (amount, pc or sc) not in valid_prices:
                errors.append(f"{label}: UNKNOWN_PRICE {match.group().strip()}不是商品正式价格及币种")
        if re.search(r"[零〇一二两三四五六七八九十][零〇一二两三四五六七八九十百千万亿点]*(?:美元|港元|欧元|英镑|日元|元|块)", text):
            errors.append(f"{label}: MONEY_FORMAT 金额请用阿拉伯数字及已知币种")
        if _discount_matches(text):
            errors.append(f"{label}: DISCOUNT_UNAUTHORIZED 不得承诺、暗示或代为申请未授权优惠")
        if _positive_matches(text, SERVICE):
            errors.append(f"{label}: UNSUPPORTED_CLAIM 检测到未授权服务、库存或保证性承诺")
        if PRESSURE.search(text):
            errors.append(f"{label}: PRESSURE 不得催促或制造紧迫感")
        if _positive_matches(text, DEROGATORY):
            errors.append(f"{label}: DEROGATORY 不得贬低商品或渠道")
        if _positive_matches(text, PRICE_CAUSE, split_commas=False):
            errors.append(f"{label}: PRICE_RATIONALE_UNSUPPORTED 商品材质与价格不能推出定价因果")
        product_features = [feature for product in request.products for feature in product.features]
        for claim in _positive_matches(text, ERGONOMICS):
            if not any(_normalized(claim) in _normalized(feature) for feature in product_features):
                errors.append(f"{label}: ERGONOMICS_UNSUPPORTED 使用体验未在商品features中提供")
        for claim in brand.forbidden_claims:
            if _positive_matches(text, re.compile(re.escape(claim))):
                errors.append(f"{label}: BRAND_FORBIDDEN 出现品牌明确禁止的表述")
        for product_id in re.findall(r"(?<![A-Z0-9-])LX-[A-Z0-9-]+(?![A-Z0-9-])", text):
            if product_id not in known_ids:
                errors.append(f"{label}: UNKNOWN_PRODUCT 商品编号不在本次事实中")
        for date_text in re.findall(r"\d{4}-\d{2}-\d{2}", text):
            if date_text not in dates:
                errors.append(f"{label}: UNKNOWN_DATE 日期不在已知事实中")
        known_month_days = {(int(d[5:7]), int(d[8:10])) for d in dates}
        for month, day in re.findall(r"(?<!\d)(\d{1,2})月(\d{1,2})[日号]", text):
            if (int(month), int(day)) not in known_month_days:
                errors.append(f"{label}: UNKNOWN_DATE 月日不在已知事实中")
        if request.scenario == "greeting":
            if any(p.name in text for p in request.products) or re.search(r"下单|入手|新品|新款|购买", text):
                errors.append(f"{label}: GREETING_UPSELL 问候不可强塞商品推销")
            day_difference = (request.scenario_context.occasion_date - request.reference_time.date()).days if request.scenario_context.occasion_date else None
            if ("今天" in text and day_difference != 0) or ("明天" in text and day_difference != 1):
                errors.append(f"{label}: RELATIVE_DATE 问候的相对日期与reference_time不符")
        if request.scenario == "post_purchase_care":
            target = next((p for p in request.products if p.id == request.scenario_context.purchase.product_id), None) if request.scenario_context.purchase else None
            if target and not target.care_instructions and re.search(r"擦拭|涂抹|皮革油|清洗|清洁剂|吹干|晾干|水洗|酒精", text):
                errors.append(f"{label}: CARE_UNSUPPORTED 无护理资料时不可给出具体护理步骤")
            if re.search(r"再买|新品|新款|入手|下单", text):
                errors.append(f"{label}: CARE_UPSELL 购后关怀不可施压增购")
        if request.scenario == "event_invitation" and request.scenario_context.event:
            event = request.scenario_context.event
            if event.starts_at:
                weekdays = "一二三四五六日"
                for weekday in re.findall(r"(?:星期|周)([一二三四五六日天])", text):
                    if ("日" if weekday == "天" else weekday) != weekdays[event.starts_at.weekday()]:
                        errors.append(f"{label}: EVENT_WEEKDAY 活动星期与starts_at不符")
                allowed_times = {(event.starts_at.hour, event.starts_at.minute)}
                if event.ends_at:
                    allowed_times.add((event.ends_at.hour, event.ends_at.minute))
                for hour, minute in re.findall(r"(?<!\d)(\d{1,2})[:：](\d{2})(?!\d)", text):
                    if (int(hour), int(minute)) not in allowed_times:
                        errors.append(f"{label}: EVENT_TIME 活动时分不在已知时间中")

    warnings.append("自动检查仅覆盖结构和部分确定性规则；事实、语气及三候选差异仍待人工复核。")
    return {"candidates": candidates if not errors else [], "errors": list(dict.fromkeys(errors)), "warnings": list(dict.fromkeys(warnings))}
