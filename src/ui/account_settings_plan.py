







from __future__ import annotations

from typing import NamedTuple

from ..core.activation_manager import get_dashboard_url, get_upgrade_url
from ..core.i18n import tr
from .dock.styles import GREEN_TEXT, ORANGE_TEXT, RED_INK
from .external_links import open_external_url

__all__ = [
    "AccountPlanMixin",
    "BalanceLine",
    "account_balance_lines",
    "format_km2_balance",
    "PlanCredits",
    "_PRO_MONTHLY_CREDITS_FALLBACK",
    "_STATUS_DISPLAY",
    "_as_int",
    "_resolve_is_subscriber",
    "resolve_plan_credits",
]


class PlanCredits(NamedTuple):


    is_subscriber: bool
    remaining: int | None
    total: int | None
    reset_date: str | None


def _as_int(value) -> int | None:

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_is_subscriber(usage: dict, sub: dict) -> bool:





    if "is_subscriber" in usage:
        return bool(usage["is_subscriber"])
    if "is_free_tier" in usage:
        return not bool(usage["is_free_tier"])
    return str(sub.get("plan", "")).lower() == "pro"








def resolve_plan_credits(usage: dict, sub: dict) -> PlanCredits:
    is_subscriber = _resolve_is_subscriber(usage, sub)
    remaining = _as_int(usage.get("remaining_credits"))
    total = _as_int(usage.get("total_credits"))
    if remaining is None or total is None:
        for limit, used in (
            (usage.get("images_limit"), usage.get("images_used")),
            (sub.get("quota_limit"), sub.get("usage_this_month")),
        ):
            limit_int = _as_int(limit)
            if limit_int is None:
                continue
            if total is None:
                total = limit_int
            if remaining is None:
                remaining = max(0, limit_int - (_as_int(used) or 0))
            break
    if total is None:


        from ..core.detection_policy import free_monthly_allowance
        from ..core.surface_dials import pro_monthly_credits_fallback

        total = (pro_monthly_credits_fallback(_PRO_MONTHLY_CREDITS_FALLBACK)
                 if is_subscriber else free_monthly_allowance())
    reset_date = (usage.get("reset_date") or usage.get("period_end")
                  or sub.get("current_period_end"))
    return PlanCredits(is_subscriber, remaining, total, reset_date)


class BalanceLine(NamedTuple):



    title: str
    figure: str
    caption: str
    left_units: int
    total_units: int
    spent: bool


def format_km2_balance(value) -> str:






    from .dock.ui_refresh import format_km2_left
    from .dock.ui_refresh_credits import grouped_locale

    try:
        number = float(value)
    except (TypeError, ValueError):
        return format_km2_left(value)
    if abs(number) >= 10:
        return grouped_locale().toString(int(round(number)))
    return format_km2_left(number)


def account_balance_lines(usage: dict, sub: dict) -> list:







    from ..core.quota_envelopes import quota_envelopes_from_account_row
    from .dock.ui_refresh import format_quota_count

    usage = usage or {}
    sub = sub or {}
    lines: list = []
    env = quota_envelopes_from_account_row(sub)
    if env is not None and env.has_objects_gauge():
        left = (env.objects_remaining if env.objects_remaining is not None
                else max(0, env.objects_cap - env.objects_used))
        lines.append(BalanceLine(
            tr("{n} of {total} cloud objects left in Semi-Auto this month").format(
                n=format_quota_count(left), total=format_quota_count(env.objects_cap)),
            format_quota_count(left),
            tr("Semi-Auto objects left of {total}").format(
                total=format_quota_count(env.objects_cap)),
            int(left), int(env.objects_cap), left <= 0))
    if env is not None and env.has_km2_gauge():
        km2_left = (env.km2_remaining if env.km2_remaining is not None
                    else max(0.0, env.km2_cap - env.km2_used))


        units = int(round(km2_left * 100))
        if units <= 0 < km2_left:
            units = 1
        lines.append(BalanceLine(
            tr("{n} of {total} km² left in Automatic this month").format(
                n=format_km2_balance(km2_left), total=format_km2_balance(env.km2_cap)),
            tr("{n} km²").format(n=format_km2_balance(km2_left)),
            tr("Automatic km² left of {total}").format(
                total=format_km2_balance(env.km2_cap)),
            units, int(round(env.km2_cap * 100)), km2_left <= 0))
    if lines:
        return lines
    plan = resolve_plan_credits(usage, sub)
    if plan.is_subscriber and plan.remaining is not None:
        total = plan.total or 0
        lines.append(BalanceLine(
            tr("{remaining} / {total} cloud detections").format(
                remaining=format_quota_count(plan.remaining),
                total=format_quota_count(total)),
            format_quota_count(plan.remaining),
            tr("cloud detections left of {total} this month").format(
                total=format_quota_count(total)),
            int(plan.remaining), int(total), plan.remaining <= 0))
        return lines
    free_left = _as_int(usage.get("free_detections_remaining"))
    if free_left is None:
        free_left = _as_int(sub.get("free_detections_remaining"))
    if plan.is_subscriber or free_left is None:
        return lines
    free_total = _as_int(usage.get("free_detections_total"))
    if free_total:
        title = tr("{n} of {total} free cloud detections left").format(
            n=format_quota_count(free_left), total=format_quota_count(free_total))
        caption = tr("free cloud detections left of {total} this month").format(
            total=format_quota_count(free_total))
    elif free_left == 1:
        title = tr("1 free cloud detection remaining")
        caption = tr("free cloud detections left this month")
    else:
        title = tr("{n} free cloud detections remaining").format(
            n=format_quota_count(free_left))
        caption = tr("free cloud detections left this month")
    lines.append(BalanceLine(title, format_quota_count(free_left), caption,
                             int(free_left), int(free_total or 0), free_left <= 0))
    return lines







_PRO_MONTHLY_CREDITS_FALLBACK = 2000




_STATUS_DISPLAY = {
    "active": (tr("Active"), GREEN_TEXT),
    "trialing": (tr("Free trial"), ORANGE_TEXT),
    "canceled": (tr("Cancelled"), RED_INK),


    "past_due": (tr("Your last payment may have failed"), RED_INK),
    "unpaid": (tr("Your last payment may have failed"), RED_INK),
}


class AccountPlanMixin:







    def _on_upgrade_clicked(self):



        from ..core.pro_page_link import open_pro_page
        open_pro_page("plugin_account_dialog", "account_dialog", parent=self,
                      fallback_url=get_upgrade_url())

    def _open_dashboard(self, source: str = "account_card"):



        try:
            from ..core import telemetry_session_events
            telemetry_session_events.track_account_dashboard_opened(source)
        except Exception:
            pass  # nosec B110
        open_external_url(get_dashboard_url(), parent=self)
