







from __future__ import annotations

from qgis.PyQt.QtCore import QLocale

from ...core.i18n import tr


def grouped_locale() -> QLocale:






    loc = QLocale()
    try:
        loc.setNumberOptions(QLocale.NumberOption.DefaultNumberOptions)
    except (AttributeError, TypeError):
        pass  # nosec B110
    return loc


def _grouped_decimal(number: float, decimals: int) -> str:

    if decimals <= 0 or float(number).is_integer():
        return grouped_locale().toString(int(round(number)))
    loc = grouped_locale()
    text = loc.toString(float(number), "f", decimals)
    point = loc.decimalPoint()
    if point in text:
        text = text.rstrip("0").rstrip(point)
    return text


def format_km2_left(value) -> str:








    try:
        number = float(value)
    except (TypeError, ValueError):
        return "0"
    rounded = round(number, 1)
    if number > 0 and rounded == 0:
        rounded = round(number, 2)
        if rounded == 0:
            return "< " + _grouped_decimal(0.01, 2)
        return _grouped_decimal(rounded, 2)
    return _grouped_decimal(rounded, 1)


def _gauge_percent_used(used, cap, left) -> int:





    try:
        percent = min(100, int(round(100.0 * float(used) / float(cap))))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0
    if percent >= 100 and (left or 0) > 0:
        return 99
    return max(0, percent)


def format_quota_count(value) -> str:





    try:
        return grouped_locale().toString(int(value or 0))
    except (TypeError, ValueError):
        return "0"


def format_km2_surface(value) -> str:












    try:
        number = max(0.0, float(value))
    except (TypeError, ValueError):
        return "0"
    if number <= 0:
        return "0"
    if number < 0.005:
        return _grouped_decimal(max(number, 0.0001), 4)
    if number < 1:
        return _grouped_decimal(round(number, 2), 2)
    if number < 10:
        return _grouped_decimal(round(number, 1), 1)
    return grouped_locale().toString(int(round(number)))


class DockCreditsDisplayMixin:





    def _refresh_auto_credits_display(self):







        self._update_auto_low_credit_note()
        self._refresh_auto_upsell_title()

    def _refresh_auto_upsell_title(self):



        title = getattr(self, "_auto_upsell_title", None)
        if title is None:
            return
        from ...core.server_dials import dial_copy

        wall = getattr(self, "_auto_upsell_wall", None)
        if wall is not None:
            from ...core.pro_ceiling import pro_ceiling_contact_email
            from .upsell_card import keep_working_cta
            wall.set_contact_email(pro_ceiling_contact_email())


            wall.button.setText(keep_working_cta())



        env = getattr(self, "_quota_envelopes", None)
        reset_line = getattr(self, "_auto_upsell_reset", None)
        if reset_line is not None:
            reset_day = getattr(self, "_auto_reset_display", "")
            if reset_day and env is not None and env.km2_cap:






                reset_line.setText(dial_copy(
                    "trial.reset_km2",
                    tr("It comes back on {date}."),
                ).replace("{date}", reset_day))
            elif reset_day:
                reset_line.setText(dial_copy(
                    "trial.reset_count",
                    tr("Your free detections come back on {date}."),
                ).replace("{date}", reset_day))
            reset_line.setVisible(bool(reset_day))

        if env is not None and env.km2_cap:


            from ...core.quota_envelopes import format_km2_value
            title.setText(dial_copy(
                "trial.exhausted_km2",
                tr("You covered your {n} km² of Automatic this month"),
            ).replace("{n}", format_km2_value(env.km2_cap)))
            return

        total = self._auto_credits_total
        if total and total > 0:


            served = dial_copy(
                "trial.exhausted",
                tr("Your {n} free cloud detections are used up"))
            title.setText(served.replace("{n}", str(int(total))))
        else:
            title.setText(dial_copy(
                "trial.exhausted_no_count",
                tr("Your free cloud detections are used up")))
