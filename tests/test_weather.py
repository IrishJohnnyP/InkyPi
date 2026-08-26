import pytest
import pytz

from plugins.weather.weather import WeatherPlugin


class TestParseOpenMeteoForecast:

    def setup_method(self):
        # Instantiate without __init__ so we can test the parsing logic
        # in isolation, without needing a full plugin configuration.
        self.plugin = WeatherPlugin.__new__(WeatherPlugin)

    def _minimal_daily_data(self, dates):
        n = len(dates)
        return {
            "time": dates,
            "weathercode": [0] * n,
            "temperature_2m_max": [20.0] * n,
            "temperature_2m_min": [10.0] * n,
        }

    def test_day_label_west_of_utc(self):
        """
        Regression test: a date returned by Open-Meteo (a plain local date string)
        must not be shifted backwards when the device timezone is west of UTC.

        Open-Meteo returns dates as plain local date strings (e.g. "2026-02-27").
        The old code converted each date to a timezone-aware datetime before
        extracting the weekday label. For timezones west of UTC, this would
        interpret midnight local time as early morning UTC the next day — and
        when converted back, would land on the previous calendar day. So
        "2026-02-27" (Friday) would be labelled Thursday instead.

        The fix uses date.fromisoformat() directly, with no timezone conversion,
        since Open-Meteo already returns dates in local time.

        America/New_York (UTC-5) is used as a typical negative-offset zone that
        would trigger the old bug. 2026-02-27 is a Friday.
        """
        tz = pytz.timezone("America/New_York")
        daily_data = self._minimal_daily_data(["2026-02-27"])

        forecast = self.plugin.parse_open_meteo_forecast(
            daily_data, units="metric", tz=tz, is_day=1, lat=40.7
        )

        assert len(forecast) == 1
        assert forecast[0]["day"] == "Fri", (
            f"Expected 'Fri' for 2026-02-27 in America/New_York, got '{forecast[0]['day']}'"
        )
