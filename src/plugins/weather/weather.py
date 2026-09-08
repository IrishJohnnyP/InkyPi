import os
import json
import time
import math
import logging
import pytz
import requests
from datetime import datetime, timezone, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from plugins.base_plugin.base_plugin import BasePlugin
from utils.http_client import get_http_session

logger = logging.getLogger(__name__)

CACHE_DIR = "/tmp/inkypi_weather_cache"
CACHE_TTL = 1800  # 30 minutes in seconds

def get_moon_phase_name(phase_age: float) -> str:
    PHASES_THRESHOLDS = [
        (1.0, "newmoon"),
        (7.0, "waxingcrescent"),
        (8.5, "firstquarter"),
        (14.0, "waxinggibbous"),
        (15.5, "fullmoon"),
        (22.0, "waninggibbous"),
        (23.5, "lastquarter"),
        (29.0, "waningcrescent"),
    ]
    for threshold, phase_name in PHASES_THRESHOLDS:
        if phase_age <= threshold:
            return phase_name  
    return "newmoon"

UNITS = {
    "standard": {"temperature": "K", "speed": "m/s", "distance": "km"},
    "metric": {"temperature": "°C", "speed": "m/s", "distance": "km"},
    "imperial": {"temperature": "°F", "speed": "mph", "distance": "mi"}
}

WEATHER_URL = "https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={long}&units={units}&exclude=minutely&appid={api_key}"
AIR_QUALITY_URL = "http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={long}&appid={api_key}"
GEOCODING_URL = "http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={long}&limit=1&appid={api_key}"

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={long}&hourly=weather_code,temperature_2m,precipitation,precipitation_probability,relative_humidity_2m,surface_pressure,visibility&daily=weathercode,temperature_2m_max,temperature_2m_min,sunrise,sunset&current=temperature,windspeed,winddirection,is_day,precipitation,weather_code,apparent_temperature&timezone=auto&models=best_match&forecast_days={forecast_days}"
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={long}&hourly=european_aqi,uv_index,uv_index_clear_sky&timezone=auto"
OPEN_METEO_UNIT_PARAMS = {
    "standard": "temperature_unit=celsius&wind_speed_unit=ms&precipitation_unit=mm",
    "metric":   "temperature_unit=celsius&wind_speed_unit=ms&precipitation_unit=mm",
    "imperial": "temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
}

class Weather(BasePlugin):
    def generate_settings_template(self):
        template_params = super().generate_settings_template()
        template_params['api_key'] = {
            "required": True,
            "service": "OpenWeatherMap",
            "expected_key": "OPEN_WEATHER_MAP_SECRET"
        }
        template_params['style_settings'] = True
        return template_params

    def _get_cached_data(self, cache_key):
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_file):
            if time.time() - os.path.getmtime(cache_file) < CACHE_TTL:
                try:
                    with open(cache_file, "r") as f:
                        return json.load(f)
                except Exception:
                    pass
        return None

    def _save_cached_data(self, cache_key, data):
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def generate_image(self, settings, device_config):
        raw_lat = settings.get('latitude', '').strip()
        raw_long = settings.get('longitude', '').strip()
        if not raw_lat or not raw_long:
            raise RuntimeError("Latitude and Longitude are required.")
        try:
            lat = float(raw_lat)
            long = float(raw_long)
        except (ValueError, TypeError):
            raise RuntimeError("Latitude and Longitude must be valid numbers.")

        units = settings.get('units')
        if not units or units not in ['metric', 'imperial', 'standard']:
            raise RuntimeError("Units are required.")

        weather_provider = settings.get('weatherProvider', 'OpenWeatherMap')
        title = settings.get('customTitle', '')

        timezone = device_config.get_config("timezone", default="America/New_York")
        time_format = device_config.get_config("time_format", default="12h")
        tz = pytz.timezone(timezone)

        try:
            cache_key = f"{weather_provider}_{lat}_{long}_{units}"
            cached_bundle = self._get_cached_data(cache_key)

            if cached_bundle:
                weather_data = cached_bundle.get("weather")
                aqi_data = cached_bundle.get("aqi")
                title = cached_bundle.get("title", title)
            else:
                if weather_provider == "OpenWeatherMap":
                    api_key = device_config.load_env_key("OPEN_WEATHER_MAP_SECRET")
                    if not api_key:
                        raise RuntimeError("Open Weather Map API Key not configured.")
                    
                    # Parallelize OWM requests using ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        future_weather = executor.submit(self.get_weather_data, api_key, units, lat, long)
                        future_aqi = executor.submit(self.get_air_quality, api_key, lat, long)
                        future_location = executor.submit(self.get_location, api_key, lat, long) if settings.get('titleSelection', 'location') == 'location' else None

                        weather_data = future_weather.result()
                        aqi_data = future_aqi.result()
                        if future_location:
                            title = future_location.result()

                elif weather_provider == "OpenMeteo":
                    forecast_days = 7
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        future_weather = executor.submit(self.get_open_meteo_data, lat, long, units, forecast_days + 1)
                        future_aqi = executor.submit(self.get_open_meteo_air_quality, lat, long)

                        weather_data = future_weather.result()
                        aqi_data = future_aqi.result()
                else:
                    raise RuntimeError(f"Unknown weather provider: {weather_provider}")

                self._save_cached_data(cache_key, {"weather": weather_data, "aqi": aqi_data, "title": title})

            if weather_provider == "OpenWeatherMap":
                if settings.get('weatherTimeZone', 'locationTimeZone') == 'locationTimeZone':
                    wtz = self.parse_timezone(weather_data)
                    template_params = self.parse_weather_data(weather_data, aqi_data, wtz, units, time_format, lat)
                else:
                    template_params = self.parse_weather_data(weather_data, aqi_data, tz, units, time_format, lat)
            else:
                template_params = self.parse_open_meteo_data(weather_data, aqi_data, tz, units, time_format, lat)

            template_params['title'] = title
        except Exception as e:
            logger.error(f"{weather_provider} request failed: {str(e)}")
            raise RuntimeError(f"{weather_provider} request failure, please check logs.")
       
        dimensions = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            dimensions = dimensions[::-1]

        template_params["plugin_settings"] = settings

        now = datetime.now(tz)
        if time_format == "24h":
            last_refresh_time = now.strftime("%Y-%m-%d %H:%M")
        else:
            last_refresh_time = now.strftime("%Y-%m-%d %I:%M %p")
        template_params["last_refresh_time"] = last_refresh_time

        image = self.render_image(dimensions, "weather.html", "weather.css", template_params)

        if not image:
            raise RuntimeError("Failed to take screenshot, please check logs.")
        return image

    def parse_weather_data(self, weather_data, aqi_data, tz, units, time_format, lat):
        current = weather_data.get("current")
        daily_forecast = weather_data.get("daily", [])
        dt = datetime.fromtimestamp(current.get('dt'), tz=timezone.utc).astimezone(tz)
        current_icon = current.get("weather")[0].get("icon")
        icon_codes_to_preserve = ["01", "02", "10"]
        icon_code = current_icon[:2]
        current_suffix = current_icon[-1]

        if icon_code not in icon_codes_to_preserve:
            if current_icon.endswith('n'):
                current_icon = current_icon.replace("n", "d")
        data = {
            "current_date": dt.strftime("%A, %B %d"),
            "current_day_icon": self.get_plugin_dir(f'icons/{current_icon}.png'),
            "current_temperature": str(round(current.get("temp"))),
            "feels_like": str(round(current.get("feels_like"))),
            "temperature_unit": UNITS[units]["temperature"],
            "units": units,
            "time_format": time_format
        }
        data['forecast'] = self.parse_forecast(weather_data.get('daily'), tz, current_suffix, lat)
        data['data_points'] = self.parse_data_points(weather_data, aqi_data, tz, units, time_format)
        data['hourly_forecast'] = self.parse_hourly(weather_data.get('hourly'), tz, time_format, units, daily_forecast)
        return data

    def parse_open_meteo_data(self, weather_data, aqi_data, tz, units, time_format, lat):
        current = weather_data.get("current", {})
        daily = weather_data.get('daily', {})
        dt = datetime.fromisoformat(current.get('time')).astimezone(tz) if current.get('time') else datetime.now(tz)
        weather_code = current.get("weather_code", 0)
        is_day = current.get("is_day", 1)
        current_icon = self.map_weather_code_to_icon(weather_code, is_day)
        
        temperature_conversion = 273.15 if units == "standard" else 0.

        data = {
            "current_date": dt.strftime("%A, %B %d"),
            "current_day_icon": self.get_plugin_dir(f'icons/{current_icon}.png'),
            "current_temperature": str(round(current.get("temperature", 0) + temperature_conversion)),
            "feels_like": str(round(current.get("apparent_temperature", current.get("temperature", 0)) + temperature_conversion)),
            "temperature_unit": UNITS[units]["temperature"],
            "units": units,
            "time_format": time_format
        }

        data['forecast'] = self.parse_open_meteo_forecast(weather_data.get('daily', {}), units, tz, is_day, lat)
        data['data_points'] = self.parse_open_meteo_data_points(weather_data, aqi_data, units, tz, time_format)
        data['hourly_forecast'] = self.parse_open_meteo_hourly(weather_data.get('hourly', {}), units, tz, time_format, daily.get('sunrise', []), daily.get('sunset', []))
        return data

    def map_weather_code_to_icon(self, weather_code, is_day):
        icon = "01d"
        if weather_code in [0]:   icon = "01d"
        elif weather_code in [1]: icon = "022d"
        elif weather_code in [2]: icon = "02d"
        elif weather_code in [3]: icon = "04d"
        elif weather_code in [51, 61, 80]: icon = "51d"          
        elif weather_code in [53, 63, 81]: icon = "53d"
        elif weather_code in [55, 65, 82]: icon = "09d"
        elif weather_code in [45]: icon = "50d"                       
        elif weather_code in [48]: icon = "48d"
        elif weather_code in [56, 66]: icon = "56d"            
        elif weather_code in [57, 67]: icon = "57d"            
        elif weather_code in [71, 85]: icon = "71d"
        elif weather_code in [73]:     icon = "73d"
        elif weather_code in [75, 86]: icon = "13d"
        elif weather_code in [77]:     icon = "77d"
        elif weather_code in [95, 96, 99]: icon = "11d"

        if is_day == 0:
            if icon == "01d": icon = "01n"
            elif icon == "022d": icon = "022n"
            elif icon == "02d": icon = "02n"                
            elif icon == "10d": icon = "10n"
        return icon

    def get_moon_phase_icon_path(self, phase_name: str, lat: float) -> str:
        if lat < 0:
            if phase_name == "waxingcrescent": phase_name = "waningcrescent"
            elif phase_name == "waxinggibbous": phase_name = "waninggibbous"
            elif phase_name == "waningcrescent": phase_name = "waxingcrescent"
            elif phase_name == "waninggibbous": phase_name = "waxinggibbous"
            elif phase_name == "firstquarter": phase_name = "lastquarter"
            elif phase_name == "lastquarter": phase_name = "firstquarter"
        return self.get_plugin_dir(f"icons/{phase_name}.png")

    def parse_forecast(self, daily_forecast, tz, current_suffix, lat):
        PHASES = [
            (0.0, "newmoon"), (0.25, "firstquarter"), (0.5, "fullmoon"), (0.75, "lastquarter"), (1.0, "newmoon"),
        ]

        def choose_phase_name(phase: float) -> str:
            for target, name in PHASES:
                if math.isclose(phase, target, abs_tol=1e-3):
                    return name
            if 0.0 < phase < 0.25: return "waxingcrescent"
            elif 0.25 < phase < 0.5: return "waxinggibbous"
            elif 0.5 < phase < 0.75: return "waninggibbous"
            else: return "waningcrescent"

        forecast = []
        icon_codes_to_apply_current_suffix = ["01", "02", "10"]
        for day in daily_forecast:
            weather_icon = day["weather"][0]["icon"]
            icon_code = weather_icon[:2]
            if icon_code in icon_codes_to_apply_current_suffix:
                weather_icon = weather_icon[:-1] + current_suffix
            else:
                if weather_icon.endswith('n'):
                    weather_icon = weather_icon.replace("n", "d")
            weather_icon = f"{icon_code}d"        
            weather_icon_path = self.get_plugin_dir(f"icons/{weather_icon}.png")

            moon_phase = float(day["moon_phase"])
            phase_name_north_hemi = choose_phase_name(moon_phase)
            moon_icon_path = self.get_moon_phase_icon_path(phase_name_north_hemi, lat)
            illum_fraction = (1 - math.cos(2 * math.pi * moon_phase)) / 2
            moon_pct = f"{illum_fraction * 100:.0f}"

            dt = datetime.fromtimestamp(day["dt"], tz=timezone.utc).astimezone(tz)
            forecast.append({
                "day": dt.strftime("%a"),
                "high": int(day["temp"]["max"]),
                "low": int(day["temp"]["min"]),
                "icon": weather_icon_path,
                "moon_phase_pct": moon_pct,
                "moon_phase_icon": moon_icon_path,
            })
        return forecast
        
    def parse_open_meteo_forecast(self, daily_data, units, tz, is_day, lat):
        times = daily_data.get('time', [])
        weather_codes = daily_data.get('weathercode', [])
        temp_max = daily_data.get('temperature_2m_max', [])
        temp_min = daily_data.get('temperature_2m_min', [])
        if units == "standard":
            temp_max = [T + 273.15 for T in temp_max]
            temp_min = [T + 273.15 for T in temp_min]

        forecast = []
        for i in range(0, len(times)): 
            local_date = date.fromisoformat(times[i])
            code = weather_codes[i] if i < len(weather_codes) else 0
            weather_icon = self.map_weather_code_to_icon(code, is_day=1)
            weather_icon_path = self.get_plugin_dir(f"icons/{weather_icon}.png")

            try:
                phase_age = moon.phase(local_date)
                phase_name_north_hemi = get_moon_phase_name(phase_age)
                phase_fraction = phase_age / 29.530588853
                illum_pct = (1 - math.cos(2 * math.pi * phase_fraction)) / 2 * 100
            except Exception as e:
                logger.error(f"Error calculating moon phase for {local_date}: {e}")
                illum_pct = 0
                phase_name_north_hemi = "newmoon"
            moon_icon_path = self.get_moon_phase_icon_path(phase_name_north_hemi, lat)

            forecast.append({
                "day": local_date.strftime("%a"),
                "high": int(temp_max[i]) if i < len(temp_max) else 0,
                "low": int(temp_min[i]) if i < len(temp_min) else 0,
                "icon": weather_icon_path,
                "moon_phase_pct": f"{illum_pct:.0f}",
                "moon_phase_icon": moon_icon_path
            })
        return forecast

    def parse_hourly(self, hourly_forecast, tz, time_format, units, daily_forecast):
        hourly = []
        icon_codes_to_preserve = ["01", "02", "10"]
        sun_map = {}
        for day in daily_forecast:
            day_date = datetime.fromtimestamp(day['dt'], tz=timezone.utc).astimezone(tz).date()
            sun_map[day_date] = (day['sunrise'], day['sunset'])
        
        for hour in hourly_forecast[:24]:
            dt_epoch = hour.get('dt')
            dt = datetime.fromtimestamp(dt_epoch, tz=timezone.utc).astimezone(tz)
            rain_mm = hour.get("rain", {}).get("1h", 0.0)
            snow_mm = hour.get("snow", {}).get("1h", 0.0)
            total_precip_mm = rain_mm + snow_mm
            sunrise, sunset = sun_map.get(dt.date(), (0, 0))
        
            is_day = sunrise <= dt_epoch < sunset
            suffix = 'd' if is_day else 'n'
            raw_icon = hour.get("weather", [{}])[0].get("icon", "01d")
            icon_base = raw_icon[:2]
            icon_name = f"{icon_base}{suffix}" if icon_base in icon_codes_to_preserve else f"{icon_base}d"
            
            precip_value = (total_precip_mm / 25.4) if units == "imperial" else total_precip_mm
            hourly.append({
                "time": self.format_time(dt, time_format, hour_only=True),
                "temperature": int(hour.get("temp")),
                "precipitation": hour.get("pop"),
                "rain": round(precip_value, 2),
                "icon": self.get_plugin_dir(f'icons/{icon_name}.png')
            })
        return hourly

    def parse_open_meteo_hourly(self, hourly_data, units, tz, time_format, sunrises, sunsets):
        hourly = []
        times = hourly_data.get('time', [])
        temperatures = hourly_data.get('temperature_2m', [])
        if units == "standard":
            temperatures = [t + 273.15 for t in temperatures]
        precipitation_probabilities = hourly_data.get('precipitation_probability', [])
        rain = hourly_data.get('precipitation', [])
        codes = hourly_data.get('weather_code', [])
        
        sun_map = {}
        for sr_s, ss_s in zip(sunrises, sunsets):
            sr_dt = datetime.fromisoformat(sr_s).astimezone(tz)
            ss_dt = datetime.fromisoformat(ss_s).astimezone(tz)
            sun_map[sr_dt.date()] = (sr_dt, ss_dt)
        
        current_time_in_tz = datetime.now(tz)
        start_index = 0
        for i, time_str in enumerate(times):
            try:
                dt_hourly = datetime.fromisoformat(time_str).astimezone(tz)
                if dt_hourly.date() == current_time_in_tz.date() and dt_hourly.hour >= current_time_in_tz.hour:
                    start_index = i
                    break
                if dt_hourly.date() > current_time_in_tz.date():
                    break
            except ValueError:
                continue

        sliced_times = times[start_index:]
        sliced_temperatures = temperatures[start_index:]
        sliced_precipitation_probabilities = precipitation_probabilities[start_index:]
        sliced_rain = rain[start_index:]
        sliced_codes = codes[start_index:]

        for i in range(min(24, len(sliced_times))):
            dt = datetime.fromisoformat(sliced_times[i]).astimezone(tz)
            sunrise, sunset = sun_map.get(dt.date(), (None, None))
            is_day = 1 if (sunrise and sunset and sunrise <= dt < sunset) else 0
            code = sliced_codes[i] if i < len(sliced_codes) else 0
            icon_name = self.map_weather_code_to_icon(code, is_day)
            hourly.append({
                "time": self.format_time(dt, time_format, True),
                "temperature": int(sliced_temperatures[i]) if i < len(sliced_temperatures) else 0,
                "precipitation": (sliced_precipitation_probabilities[i] / 100) if i < len(sliced_precipitation_probabilities) else 0,
                "rain": sliced_rain[i] if i < len(sliced_rain) else 0,
                "icon": self.get_plugin_dir(f"icons/{icon_name}.png")
            })
        return hourly

    def parse_data_points(self, weather, air_quality, tz, units, time_format):
        data_points = []
        sunrise_epoch = weather.get('current', {}).get("sunrise")
        if sunrise_epoch:
            sunrise_dt = datetime.fromtimestamp(sunrise_epoch, tz=timezone.utc).astimezone(tz)
            data_points.append({
                "label": "Sunrise",
                "measurement": self.format_time(sunrise_dt, time_format, include_am_pm=False),
                "unit": "" if time_format == "24h" else sunrise_dt.strftime('%p'),
                "icon": self.get_plugin_dir('icons/sunrise.png')
            })

        sunset_epoch = weather.get('current', {}).get("sunset")
        if sunset_epoch:
            sunset_dt = datetime.fromtimestamp(sunset_epoch, tz=timezone.utc).astimezone(tz)
            data_points.append({
                "label": "Sunset",
                "measurement": self.format_time(sunset_dt, time_format, include_am_pm=False),
                "unit": "" if time_format == "24h" else sunset_dt.strftime('%p'),
                "icon": self.get_plugin_dir('icons/sunset.png')
            })

        wind_deg = weather.get('current', {}).get("wind_deg", 0)
        data_points.append({
            "label": "Wind",
            "measurement": weather.get('current', {}).get("wind_speed"),
            "unit": UNITS[units]["speed"],
            "icon": self.get_plugin_dir('icons/wind.png'),
            "arrow": self.get_wind_arrow(wind_deg)
        })

        data_points.append({
            "label": "Humidity",
            "measurement": weather.get('current', {}).get("humidity"),
            "unit": '%',
            "icon": self.get_plugin_dir('icons/humidity.png')
        })

        data_points.append({
            "label": "Pressure",
            "measurement": weather.get('current', {}).get("pressure"),
            "unit": 'hPa',
            "icon": self.get_plugin_dir('icons/pressure.png')
        })

        data_points.append({
            "label": "UV Index",
            "measurement": weather.get('current', {}).get("uvi"),
            "unit": '',
            "icon": self.get_plugin_dir('icons/uvi.png')
        })

        visibility = weather.get('current', {}).get("visibility")
        if units == "imperial":
            visibility /= 1609.
            at_max_visibility = visibility >= 6.2
        else:
            visibility /= 1000.
            at_max_visibility = visibility >= 10
        visibility_str = f"{visibility:.1f}"
        if at_max_visibility:
            visibility_str = u"\u2265" + visibility_str
        data_points.append({
            "label": "Visibility",
            "measurement": visibility_str,
            "unit": UNITS[units]["distance"],
            "icon": self.get_plugin_dir('icons/visibility.png')
        })

        aqi = air_quality.get('list', [])[0].get("main", {}).get("aqi")
        data_points.append({
            "label": "Air Quality",
            "measurement": aqi,
            "unit": ["Good", "Fair", "Moderate", "Poor", "Very Poor"][int(aqi)-1],
            "icon": self.get_plugin_dir('icons/aqi.png')
        })
        return data_points

    def parse_open_meteo_data_points(self, weather_data, aqi_data, units, tz, time_format):
        data_points = []
        daily_data = weather_data.get('daily', {})
        current_data = weather_data.get('current', {})
        hourly_data = weather_data.get('hourly', {})
        current_time = datetime.now(tz)

        sunrise_times = daily_data.get('sunrise', [])
        if sunrise_times:
            sunrise_dt = datetime.fromisoformat(sunrise_times[0]).astimezone(tz)
            data_points.append({
                "label": "Sunrise",
                "measurement": self.format_time(sunrise_dt, time_format, include_am_pm=False),
                "unit": "" if time_format == "24h" else sunrise_dt.strftime('%p'),
                "icon": self.get_plugin_dir('icons/sunrise.png')
            })

        sunset_times = daily_data.get('sunset', [])
        if sunset_times:
            sunset_dt = datetime.fromisoformat(sunset_times[0]).astimezone(tz)
            data_points.append({
                "label": "Sunset",
                "measurement": self.format_time(sunset_dt, time_format, include_am_pm=False),
                "unit": "" if time_format == "24h" else sunset_dt.strftime('%p'),
                "icon": self.get_plugin_dir('icons/sunset.png')
            })

        data_points.append({
            "label": "Wind", "measurement": current_data.get("windspeed", 0), "unit": UNITS[units]["speed"],
            "icon": self.get_plugin_dir('icons/wind.png'), "arrow": self.get_wind_arrow(current_data.get("winddirection", 0))
        })

        current_humidity = "N/A"
        for i, time_str in enumerate(hourly_data.get('time', [])):
            try:
                if datetime.fromisoformat(time_str).astimezone(tz).hour == current_time.hour:
                    current_humidity = int(hourly_data.get('relative_humidity_2m', [])[i])
                    break
            except ValueError:
                continue
        data_points.append({
            "label": "Humidity", "measurement": current_humidity, "unit": '%',
            "icon": self.get_plugin_dir('icons/humidity.png')
        })

        current_pressure = "N/A"
        for i, time_str in enumerate(hourly_data.get('time', [])):
            try:
                if datetime.fromisoformat(time_str).astimezone(tz).hour == current_time.hour:
                    current_pressure = int(hourly_data.get('surface_pressure', [])[i])
                    break
            except ValueError:
                continue
        data_points.append({
            "label": "Pressure", "measurement": current_pressure, "unit": 'hPa',
            "icon": self.get_plugin_dir('icons/pressure.png')
        })

        current_uv_index = "N/A"
        for i, time_str in enumerate(aqi_data.get('hourly', {}).get('time', [])):
            try:
                if datetime.fromisoformat(time_str).astimezone(tz).hour == current_time.hour:
                    current_uv_index = aqi_data.get('hourly', {}).get('uv_index', [])[i]
                    break
            except ValueError:
                continue
        data_points.append({
            "label": "UV Index", "measurement": current_uv_index, "unit": '',
            "icon": self.get_plugin_dir('icons/uvi.png')
        })

        current_visibility = "N/A"
        visibility_conversion = 1/5280. if units == "imperial" else 0.001
        visibility_max = 6.2 if units == "imperial" else 10.
        for i, time_str in enumerate(hourly_data.get('time', [])):
            try:
                if datetime.fromisoformat(time_str).astimezone(tz).hour == current_time.hour:
                    current_visibility = hourly_data.get('visibility', [])[i] * visibility_conversion
                    at_max_visibility = current_visibility >= visibility_max
                    break
            except ValueError:
                continue
        visibility_str = f"{current_visibility:.1f}"
        if at_max_visibility:
            visibility_str = u"\u2265" + visibility_str
        data_points.append({
            "label": "Visibility", "measurement": visibility_str, "unit": UNITS[units]["distance"],
            "icon": self.get_plugin_dir('icons/visibility.png')
        })

        current_aqi = "N/A"
        for i, time_str in enumerate(aqi_data.get('hourly', {}).get('time', [])):
            try:
                if datetime.fromisoformat(time_str).astimezone(tz).hour == current_time.hour:
                    current_aqi = round(aqi_data.get('hourly', {}).get('european_aqi', [])[i], 1)
                    break
            except ValueError:
                continue
        scale = ["Good","Fair","Moderate","Poor","Very Poor","Ext Poor"][min(int(current_aqi)//20, 5)] if current_aqi != "N/A" else ""
        data_points.append({
            "label": "Air Quality", "measurement": current_aqi, "unit": scale, "icon": self.get_plugin_dir('icons/aqi.png')
        })
        return data_points

    def get_wind_arrow(self, wind_deg: float) -> str:
        DIRECTIONS = [
            ("↓", 22.5), ("↙", 67.5), ("←", 112.5), ("↖", 157.5),
            ("↑", 202.5), ("↗", 247.5), ("→", 292.5), ("↘", 337.5), ("↓", 360.0)
        ]
        wind_deg = wind_deg % 360
        for arrow, upper_bound in DIRECTIONS:
            if wind_deg < upper_bound:
                return arrow
        return "↑"

    def get_weather_data(self, api_key, units, lat, long):
        url = WEATHER_URL.format(lat=lat, long=long, units=units, api_key=api_key)
        response = requests.get(url, timeout=(3.05, 10))
        response.raise_for_status()
        return response.json()

    def get_air_quality(self, api_key, lat, long):
        url = AIR_QUALITY_URL.format(lat=lat, long=long, api_key=api_key)
        response = requests.get(url, timeout=(3.05, 10))
        response.raise_for_status()
        return response.json()

    def get_location(self, api_key, lat, long):
        url = GEOCODING_URL.format(lat=lat, long=long, api_key=api_key)
        response = requests.get(url, timeout=(3.05, 10))
        response.raise_for_status()
        location_data = response.json()[0]
        return f"{location_data.get('name')}, {location_data.get('state', location_data.get('country'))}"

    def get_open_meteo_data(self, lat, long, units, forecast_days):
        url = OPEN_METEO_FORECAST_URL.format(lat=lat, long=long, forecast_days=forecast_days) + f"&{OPEN_METEO_UNIT_PARAMS[units]}"
        response = requests.get(url, timeout=(3.05, 10))
        response.raise_for_status()
        return response.json()

    def get_open_meteo_air_quality(self, lat, long):
        url = OPEN_METEO_AIR_QUALITY_URL.format(lat=lat, long=long)
        response = requests.get(url, timeout=(3.05, 10))
        response.raise_for_status()
        return response.json()
    
    def format_time(self, dt, time_format, hour_only=False, include_am_pm=True):
        if time_format == "24h":
            return dt.strftime("%H:00" if hour_only else "%H:%M")
        fmt = "%I %p" if (hour_only and include_am_pm) else ("%I:%M %p" if include_am_pm else ("%I" if hour_only else "%I:%M"))
        return dt.strftime(fmt).lstrip("0")
    
    def parse_timezone(self, weatherdata):
        if 'timezone' in weatherdata:
            return pytz.timezone(weatherdata['timezone'])
        raise RuntimeError("Timezone not found in weather data.")
