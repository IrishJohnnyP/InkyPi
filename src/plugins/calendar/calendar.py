import os
import hashlib
import time
import json
import logging
import requests
from datetime import datetime, timedelta
import pytz
import icalendar
import recurring_ical_events
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.app_utils import resolve_path, get_font
from plugins.base_plugin.base_plugin import BasePlugin
from plugins.calendar.constants import LOCALE_MAP, FONT_SIZES
from PIL import Image, ImageColor, ImageDraw, ImageFont
from io import BytesIO

logger = logging.getLogger(__name__)

CACHE_DIR = "/tmp/inkypi_calendar_cache"
CACHE_TTL = 900  # 15 minutes in seconds

class Calendar(BasePlugin):
    def generate_settings_template(self):
        template_params = super().generate_settings_template()
        template_params['style_settings'] = True
        template_params['locale_map'] = LOCALE_MAP
        return template_params

    def _get_cached_or_fetch_calendar(self, calendar_url):
        os.makedirs(CACHE_DIR, exist_ok=True)
        url_hash = hashlib.md5(calendar_url.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{url_hash}.ics")
        
        # Check local cache validity
        if os.path.exists(cache_file):
            if time.time() - os.path.getmtime(cache_file) < CACHE_TTL:
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        ics_text = f.read()
                        return icalendar.Calendar.from_ical(ics_text)
                except Exception:
                    pass
        
        # Fetch fresh if cache is missing or expired
        ics_text = self.fetch_calendar_text(calendar_url)
        if ics_text:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(ics_text)
            except Exception:
                pass
            return icalendar.Calendar.from_ical(ics_text)
        return None

    def fetch_calendar_text(self, calendar_url):
        if calendar_url.startswith("webcal://"):
            calendar_url = calendar_url.replace("webcal://", "https://")
        try:
            # Fail fast with a (connect, read) timeout tuple instead of hanging for 30s
            response = requests.get(calendar_url, timeout=(3.05, 10))
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"[{self.name}] Failed to fetch iCalendar url {calendar_url}: {str(e)}")
            return None

    def generate_image(self, settings, device_config):
        calendar_urls = settings.get('calendarURLs[]')
        calendar_colors = settings.get('calendarColors[]')
        view = settings.get("viewMode")

        if not view:
            raise RuntimeError("View is required")
        elif view not in ["timeGridDay", "timeGridWeek", "dayGrid", "dayGridMonth", "listMonth"]:
            raise RuntimeError("Invalid view")

        if not calendar_urls:
            raise RuntimeError("At least one calendar URL is required")
        for url in calendar_urls:
            if not url.strip():
                raise RuntimeError("Invalid calendar URL")

        dimensions = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            dimensions = dimensions[::-1]
        
        timezone = device_config.get_config("timezone", default="America/New_York")
        time_format = device_config.get_config("time_format", default="12h")
        tz = pytz.timezone(timezone)

        current_dt = datetime.now(tz)
        start, end = self.get_view_range(view, current_dt, settings)
        logger.debug(f"Fetching events for {start} --> [{current_dt}] --> {end}")
        
        events = self.fetch_ics_events(calendar_urls, calendar_colors, tz, start, end)
        if not events:
            logger.warning("No events found for ics url")

        if view == 'timeGridWeek' and settings.get("displayPreviousDays") != "true":
            view = 'timeGrid'

        template_params = {
            "view": view,
            "events": events,
            "current_dt": current_dt.replace(minute=0, second=0, microsecond=0).isoformat(),
            "timezone": timezone,
            "plugin_settings": settings,
            "time_format": time_format,
            "font_scale": FONT_SIZES.get(settings.get("fontSize", "normal"))
        }

        image = self.render_image(dimensions, "calendar.html", "calendar.css", template_params)

        if not image:
            raise RuntimeError("Failed to take screenshot, please check logs.")
        return image
    
    def fetch_ics_events(self, calendar_urls, colors, tz, start_range, end_range):
        parsed_events = []

        # Concurrently fetch all configured calendars using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(calendar_urls) or 1) as executor:
            future_to_url = {
                executor.submit(self._get_cached_or_fetch_calendar, url): (url, color) 
                for url, color in zip(calendar_urls, colors)
            }
            
            for future in as_completed(future_to_url):
                url, color = future_to_url[future]
                try:
                    cal = future.result()
                    if not cal:
                        continue
                    events = recurring_ical_events.of(cal).between(start_range, end_range)
                    contrast_color = self.get_contrast_color(color)

                    for event in events:
                        start, end, all_day = self.parse_data_points(event, tz)
                        parsed_event = {
                            "title": str(event.get("summary")),
                            "start": start,
                            "backgroundColor": color,
                            "textColor": contrast_color,
                            "allDay": all_day
                        }
                        if end:
                            parsed_event['end'] = end

                        parsed_events.append(parsed_event)
                except Exception as e:
                    logger.error(f"[{self.name}] Failed processing events for calendar {url}: {e}")

        return parsed_events
    
    def get_view_range(self, view, current_dt, settings):
        start = datetime(current_dt.year, current_dt.month, current_dt.day)
        if view == "timeGridDay":
            end = start + timedelta(days=1)
        elif view == "timeGridWeek":
            if settings.get("displayPreviousDays") == "true":
                week_start_day = int(settings.get("weekStartDay", 1))
                python_week_start = (week_start_day - 1) % 7
                offset = (current_dt.weekday() - python_week_start) % 7
                start = current_dt - timedelta(days=offset)
                start = datetime(start.year, start.month, start.day)
            end = start + timedelta(days=7)
        elif view == "dayGrid":
            start = current_dt - timedelta(weeks=1)
            end = current_dt + timedelta(weeks=int(settings.get("displayWeeks") or 4))
        elif view == "dayGridMonth":
            start = datetime(current_dt.year, current_dt.month, 1) - timedelta(weeks=1)
            end = datetime(current_dt.year, current_dt.month, 1) + timedelta(weeks=6)
        elif view == "listMonth":
            end = start + timedelta(weeks=5)
        return start, end
        
    def parse_data_points(self, event, tz):
        all_day = False
        dtstart = event.decoded("dtstart")
        if isinstance(dtstart, datetime):
            start = dtstart.astimezone(tz).isoformat()
        else:
            start = dtstart.isoformat()
            all_day = True

        end = None
        if "dtend" in event:
            dtend = event.decoded("dtend")
            if isinstance(dtend, datetime):
                end = dtend.astimezone(tz).isoformat()
            else:
                end = dtend.isoformat()
        elif "duration" in event:
            duration = event.decoded("duration")
            end = (dtstart + duration).isoformat()
        return start, end, all_day

    def get_contrast_color(self, color):
        r, g, b = ImageColor.getrgb(color)
        yiq = (r * 299 + g * 587 + b * 114) / 1000
        return '#000000' if yiq >= 150 else '#ffffff'
