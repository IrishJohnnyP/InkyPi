import os
import logging
import gc
from jinja2 import Environment, FileSystemLoader
from utils.image_loader import AdaptiveImageLoader
from utils.image_utils import take_screenshot_html

logger = logging.getLogger(__name__)


class BasePlugin:
    """
    Base class for all InkyPi plugins. Handles template rendering, 
    jinja context injection, and strict Spectra 6 hardware dithering.
    """

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self._setup_jinja_env()

    def _setup_jinja_env(self):
        """Configure Jinja2 environment to load templates from plugin directories."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Look for templates in standard paths relative to plugins
        candidate_paths = [
            os.path.abspath(os.path.join(current_dir, "../templates")),
            os.path.abspath(os.path.join(current_dir, "../../templates")),
            "/home/john/InkyPi/src/templates",
            "/usr/local/inkypi/src/templates"
        ]
        
        template_dir = next((p for p in candidate_paths if os.path.isdir(p)), candidate_paths[0])
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))

    def generate_settings_template(self):
        """Return default settings template structure for the plugin UI."""
        return {
            "style_settings": False
        }

    def generate_image(self, settings, device_config):
        """Override this method in subclasses to return a Pillow Image."""
        raise NotImplementedError("Plugins must implement generate_image()")

    def render_image(self, dimensions, html_filename, css_filename, context):
        """
        Renders HTML/CSS templates into a Pillow image and applies 
        strict 6-color Floyd-Steinberg dithering for Spectra 6 displays.
        """
        try:
            # 1. Load and render HTML template with context
            template = self.jinja_env.get_template(html_filename)
            
            # Read CSS file content if needed for inline injection
            css_content = ""
            if css_filename:
                try:
                    css_path = os.path.join(self.jinja_env.loader.searchpath[0], css_filename)
                    if os.path.exists(css_path):
                        with open(css_path, "r", encoding="utf-8") as f:
                            css_content = f.read()
                except Exception as css_err:
                    logger.warning(f"[{self.name}] Could not load CSS file {css_filename}: {css_err}")

            context["css_content"] = css_content
            html_str = template.render(context)

            # 2. Take browser screenshot of the rendered HTML
            img = take_screenshot_html(html_str, dimensions)

            if img is None:
                logger.error(f"[{self.name}] Failed to render screenshot from template.")
                return None

            # 3. Ensure image is in standard RGB mode before processing
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 4. Pass through AdaptiveImageLoader for strict 6-color hardware dither
            loader = AdaptiveImageLoader()
            dithered_img = loader._apply_spectra6_dither(img)

            # 5. Explicitly free the un-dithered original image to prevent RAM spikes
            # crucial for stability on low-resource devices like the Pi Zero 2
            del img
            gc.collect()

            logger.info(f"[{self.name}] Template rendered and dithered successfully for Spectra 6.")
            return dithered_img

        except Exception as e:
            logger.error(f"[{self.name}] Error during template rendering: {e}", exc_info=True)
            return None
