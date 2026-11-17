import fnmatch
import json
import logging

from utils.image_utils import resize_image, change_orientation, apply_image_enhancement
from utils import eink
from display.mock_display import MockDisplay
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import hardware displays, but don't fail if they're not available
try:
    from display.inky_display import InkyDisplay
except ImportError:
    logger.info("Inky display not available, hardware support disabled")

try:
    from display.waveshare_display import WaveshareDisplay
except ImportError:
    logger.info("Waveshare display not available, hardware support disabled")


def boost_blue_green_channels(img):
    """
    Fallback pigment compensation for displays not using eink.py's measured Spectra palette.
    Isolates and brightens only the Green and Blue channels of an RGB image.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    r, g, b = img.split()
    g = g.point(lambda i: min(255, int(i * 1.2)))
    b = b.point(lambda i: min(255, int(i * 1.3)))
    return Image.merge("RGB", (r, g, b))


def _has_active_enhancements(image_settings):
    """True only if any global enhancement multiplier differs from neutral 1.0."""
    if not isinstance(image_settings, dict):
        return False
    for key in ("saturation", "contrast", "brightness", "sharpness"):
        try:
            if float(image_settings.get(key, 1.0)) != 1.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


class DisplayManager:

    """Manages the display and rendering of images."""

    def __init__(self, device_config):
        """
        Initializes the display manager and selects the correct display type
        based on the configuration.

        Args:
            device_config (object): Configuration object containing display settings.

        Raises:
            ValueError: If an unsupported display type is specified.
        """
        self.device_config = device_config

        display_type = device_config.get_config("display_type", default="inky")

        if display_type == "mock":
            self.display = MockDisplay(device_config)
        elif display_type == "inky":
            self.display = InkyDisplay(device_config)
        elif fnmatch.fnmatch(display_type, "epd*in*"):
            self.display = WaveshareDisplay(device_config)
        else:
            raise ValueError(f"Unsupported display type: {display_type}")

    def display_image(self, image, image_settings=[], plugin_id=None, plugin_settings=None):
        """
        Delegates image rendering to the appropriate display instance.

        Args:
            image (PIL.Image): The image to be displayed.
            image_settings (list, optional): List of layout flags (e.g. 'keep-width').
            plugin_id (str, optional): Active plugin identifier for mode routing.
            plugin_settings (dict, optional): Active plugin instance settings.

        Raises:
            ValueError: If no valid display instance is found.
        """
        if not hasattr(self, "display"):
            raise ValueError("No valid display instance initialized.")

        # Save clean preview for the web UI
        logger.info(f"Saving image to {self.device_config.current_image_file}")
        image.save(self.device_config.current_image_file)

        # Orient and resize only if dimensions do not already match the target panel
        target_resolution = tuple(self.device_config.get_resolution())
        image = change_orientation(image, self.device_config.get_config("orientation"))
        if image.size != target_resolution:
            image = resize_image(image, target_resolution, image_settings)
        if self.device_config.get_config("inverted_image"):
            image = image.rotate(180)

        # Skip redundant global enhancement passes when all multipliers are neutral 1.0
        global_image_settings = self.device_config.get_config("image_settings") or {}
        if _has_active_enhancements(global_image_settings):
            image = apply_image_enhancement(image, global_image_settings)

        # Route Spectra 6 displays through eink.py; use legacy G/B boost only for other displays
        if getattr(self.display, "spectra_palette", None) is not None:
            plan = eink.colour_plan(
                plugin_id=plugin_id,
                settings=plugin_settings,
                image_settings=global_image_settings,
                image=image,
                resolution=target_resolution,
            )
            logger.info(
                f"Spectra 6 colour plan (plugin={plugin_id}): "
                f"mode={plan['mode']}, saturation={plan['saturation']:.2f}"
            )
            self.display.display_image(image, image_settings=image_settings, colour_plan=plan)
        else:
            image = boost_blue_green_channels(image)
            self.display.display_image(image, image_settings=image_settings)
