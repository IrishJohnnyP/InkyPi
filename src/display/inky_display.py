import logging
from inky.auto import auto
from display.abstract_display import AbstractDisplay
from utils import eink


logger = logging.getLogger(__name__)


class InkyDisplay(AbstractDisplay):

    """
    Handles the Inky e-paper display.

    This class initializes and manages interactions with the Inky display,
    ensuring proper image rendering and configuration storage.

    The Inky display driver supports auto configuration.
    """

    def initialize_display(self):
        """
        Initializes the Inky display device.

        Sets the display border, detects Spectra 6 pure-palette bypass support,
        and stores the display resolution in the device configuration.
        """
        self.inky_display = auto()
        self.inky_display.set_border(self.inky_display.BLACK)

        # Detect Spectra 6 pure-palette bypass support once hardware is initialized
        detected = eink.palette_from_inky(self.inky_display)
        self.spectra_palette = detected if (detected and len(detected["pure"]) == 6) else None
        if self.spectra_palette:
            logger.info("Spectra 6 driver detected: using eink.py pure-palette bypass")

        # Store display resolution in device config
        if not self.device_config.get_config("resolution"):
            self.device_config.update_value(
                "resolution",
                [int(self.inky_display.width), int(self.inky_display.height)],
                write=True)

    def display_image(self, image, image_settings=[], colour_plan=None):
        """
        Displays the provided image on the Inky display.

        Args:
            image (PIL.Image): The image to be displayed.
            image_settings (list, optional): Additional layout settings.
            colour_plan (dict, optional): Precomputed rendering plan from eink.colour_plan().

        Raises:
            ValueError: If no image is provided.
        """
        logger.info("Displaying image to Inky display.")
        if not image:
            raise ValueError("No image provided.")

        global_image_settings = self.device_config.get_config("image_settings") or {}
        inky_saturation = global_image_settings.get("inky_saturation", 0.5)

        if getattr(self, "spectra_palette", None) is not None:
            plan = colour_plan or eink.colour_plan(
                image_settings=global_image_settings,
                image=image,
            )
            logger.info(f"Spectra 6 rendering mode: {plan['mode']} (saturation={plan['saturation']})")
            indices = eink.render_indices(
                image,
                self.spectra_palette,
                mode=plan["mode"],
                saturation=plan["saturation"],
            )
            pure_p_image = eink.to_pure_palette_image(indices, self.spectra_palette)
            self.inky_display.set_image(pure_p_image)
            self.inky_display.show()
            return

        # Fallback path for non-Spectra Inky displays
        logger.info(f"Inky Saturation: {inky_saturation}")
        try:
            self.inky_display.set_image(image, saturation=inky_saturation)
        except TypeError:
            self.inky_display.set_image(image)
        self.inky_display.show()
