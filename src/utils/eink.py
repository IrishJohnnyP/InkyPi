"""Colour rendering for limited-palette e-paper displays (e.g. Inky Spectra 6).

Rendering modes:
- "photo": Floyd-Steinberg dithering towards the driver's blended palette (controlled by
  inky_saturation) with gentle midtone lift, cool-white paper compensation, and highlight
  preservation—natural skin tones and smooth gradients without heavy ink clustering.
- "vivid": Floyd-Steinberg dithering towards the inks as they really look (measured
  palette, relative to the paper white) for high-punch landscapes/illustrations.
- "graphics": tuned for UI-like images (weather, calendar, charts, text) with colour
  adaptation, palette snapping, 8x8 Bayer ordered dithering, and edge sharpening so
  black text renders solid black instead of grey dithered dots.

Mode selection priority:
  1. Per-instance setting: plugin_settings["colourMode"]
  2. Per-plugin config map: device.json -> image_settings["plugin_modes"][plugin_id]
  3. Built-in plugin sets: PHOTO_PLUGINS / GRAPHICS_PLUGINS
  4. Automatic pixel-statistic fallback: detect_mode(image)
"""

import colorsys
import inspect
import logging
import re

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

PHOTO, VIVID, GRAPHICS, MANUAL = "photo", "vivid", "graphics", "manual"
MODES = (PHOTO, VIVID, GRAPHICS, MANUAL)
RENDERINGS = (PHOTO, VIVID, GRAPHICS)
_ALIASES = {"classic": PHOTO}

# Photo & Vivid rendering curves
PHOTO_GAMMA = 0.87                      # Gentle lift (1 / 1.15) for driver-blended "photo" mode
VIVID_GAMMA = 0.87                      # Balanced lift (1 / 1.15) for measured-ink "vivid" mode
PHOTO_SATURATION = 1.0                  # Measured palette + image_loader profile already supply saturation
WHITE_POINT_MULTS = (0.98, 1.00, 1.06)  # Cool-white paper tint compensation (R, G, B)
HIGHLIGHT_CLIP = 248                    # Preserves highlight detail while stopping off-white stipple

# Manual mode: per-instance values replacing the global image settings
MANUAL_FIELDS = {
    "saturation": "colourSaturation",
    "contrast": "colourContrast",
    "brightness": "colourBrightness",
    "sharpness": "colourSharpness",
}
RENDER_FIELDS = {"colourMode", "colourRendering", "colourInk", *MANUAL_FIELDS.values()}

# Built-in plugin classification sets (overridable via image_settings["plugin_modes"] in device.json)
PHOTO_PLUGINS = {
    "image_folder", "image_upload", "image_album", "image_url", "google_photos",
    "immich", "icloud", "unsplash", "apod", "wpotd", "ai_image", "comic",
    "screenshot", "newspaper", "clock",
}

GRAPHICS_PLUGINS = {
    "market_snapshot", "daily_quote", "weather", "calendar", "todo_list", "countdown",
    "college_football_rankings", "nd_football_schedule", "scoreboard",
    "nfl", "mlb", "nba", "nhl", "cfb", "cbb",
}

# Inky Spectra 6 (E673 / EL133UF1) palettes, used as fallback or in development simulation
SPECTRA6 = {
    "pure": [[0, 0, 0], [255, 255, 255], [255, 255, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]],
    "measured": [[0, 0, 0], [161, 164, 165], [208, 190, 71], [156, 72, 75], [61, 59, 94], [58, 91, 70]],
}

_BAYER8 = (np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], dtype=np.float32) + 0.5) / 64.0 - 0.5


def default_mode(plugin_id, image=None, photo_mode=PHOTO, plugin_modes=None):
    """Recommended rendering for a plugin: config map -> built-in sets -> pixel auto-detection."""
    if plugin_id:
        if isinstance(plugin_modes, dict) and plugin_id in plugin_modes:
            configured = _ALIASES.get(plugin_modes[plugin_id], plugin_modes[plugin_id])
            if configured in RENDERINGS:
                return configured
        if plugin_id in PHOTO_PLUGINS:
            return photo_mode
        if plugin_id in GRAPHICS_PLUGINS:
            return GRAPHICS
    return detect_mode(image, photo_mode=photo_mode)


def detect_mode(image, photo_mode=PHOTO):
    """Classify an image as photo vs. pure UI graphics from pixel statistics (~2ms)."""
    if image is None:
        return photo_mode

    w, h = image.size
    if w > 400:
        scale = 400.0 / w
        sample = image.resize((400, max(1, int(h * scale))), Image.NEAREST)
    else:
        sample = image

    arr = np.asarray(sample.convert("RGB"), dtype=np.int16)

    # 1. Fraction of horizontal & vertical neighbors with zero RGB difference
    same_x = np.all(arr[:, 1:] == arr[:, :-1], axis=-1).mean()
    same_y = np.all(arr[1:, :] == arr[:-1, :], axis=-1).mean()
    flat_ratio = float(0.5 * (same_x + same_y))

    # 2. Fraction of pixels covered by the 8 most frequent exact RGB colors
    packed = (
        (arr[..., 0].astype(np.uint32) << 16)
        | (arr[..., 1].astype(np.uint32) << 8)
        | arr[..., 2]
    )
    _, counts = np.unique(packed, return_counts=True)
    top8_share = float(np.sort(counts)[-8:].sum() / packed.size)

    is_graphics = (flat_ratio > 0.60 and top8_share > 0.45) or (flat_ratio > 0.78)
    return GRAPHICS if is_graphics else photo_mode


def resolve_mode(plugin_id=None, settings=None, image=None, photo_mode=PHOTO, plugin_modes=None):
    """Rendering algorithm: instance setting -> plugin map -> built-in sets -> auto-detect."""
    settings = settings or {}
    mode = settings.get("colourMode")
    fallback = default_mode(plugin_id, image=image, photo_mode=photo_mode, plugin_modes=plugin_modes)

    if mode == MANUAL:
        rendering = _ALIASES.get(settings.get("colourRendering"), settings.get("colourRendering"))
        return rendering if rendering in RENDERINGS else fallback
    mode = _ALIASES.get(mode, mode)
    return mode if mode in RENDERINGS else fallback


def _number(value, default, low, high):
    try:
        return min(high, max(low, float(value)))
    except (TypeError, ValueError):
        return default


def colour_plan(plugin_id=None, settings=None, image_settings=None, image=None):
    """How to render an instance: algorithm, image enhancement and ink saturation."""
    settings = settings or {}
    image_settings = dict(image_settings) if isinstance(image_settings, dict) else {}
    saturation = _number(image_settings.get("inky_saturation"), 0.5, 0.0, 1.0)
    preferred_photo = image_settings.get("photo_mode", PHOTO)
    if preferred_photo not in (PHOTO, VIVID):
        preferred_photo = PHOTO
    plugin_modes = image_settings.get("plugin_modes")

    manual = settings.get("colourMode") == MANUAL
    if manual:
        for key, field in MANUAL_FIELDS.items():
            image_settings[key] = _number(
                settings.get(field),
                _number(image_settings.get(key), 1.0, 0.0, 2.0),
                0.0,
                2.0,
            )
        saturation = _number(settings.get("colourInk"), saturation, 0.0, 1.0)
        image_settings["inky_saturation"] = saturation

    mode = resolve_mode(
        plugin_id,
        settings,
        image=image,
        photo_mode=preferred_photo,
        plugin_modes=plugin_modes,
    )
    return {"mode": mode, "enhance": image_settings, "saturation": saturation, "manual": manual}


def palette_from_inky(inky_display):
    """Describes an Inky driver's palette, or None if it has no pure-colour (no dither) path."""
    module = inspect.getmodule(type(inky_display))
    pure = getattr(module, "DESATURATED_PALETTE", None)
    measured = getattr(module, "SATURATED_PALETTE", None)
    try:
        source = inspect.getsource(type(inky_display).set_image)
    except (AttributeError, OSError, TypeError):
        return None
    match = re.search(r"len\(image\.palette\.colors\)\s*==\s*(\d+)", source)
    if not (pure and measured and match):
        return None
    count = int(match.group(1))
    if len(pure) < count or len(measured) < count:
        return None
    return {"pure": [list(c) for c in pure[:count]], "measured": [list(c) for c in measured[:count]]}


def blended(palette, saturation):
    """The driver's quantization palette for a given 'saturation' (pure <-> measured)."""
    pure = np.array(palette["pure"], dtype=np.float32)
    measured = np.array(palette["measured"], dtype=np.float32)
    return measured * saturation + pure * (1.0 - saturation)


def _white_index(palette):
    pure = np.array(palette["pure"])
    return int(np.argmax(pure.sum(axis=1)))


def _apply_photo_tonecurve(image, gamma=1.0, white_point=WHITE_POINT_MULTS, highlight_clip=HIGHLIGHT_CLIP):
    """Single-pass LUT combining gamma lift, cool-white balance, and highlight preservation."""
    lut = []
    for channel_mult in white_point:
        for i in range(256):
            val = 255.0 * ((i / 255.0) ** gamma) if i > 0 else 0.0
            val *= channel_mult
            if highlight_clip is not None and val > highlight_clip:
                val = 255.0
            lut.append(max(0, min(255, int(round(val)))))
    return image.point(lut)


def adapt_colours(image, palette, strength=0.4, boost=1.7, min_chroma=24):
    """Saturate coloured pixels and darken them according to physical ink darkness."""
    measured = np.array(palette["measured"], dtype=np.float32)
    white = measured[_white_index(palette)]
    white_luma = float(white @ [0.299, 0.587, 0.114]) or 255.0

    inks = []
    for rgb in measured:
        h, s, v = colorsys.rgb_to_hsv(*(rgb / 255.0))
        if s < 0.15:
            continue
        darkness = max(0.0, 1.0 - float(rgb @ [0.299, 0.587, 0.114]) / white_luma)
        inks.append((h * 360.0, darkness))
    if not inks:
        return image

    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    chroma = rgb.max(-1) - rgb.min(-1)
    weight = np.clip((chroma - min_chroma) / 40.0, 0, 1)

    hue = hsv[..., 0] * 360.0
    ink_hues = np.array([h for h, _ in inks], dtype=np.float32)
    ink_dark = np.array([d for _, d in inks], dtype=np.float32)
    dist = np.abs(((hue[..., None] - ink_hues[None, None, :]) + 180.0) % 360.0 - 180.0)
    inv = 1.0 / (dist + 1.0) ** 2
    darkness = (inv * ink_dark).sum(-1) / inv.sum(-1)

    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + (boost - 1) * weight), 0, 1)
    hsv[..., 2] = hsv[..., 2] * (1 - strength * darkness * weight * 1.6)
    return Image.fromarray((np.clip(hsv, 0, 1) * 255).astype(np.uint8), "HSV").convert("RGB")


def _distances(rgb, quant_palette):
    """Squared distance of each pixel to each palette colour (|x|^2 - 2x.p + |p|^2)."""
    pal = quant_palette.astype(np.float32)
    dist = rgb @ (-2.0 * pal.T)
    dist += (pal ** 2).sum(-1)
    dist += (rgb ** 2).sum(-1, keepdims=True)
    return dist


def _snap(rgb, quant_palette, threshold=48):
    """Pixels within `threshold` of a quantization colour become exactly that colour."""
    dist = _distances(rgb, quant_palette)
    nearest = dist.argmin(-1)
    close = dist.min(-1) <= threshold ** 2
    out = rgb.copy()
    out[close] = quant_palette[nearest[close]]
    return out


def _neutral_inks(palette):
    """Indices of the achromatic inks (black, white): greys are dithered with these only."""
    measured = np.array(palette["measured"], dtype=np.float32)
    return [i for i, rgb in enumerate(measured) if colorsys.rgb_to_hsv(*(rgb / 255.0))[1] < 0.15]


def _ordered_indices(rgb, quant_palette, neutral_inks=(), spread=96.0, neutral_chroma=24):
    h, w, _ = rgb.shape
    threshold = np.tile(_BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w][..., None] * spread
    shifted = rgb + threshold
    neutral = (rgb.max(-1) - rgb.min(-1)) < neutral_chroma
    chromatic = np.ones(len(quant_palette), dtype=bool)
    chromatic[list(neutral_inks)] = False
    shifted = shifted.astype(np.float32)
    indices = np.empty((h, w), dtype=np.uint8)
    for y0 in range(0, h, 120):
        band = shifted[y0:y0 + 120]
        dist = _distances(band, quant_palette)
        if len(neutral_inks) >= 2:
            dist[neutral[y0:y0 + 120][..., None] & chromatic[None, None, :]] = np.inf
        indices[y0:y0 + 120] = dist.argmin(-1)
    return indices


def _sharpen_edges(indices, pixels, rgb, palette, neutral_inks, edge_range=64, text_threshold=170,
                   neutral_chroma=24):
    """Snap grey anti-aliasing pixels on sharp edges to pure black or white ink."""
    if len(neutral_inks) < 2:
        return indices
    measured = np.array(palette["measured"], dtype=np.float32)
    luma_of = lambda c: c @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    black = min(neutral_inks, key=lambda i: luma_of(measured[i]))
    white = max(neutral_inks, key=lambda i: luma_of(measured[i]))
    grey = Image.fromarray(pixels).convert("L")
    local_range = (np.asarray(grey.filter(ImageFilter.MaxFilter(3)), dtype=np.int16)
                   - np.asarray(grey.filter(ImageFilter.MinFilter(3)), dtype=np.int16))
    neutral = (rgb.max(-1) - rgb.min(-1)) < neutral_chroma
    edge = neutral & (local_range > edge_range)
    luma = np.asarray(grey, dtype=np.int16)
    out = indices.copy()
    out[edge] = np.where(luma[edge] < text_threshold, black, white)
    return out


def render_indices(image, palette, mode, saturation=0.5):
    """Palette indices (one per pixel) as the display will show them."""
    quant = blended(palette, saturation)
    if mode == GRAPHICS:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        packed = (pixels[..., 0].astype(np.uint32) << 16) | (pixels[..., 1].astype(np.uint32) << 8) | pixels[..., 2]
        colours, inverse = np.unique(packed.ravel(), return_inverse=True)
        unique_rgb = np.stack([(colours >> 16) & 255, (colours >> 8) & 255, colours & 255], -1).astype(np.uint8)
        adapted = adapt_colours(Image.fromarray(unique_rgb[None, :, :], "RGB"), palette)
        snapped = _snap(np.asarray(adapted, dtype=np.float32), quant)[0]
        rgb = snapped[inverse.ravel()].reshape(pixels.shape)
        neutral_inks = _neutral_inks(palette)
        indices = _ordered_indices(rgb, quant, neutral_inks)
        return _sharpen_edges(indices, pixels, rgb, palette, neutral_inks)

    image = image.convert("RGB")
    if mode == VIVID:
        quant = perceived_palette(palette)
        image = _apply_photo_tonecurve(image, gamma=VIVID_GAMMA)
        if PHOTO_SATURATION != 1.0:
            image = ImageEnhance.Color(image).enhance(PHOTO_SATURATION)
    else:  # PHOTO
        image = _apply_photo_tonecurve(image, gamma=PHOTO_GAMMA)

    pal_image = Image.new("P", (1, 1))
    pal_image.putpalette([int(v) for v in np.asarray(quant).flatten()])
    quantized = image.quantize(len(quant), palette=pal_image, dither=Image.Dither.FLOYDSTEINBERG)
    return np.asarray(quantized, dtype=np.uint8)


def to_pure_palette_image(indices, palette):
    """Palette ("P") image in the display's pure colours: the driver sends it without dithering."""
    image = Image.fromarray(indices.astype(np.uint8), "P")
    image.putpalette([int(v) for colour in palette["pure"] for v in colour])
    return image


def perceived_palette(palette):
    """The inks as the eye sees them: measured colours relative to the paper white."""
    measured = np.array(palette["measured"], dtype=np.float32)
    white = measured[_white_index(palette)]
    return np.clip(measured / np.maximum(white, 1) * 255.0, 0, 255)


def simulate(image, palette, mode, saturation=0.5):
    """What the panel will look like (measured ink colours, normalised to the paper white)."""
    indices = render_indices(image, palette, mode, saturation)
    return Image.fromarray(perceived_palette(palette).astype(np.uint8)[indices], "RGB")
