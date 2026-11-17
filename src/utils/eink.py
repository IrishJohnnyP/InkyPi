"""Colour rendering for limited-palette e-paper displays (e.g. Inky Spectra 6).

Rendering modes:
- "photo": Cached 33x33x33 perceptual OKLab 3D LUT (skin-tone chroma protection,
  perceptual shadow/midtone lift, soft-knee gamut compression, sub-palette hue
  alignment, and paper-white highlight clip) paired with edge-preserving flat-area
  noise smoothing and dual-palette Floyd-Steinberg dithering.
- "vivid": High-punch OKLab 3D LUT + full perceived ink gamut dithering for
  landscapes and illustrations.
- "graphics": Tuned for UI-like images (weather, calendar, charts, text) with colour
  adaptation, tight palette snapping (preserving subtle card borders and pale tints),
  8x8 Bayer ordered dithering, and edge sharpening so black text renders solid black.
- "manual": Per-instance image adjustments (colourSaturation, colourContrast,
  colourBrightness, colourSharpness) and per-instance ink blend (colourInk) over a
  chosen rendering algorithm (colourRendering).
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

# Per-mode ink saturation (0.0 = pure driver RGB, 1.0 = full perceived ink gamut).
# Independent of global device.json -> image_settings["inky_saturation"].
MODE_INK_SATURATION = {
    PHOTO: 0.85,     # 3D LUT normalizes paper white & gamut, allowing accurate perceived-ink dithering
    GRAPHICS: 0.50,  # Solid black text + rich, clean color fills on UI dashboards
    VIVID: 1.00,     # Full perceived ink gamut normalized to paper white
}

# 3D LUT & perceptual OKLab curve parameters
LUT_SIZE = 33                           # 33x33x33 grid (~430 KB cached, C-level trilinear interpolation)
PHOTO_SHADOW_LIFT = 0.20                # Perceptual OKLab lightness lift for dark indoor shadows
PHOTO_MID_LIFT = 0.03                   # Gentle lower-midtone separation
PHOTO_LIFT_KNEE = 0.70                  # Tapers lift smoothly to 0 above OKLab L=0.70 so highlights stay intact
PHOTO_BLUE_DARKEN = 0.18                # Deepens mid/deep blues while keeping sky highlights airy
PHOTO_GREEN_DARKEN = 0.08               # Anchors foliage onto solid green ink
VIVID_GAMMA = 0.88                      # Perceptual lightness lift for "vivid" mode
PHOTO_SATURATION = 1.10                 # OKLab chroma boost (skin tones protected automatically)
VIVID_SATURATION = 1.22                 # Higher OKLab chroma boost for landscapes/illustrations
PHOTO_CONTRAST = 1.00                   # Neutral contrast
PHOTO_DENOISE_STRENGTH = 0.65           # Edge-preserving flat-area smoothing (integer math, low RAM)
PHOTO_DENOISE_EDGE = 18                 # Local RGB diff threshold above which edges/text stay untouched
GAMUT_COMPRESS_STRENGTH = 0.45          # Soft-knee OKLab chroma roll-off for out-of-gamut colors
WHITE_POINT_MULTS = (1.00, 1.00, 1.00)  # Neutral white balance
HIGHLIGHT_CLIP = 246                    # Snaps near-white paper backgrounds/clouds cleanly to #FFFFFF

# Cache for compiled PIL.ImageFilter.Color3DLUT instances keyed by (mode, saturation_rounded)
_LUT_CACHE = {}

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
    "immich", "icloud", "unsplash", "apod", "wpotd", "ai_image", "comic", "countdown",
    "screenshot", "newspaper", "clock", "daily_quote", "weather", "calendar",
}

GRAPHICS_PLUGINS = {
    "market_snapshot", "todo_list", "text",
    "college_football_rankings", "nd_football_schedule",
    "scoreboard", "nfl", "mlb", "nba", "nhl", "cfb", "cbb",
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


def mode_saturation(mode, image_settings=None):
    """Ink saturation for a specific rendering mode, ignoring global inky_saturation."""
    default = MODE_INK_SATURATION.get(mode, MODE_INK_SATURATION[PHOTO])
    if isinstance(image_settings, dict):
        overrides = image_settings.get("mode_saturation")
        if isinstance(overrides, dict) and mode in overrides:
            return _number(overrides[mode], default, 0.0, 1.0)
    return default


def colour_plan(plugin_id=None, settings=None, image_settings=None, image=None):
    """How to render an instance: algorithm, image enhancement and mode-specific ink saturation."""
    settings = settings or {}
    image_settings = dict(image_settings) if isinstance(image_settings, dict) else {}
    preferred_photo = image_settings.get("photo_mode", PHOTO)
    if preferred_photo not in (PHOTO, VIVID):
        preferred_photo = PHOTO
    plugin_modes = image_settings.get("plugin_modes")

    mode = resolve_mode(
        plugin_id,
        settings,
        image=image,
        photo_mode=preferred_photo,
        plugin_modes=plugin_modes,
    )
    saturation = mode_saturation(mode, image_settings=image_settings)

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


def _smooth_flat_areas(image, strength=PHOTO_DENOISE_STRENGTH, edge_threshold=PHOTO_DENOISE_EDGE):
    """Low-RAM integer edge-preserving denoise: smooths grain in flat skin/sky/walls while keeping edges sharp."""
    if strength <= 0.0:
        return image
    blurred = image.filter(ImageFilter.BoxBlur(1))
    orig = np.asarray(image, dtype=np.int16)
    blur = np.asarray(blurred, dtype=np.int16)
    diff = np.abs(orig - blur).max(axis=-1, keepdims=True)
    # Fixed-point weight in [0, 256] to avoid allocating full-frame float32 arrays
    max_w = int(round(strength * 256.0))
    denom = max(1, int(edge_threshold))
    weight = np.clip(((denom - (diff - 5)) * max_w) // denom, 0, max_w)
    out = orig + ((weight * (blur - orig)) >> 8)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


# ============================================================
# Perceptual OKLab 3D LUT Builder (cached once per mode)
# ============================================================

def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


def _linear_srgb_to_oklab(r, g, b):
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = np.cbrt(np.maximum(l, 0.0))
    m_ = np.cbrt(np.maximum(m, 0.0))
    s_ = np.cbrt(np.maximum(s, 0.0))

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ok = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, a, b_ok


def _oklab_to_linear_srgb(L, a, b_ok):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b_ok
    m_ = L - 0.1055613458 * a - 0.0638541728 * b_ok
    s_ = L - 0.0894841775 * a - 1.2914855480 * b_ok

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return r, g, b


def _build_spectra6_3dlut(mode=PHOTO, size=LUT_SIZE):
    """Compile a 33x33x33 perceptual OKLab + sub-palette alignment 3D LUT for Pillow."""
    # Pillow Color3DLUT expects R varying fastest, then G, then B (indexing="ij" with b, g, r)
    grid = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b_in, g_in, r_in = np.meshgrid(grid, grid, grid, indexing="ij")

    r_lin = _srgb_to_linear(r_in)
    g_lin = _srgb_to_linear(g_in)
    b_lin = _srgb_to_linear(b_in)

    L, a, b_ok = _linear_srgb_to_oklab(r_lin, g_lin, b_lin)
    C = np.hypot(a, b_ok)
    hue = np.arctan2(b_ok, a)
    hue_deg = np.degrees(hue) % 360.0

    if mode == VIVID:
        L = np.clip(L ** VIVID_GAMMA, 0.0, 1.0)
        base_sat = VIVID_SATURATION
    else:
        # Perceptual shadow & lower-midtone lift tapered cleanly to 0 above PHOTO_LIFT_KNEE
        if PHOTO_SHADOW_LIFT > 0.0 or PHOTO_MID_LIFT > 0.0:
            t = np.clip(L / PHOTO_LIFT_KNEE, 0.0, 1.0)
            w = (1.0 - t) ** 2 * (1.0 + 2.0 * t)
            L = np.clip(
                L
                + PHOTO_SHADOW_LIFT * (np.maximum(L, 1e-4) ** 0.75) * w
                + PHOTO_MID_LIFT * np.sin(np.pi * t) * (1.0 - t),
                0.0,
                1.0,
            )
        base_sat = PHOTO_SATURATION

    # Skin-tone chroma protection: warm hues (~25-65 deg in OKLab) resist over-saturation
    # so faces stay smooth instead of fracturing into red/yellow dither dots
    skin_hue_w = np.exp(-0.5 * ((hue_deg - 45.0) / 22.0) ** 2)
    skin_chroma_w = np.clip(1.0 - np.maximum(0.0, C - 0.14) / 0.08, 0.0, 1.0)
    skin_mask = skin_hue_w * skin_chroma_w
    sat_scale = 1.0 + (base_sat - 1.0) * (1.0 - 0.65 * skin_mask)
    C = C * sat_scale

    # Soft-knee OKLab chroma compression to pull extreme out-of-gamut colors onto Spectra 6 inks
    if GAMUT_COMPRESS_STRENGTH > 0.0:
        knee_c = 0.16
        excess = np.maximum(0.0, C - knee_c)
        C = np.where(C > knee_c, knee_c + excess / (1.0 + GAMUT_COMPRESS_STRENGTH * (excess / 0.16)), C)

    a = C * np.cos(hue)
    b_ok = C * np.sin(hue)

    r_out, g_out, b_out = _oklab_to_linear_srgb(L, a, b_ok)
    r = _linear_to_srgb(r_out)
    g = _linear_to_srgb(g_out)
    b = _linear_to_srgb(b_out)

    if mode == PHOTO:
        # Sub-palette harmonic alignment so each color family dithers with 1-2 inks
        # 1. Blue deepening + R/G alignment onto {Blue, White, Black}
        max_rg = np.maximum(r, g)
        blue_dom = np.clip((b - max_rg) / 0.25, 0.0, 1.0)
        if PHOTO_BLUE_DARKEN > 0.0:
            blue_luma = r * 0.299 + g * 0.587 + b * 0.114
            blue_weight = blue_dom * np.clip(1.15 - 0.75 * blue_luma, 0.20, 1.0)
            darken = 1.0 - PHOTO_BLUE_DARKEN * blue_weight
            r *= darken
            g *= darken
            b *= darken
        blue_align = np.clip((b - np.maximum(r, g)) / 0.12, 0.0, 1.0)
        rg_mid = 0.5 * (r + g)
        r = r + blue_align * (rg_mid - r)
        g = g + blue_align * (rg_mid - g)

        # 2. Green alignment onto {Green, Black, White, Yellow}
        max_rb = np.maximum(r, b)
        green_dom = np.clip((g - max_rb) / 0.12, 0.0, 1.0)
        g_adj = g * (1.0 - PHOTO_GREEN_DARKEN * green_dom) if PHOTO_GREEN_DARKEN > 0.0 else g
        pastel = np.clip((g_adj - 0.68) / 0.32, 0.0, 1.0)
        b_clean = g_adj * (0.18 + 0.55 * pastel)
        warm_offset = np.clip(r - b, 0.0, 0.25 * g_adj)
        r_clean = np.minimum(b_clean + warm_offset, 0.85 * g_adj)
        r = r + green_dom * (r_clean - r)
        g = g + green_dom * (g_adj - g)
        b = b + green_dom * (b_clean - b)

        # 3. Warm / skin alignment onto {Red, Yellow, White, Black}
        warm_mask = np.clip((r - np.maximum(g, b)) / 0.12, 0.0, 1.0) * np.clip((g - b + 0.04) / 0.08, 0.0, 1.0)
        b_warm = np.minimum(b, g * 0.92)
        b = b + warm_mask * (b_warm - b)

    # White-point balance & paper-white highlight clip
    r = np.clip(r * WHITE_POINT_MULTS[0], 0.0, 1.0)
    g = np.clip(g * WHITE_POINT_MULTS[1], 0.0, 1.0)
    b = np.clip(b * WHITE_POINT_MULTS[2], 0.0, 1.0)

    if HIGHLIGHT_CLIP is not None:
        clip_thresh = HIGHLIGHT_CLIP / 255.0
        near_white = np.minimum(np.minimum(r, g), b) > clip_thresh
        r = np.where(near_white, 1.0, r)
        g = np.where(near_white, 1.0, g)
        b = np.where(near_white, 1.0, b)

    table = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    return ImageFilter.Color3DLUT(size, table.ravel().tolist(), channels=3)


def get_spectra6_3dlut(mode=PHOTO):
    """Return a cached Color3DLUT filter for the given rendering mode."""
    lut = _LUT_CACHE.get(mode)
    if lut is None:
        lut = _build_spectra6_3dlut(mode=mode, size=LUT_SIZE)
        _LUT_CACHE[mode] = lut
        logger.info(f"Compiled {LUT_SIZE}^3 OKLab Color3DLUT for Spectra 6 mode='{mode}'")
    return lut


# ============================================================
# Graphics Mode & Quantization Helpers
# ============================================================

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


def _snap(rgb, quant_palette, threshold=26):
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


def render_indices(image, palette, mode, saturation=None):
    """Palette indices (one per pixel) as the display will show them."""
    if saturation is None:
        saturation = MODE_INK_SATURATION.get(mode, MODE_INK_SATURATION[PHOTO])

    if mode == GRAPHICS:
        quant = blended(palette, saturation)
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
    if mode == PHOTO:
        image = _smooth_flat_areas(image)
        if PHOTO_CONTRAST != 1.0:
            image = ImageEnhance.Contrast(image).enhance(PHOTO_CONTRAST)

    # Single-pass C-level OKLab 3D LUT transform
    image = image.filter(get_spectra6_3dlut(mode=mode))

    # Dual-palette quantization: dither against perceived paper-normalized pigments
    pure = np.array(palette["pure"], dtype=np.float32)
    perceived = perceived_palette(palette)
    quant = perceived * saturation + pure * (1.0 - saturation)

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


def simulate(image, palette, mode, saturation=None):
    """What the panel will look like (measured ink colours, normalised to the paper white)."""
    indices = render_indices(image, palette, mode, saturation)
    return Image.fromarray(perceived_palette(palette).astype(np.uint8)[indices], "RGB")
