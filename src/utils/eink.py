"""Colour rendering for limited-palette e-paper displays (e.g. Inky Spectra 6)."""

import colorsys
import glob
import hashlib
import inspect
import logging
import os
import re
import tempfile
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

PHOTO, VIVID, GRAPHICS, HYBRID, MANUAL = "photo", "vivid", "graphics", "hybrid", "manual"
MODES = (PHOTO, VIVID, GRAPHICS, HYBRID, MANUAL)
RENDERINGS = (PHOTO, VIVID, GRAPHICS, HYBRID)
_ALIASES = {"classic": PHOTO, "mixed": HYBRID, "overlay": HYBRID}

# Per-mode ink saturation (0.0 = pure driver RGB, 1.0 = full calibrated perceived ink gamut).
MODE_INK_SATURATION = {
    PHOTO: 1.00,
    HYBRID: 1.00,
    GRAPHICS: 0.50,
    VIVID: 1.00,
}

# Default 3D LUT & perceptual OKLab curve parameters (overridable in device.json)
LUT_SIZE = 33
PHOTO_SHADOW_LIFT = 0.10
PHOTO_MID_LIFT = 0.03
PHOTO_LIFT_KNEE = 0.70
PHOTO_BLUE_DARKEN = 0.18
PHOTO_GREEN_DARKEN = 0.08
VIVID_GAMMA = 0.88
PHOTO_SATURATION = 1.12
VIVID_SATURATION = 1.22
PHOTO_CONTRAST = 1.00
PHOTO_DENOISE_STRENGTH = 0.65
PHOTO_DENOISE_EDGE = 18
PHOTO_SHARPEN_RADIUS = 1.0
PHOTO_SHARPEN_PERCENT = 45
PHOTO_SHARPEN_THRESHOLD = 4
GAMUT_COMPRESS_STRENGTH = 0.40
WHITE_POINT_MULTS = (1.00, 1.00, 1.00)
HIGHLIGHT_CLIP = 246

# Bounded RAM & /tmp disk caches for compiled 3D LUTs
_LUT_CACHE = {}
_MAX_LUT_CACHE = 6
_MAX_DISK_LUT_CACHE = 8

MANUAL_FIELDS = {
    "saturation": "colourSaturation",
    "contrast": "colourContrast",
    "brightness": "colourBrightness",
    "sharpness": "colourSharpness",
}
RENDER_FIELDS = {"colourMode", "colourRendering", "colourInk", *MANUAL_FIELDS.values()}

PHOTO_PLUGINS = {
    "image_folder", "image_upload", "image_album", "image_url", "google_photos",
    "immich", "icloud", "unsplash", "apod", "wpotd", "ai_image", "comic",
}

HYBRID_PLUGINS = {
    "calendar", "weather", "daily_quote", "countdown", "clock", "newspaper", "screenshot",
    "market_snapshot", "nd_football_schedule", "college_football_rankings",
    "scoreboard", "nfl", "mlb", "nhl", "nba", "cfb", "cbb",
}

GRAPHICS_PLUGINS = {
    "todo_list", "text",
}

SPECTRA6 = {
    "pure": [[0, 0, 0], [255, 255, 255], [255, 255, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]],
    "measured": [[0, 0, 0], [161, 164, 165], [208, 190, 71], [156, 72, 75], [61, 59, 94], [58, 91, 70]],
}

CALIBRATED_PERCEIVED = np.array([
    [15, 18, 20],
    [255, 255, 255],
    [239, 222, 68],
    [178, 35, 35],
    [42, 88, 178],
    [36, 108, 54],
], dtype=np.float32)

PANEL_PROFILES = {
    (1600, 1200): {
        "sharpen_radius": 1.0,
        "sharpen_percent": 45,
        "sharpen_threshold": 4,
        "perceived": CALIBRATED_PERCEIVED,
    },
    (800, 480): {
        "sharpen_radius": 0.8,
        "sharpen_percent": 35,
        "sharpen_threshold": 4,
        "perceived": np.array([
            [18, 20, 22],
            [255, 255, 255],
            [236, 218, 65],
            [172, 38, 38],
            [45, 92, 175],
            [38, 105, 56],
        ], dtype=np.float32),
    },
}

_BAYER8 = (np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], dtype=np.float32) + 0.5) / 64.0 - 0.5


def _panel_profile(resolution=None):
    """Look up panel-specific calibration (matches both landscape and portrait orientations)."""
    if resolution and len(resolution) >= 2:
        w, h = int(resolution[0]), int(resolution[1])
        if (w, h) in PANEL_PROFILES:
            return PANEL_PROFILES[(w, h)]
        if (h, w) in PANEL_PROFILES:
            return PANEL_PROFILES[(h, w)]
    return PANEL_PROFILES[(1600, 1200)]


def default_mode(plugin_id, image=None, photo_mode=PHOTO, plugin_modes=None, settings=None):
    """Recommended rendering for a plugin: config map -> built-in sets -> pixel auto-detection."""
    if plugin_id:
        if isinstance(plugin_modes, dict) and plugin_id in plugin_modes:
            configured = _ALIASES.get(plugin_modes[plugin_id], plugin_modes[plugin_id])
            if configured in RENDERINGS:
                return configured
        if plugin_id in PHOTO_PLUGINS:
            return photo_mode
        if plugin_id in HYBRID_PLUGINS:
            if isinstance(settings, dict) and settings.get("backgroundOption") == "color":
                return GRAPHICS
            return HYBRID
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

    same_x = np.all(arr[:, 1:] == arr[:, :-1], axis=-1).mean()
    same_y = np.all(arr[1:, :] == arr[:-1, :], axis=-1).mean()
    flat_ratio = float(0.5 * (same_x + same_y))

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
    fallback = default_mode(
        plugin_id,
        image=image,
        photo_mode=photo_mode,
        plugin_modes=plugin_modes,
        settings=settings,
    )

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


def _resolve_lut_params(image_settings=None, settings=None, resolution=None):
    """Resolve OKLab 3D LUT & sharpening parameters from panel defaults + device.json overrides."""
    profile = _panel_profile(resolution)
    merged = {}
    if isinstance(image_settings, dict):
        merged.update(image_settings)
    if isinstance(settings, dict):
        merged.update(settings)

    raw_clip = merged.get("highlight_clip", HIGHLIGHT_CLIP)
    if raw_clip in (False, "none", "None"):
        highlight_clip = None
    elif raw_clip is None:
        highlight_clip = HIGHLIGHT_CLIP
    else:
        highlight_clip = int(round(_number(raw_clip, HIGHLIGHT_CLIP, 200, 255)))

    return {
        "shadow_lift": _number(merged.get("shadow_lift"), PHOTO_SHADOW_LIFT, 0.0, 0.50),
        "mid_lift": _number(merged.get("mid_lift"), PHOTO_MID_LIFT, 0.0, 0.30),
        "lift_knee": _number(merged.get("lift_knee"), PHOTO_LIFT_KNEE, 0.20, 1.00),
        "blue_darken": _number(merged.get("blue_darken"), PHOTO_BLUE_DARKEN, 0.0, 0.50),
        "green_darken": _number(merged.get("green_darken"), PHOTO_GREEN_DARKEN, 0.0, 0.50),
        "vivid_gamma": _number(merged.get("vivid_gamma"), VIVID_GAMMA, 0.50, 1.50),
        "photo_saturation": _number(merged.get("photo_saturation"), PHOTO_SATURATION, 0.50, 1.80),
        "vivid_saturation": _number(merged.get("vivid_saturation"), VIVID_SATURATION, 0.50, 2.00),
        "gamut_compress": _number(merged.get("gamut_compress"), GAMUT_COMPRESS_STRENGTH, 0.0, 1.50),
        "highlight_clip": highlight_clip,
        "denoise_strength": _number(merged.get("denoise_strength"), PHOTO_DENOISE_STRENGTH, 0.0, 1.0),
        "sharpen_radius": _number(merged.get("sharpen_radius"), profile["sharpen_radius"], 0.0, 3.0),
        "sharpen_percent": int(round(_number(merged.get("sharpen_percent"), profile["sharpen_percent"], 0, 150))),
        "sharpen_threshold": int(round(_number(merged.get("sharpen_threshold"), profile["sharpen_threshold"], 0, 32))),
    }


def mode_saturation(mode, image_settings=None):
    """Ink saturation for a specific rendering mode, ignoring global inky_saturation."""
    default = MODE_INK_SATURATION.get(mode, MODE_INK_SATURATION[PHOTO])
    if isinstance(image_settings, dict):
        overrides = image_settings.get("mode_saturation")
        if isinstance(overrides, dict) and mode in overrides:
            return _number(overrides[mode], default, 0.0, 1.0)
    return default


def colour_plan(plugin_id=None, settings=None, image_settings=None, image=None, resolution=None):
    """How to render an instance: algorithm, image enhancement, LUT params, and ink saturation."""
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
    res = tuple(resolution) if resolution else (image.size if image is not None else None)
    lut_params = _resolve_lut_params(image_settings=image_settings, settings=settings, resolution=res)

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
    return {
        "mode": mode,
        "enhance": image_settings,
        "saturation": saturation,
        "manual": manual,
        "resolution": res,
        "lut_params": lut_params,
    }


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
    """Low-RAM integer edge-preserving denoise: smooths grain in flat areas while keeping edges sharp."""
    if strength <= 0.0:
        return image
    blurred = image.filter(ImageFilter.BoxBlur(1))
    orig = np.asarray(image, dtype=np.int16)
    blur = np.asarray(blurred, dtype=np.int16)
    diff = np.abs(orig - blur).max(axis=-1, keepdims=True)
    max_w = int(round(strength * 256.0))
    denom = max(1, int(edge_threshold))
    weight = np.clip(((denom - (diff - 5)) * max_w) // denom, 0, max_w)
    out = orig + ((weight * (blur - orig)) >> 8)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def _sharpen_photo_details(
    image,
    radius=PHOTO_SHARPEN_RADIUS,
    percent=PHOTO_SHARPEN_PERCENT,
    threshold=PHOTO_SHARPEN_THRESHOLD,
):
    """Edge-aware UnsharpMask: restores fine detail without re-amplifying flat-area grain."""
    if percent <= 0 or radius <= 0.0:
        return image
    return image.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=int(percent), threshold=int(threshold))
    )


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


def _compute_spectra6_lut_table(mode=PHOTO, size=LUT_SIZE, params=None):
    """Compute the (size^3, 3) float32 OKLab + sub-palette alignment LUT table."""
    p = params or _resolve_lut_params()
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
        L = np.clip(L ** p["vivid_gamma"], 0.0, 1.0)
        base_sat = p["vivid_saturation"]
    else:
        shadow_lift = p["shadow_lift"]
        mid_lift = p["mid_lift"]
        lift_knee = p["lift_knee"]
        if shadow_lift > 0.0 or mid_lift > 0.0:
            t = np.clip(L / lift_knee, 0.0, 1.0)
            w = (1.0 - t) ** 2 * (1.0 + 2.0 * t)
            L = np.clip(
                L
                + shadow_lift * (np.maximum(L, 1e-4) ** 0.75) * w
                + mid_lift * np.sin(np.pi * t) * (1.0 - t),
                0.0,
                1.0,
            )
        base_sat = p["photo_saturation"]

    # Skin-tone chroma protection
    skin_hue_w = np.exp(-0.5 * ((hue_deg - 45.0) / 22.0) ** 2)
    skin_chroma_w = np.clip(1.0 - np.maximum(0.0, C - 0.14) / 0.08, 0.0, 1.0)
    skin_mask = skin_hue_w * skin_chroma_w
    sat_scale = 1.0 + (base_sat - 1.0) * (1.0 - 0.65 * skin_mask)
    C = C * sat_scale

    # Soft-knee OKLab chroma compression
    gamut_compress = p["gamut_compress"]
    if gamut_compress > 0.0:
        knee_c = 0.16
        excess = np.maximum(0.0, C - knee_c)
        C = np.where(C > knee_c, knee_c + excess / (1.0 + gamut_compress * (excess / 0.16)), C)

    a = C * np.cos(hue)
    b_ok = C * np.sin(hue)

    r_out, g_out, b_out = _oklab_to_linear_srgb(L, a, b_ok)
    r = _linear_to_srgb(r_out)
    g = _linear_to_srgb(g_out)
    b = _linear_to_srgb(b_out)

    if mode in (PHOTO, HYBRID):
        max_rg = np.maximum(r, g)
        blue_dom = np.clip((b - max_rg) / 0.25, 0.0, 1.0)
        blue_darken = p["blue_darken"]
        if blue_darken > 0.0:
            blue_luma = r * 0.299 + g * 0.587 + b * 0.114
            blue_weight = blue_dom * np.clip(1.15 - 0.75 * blue_luma, 0.20, 1.0)
            darken = 1.0 - blue_darken * blue_weight
            r *= darken
            g *= darken
            b *= darken
        blue_align = np.clip((b - np.maximum(r, g)) / 0.12, 0.0, 1.0)
        rg_mid = 0.5 * (r + g)
        r = r + blue_align * (rg_mid - r)
        g = g + blue_align * (rg_mid - g)

        max_rb = np.maximum(r, b)
        green_dom = np.clip((g - max_rb) / 0.12, 0.0, 1.0)
        green_darken = p["green_darken"]
        g_adj = g * (1.0 - green_darken * green_dom) if green_darken > 0.0 else g
        pastel = np.clip((g_adj - 0.68) / 0.32, 0.0, 1.0)
        b_clean = g_adj * (0.18 + 0.55 * pastel)
        warm_offset = np.clip(r - b, 0.0, 0.25 * g_adj)
        r_clean = np.minimum(b_clean + warm_offset, 0.85 * g_adj)
        r = r + green_dom * (r_clean - r)
        g = g + green_dom * (g_adj - g)
        b = b + green_dom * (b_clean - b)

        warm_mask = np.clip((r - np.maximum(g, b)) / 0.12, 0.0, 1.0) * np.clip((g - b + 0.04) / 0.08, 0.0, 1.0)
        b_warm = np.minimum(b, g * 0.92)
        b = b + warm_mask * (b_warm - b)

    r = np.clip(r * WHITE_POINT_MULTS[0], 0.0, 1.0)
    g = np.clip(g * WHITE_POINT_MULTS[1], 0.0, 1.0)
    b = np.clip(b * WHITE_POINT_MULTS[2], 0.0, 1.0)

    highlight_clip = p["highlight_clip"]
    if highlight_clip is not None:
        clip_thresh = highlight_clip / 255.0
        near_white = np.minimum(np.minimum(r, g), b) > clip_thresh
        r = np.where(near_white, 1.0, r)
        g = np.where(near_white, 1.0, g)
        b = np.where(near_white, 1.0, b)

    return np.stack([r, g, b], axis=-1).reshape(-1, 3).astype(np.float32)


def _lut_cache_path(mode, size, params):
    """Parameter-hashed .npy path so any constant or device.json tweak automatically invalidates."""
    p = params or _resolve_lut_params()
    sig = (
        f"{mode}|{size}|{p['shadow_lift']:.4f}|{p['mid_lift']:.4f}|{p['lift_knee']:.4f}|"
        f"{p['blue_darken']:.4f}|{p['green_darken']:.4f}|{p['vivid_gamma']:.4f}|"
        f"{p['photo_saturation']:.4f}|{p['vivid_saturation']:.4f}|{p['gamut_compress']:.4f}|"
        f"{WHITE_POINT_MULTS}|{p['highlight_clip']}"
    )
    digest = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"inkypi_spectra6_lut_{mode}_{digest}.npy")


def _prune_lut_disk_cache(keep_paths=None, max_files=_MAX_DISK_LUT_CACHE):
    """Prune stale /tmp/inkypi_spectra6_lut_*.npy files (LRU) and orphaned temp files."""
    tmp_dir = tempfile.gettempdir()
    protected = set(_LUT_CACHE.keys())
    if keep_paths:
        protected.update(keep_paths)

    now = time.time()
    entries = []

    for path in glob.glob(os.path.join(tmp_dir, "inkypi_spectra6_lut_*.npy")):
        try:
            st = os.stat(path)
        except OSError:
            continue

        if ".tmp." in os.path.basename(path):
            if (now - st.st_mtime) > 60.0:
                try:
                    os.remove(path)
                except OSError:
                    pass
            continue

        entries.append((st.st_mtime, path))

    entries.sort(key=lambda item: item[0], reverse=True)
    for _, path in entries[max_files:]:
        if path in protected:
            continue
        try:
            os.remove(path)
            logger.debug(f"Pruned stale 3D LUT cache file: {path}")
        except OSError:
            pass


def _build_spectra6_3dlut(mode=PHOTO, size=LUT_SIZE, params=None, cache_path=None):
    """Load or compile a 33x33x33 perceptual OKLab 3D LUT for Pillow."""
    p = params or _resolve_lut_params()
    cache_path = cache_path or _lut_cache_path(mode, size, p)
    expected_shape = (size ** 3, 3)
    table = None

    if os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path)
            if loaded.shape == expected_shape and loaded.dtype == np.float32:
                table = loaded
                try:
                    os.utime(cache_path, None)
                except OSError:
                    pass
                logger.info(f"Loaded cached {size}^3 OKLab Color3DLUT from {cache_path}")
        except Exception as e:
            logger.debug(f"Could not read cached 3D LUT {cache_path}: {e}")

    if table is None:
        table = _compute_spectra6_lut_table(mode=mode, size=size, params=p)
        try:
            tmp_path = f"{cache_path}.{os.getpid()}.tmp.npy"
            np.save(tmp_path, table)
            os.replace(tmp_path, cache_path)
            logger.info(f"Compiled and cached {size}^3 OKLab Color3DLUT to {cache_path}")
        except Exception as e:
            logger.debug(f"Could not persist 3D LUT cache to {cache_path}: {e}")
        _prune_lut_disk_cache(keep_paths={cache_path})

    return ImageFilter.Color3DLUT(size, table.ravel().tolist(), channels=3)


def get_spectra6_3dlut(mode=PHOTO, params=None):
    """Return a cached Color3DLUT filter for the given rendering mode and LUT parameters."""
    lut_key = PHOTO if mode == HYBRID else mode
    p = params or _resolve_lut_params()
    cache_path = _lut_cache_path(lut_key, LUT_SIZE, p)
    lut = _LUT_CACHE.get(cache_path)
    if lut is None:
        lut = _build_spectra6_3dlut(mode=lut_key, size=LUT_SIZE, params=p, cache_path=cache_path)
        if len(_LUT_CACHE) >= _MAX_LUT_CACHE:
            _LUT_CACHE.pop(next(iter(_LUT_CACHE)))
        _LUT_CACHE[cache_path] = lut
    return lut


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


def _lock_hybrid_ui(indices, orig_pixels, palette):
    """Protect crisp typography, zero-blur text halos, and pure-primary UI accents over dithered photos."""
    pure = np.array(palette["pure"], dtype=np.int32)
    neutral_inks = set(_neutral_inks(palette))
    out = indices.copy()

    h, _w, _ = orig_pixels.shape
    for y0 in range(0, h, 120):
        band = orig_pixels[y0:y0 + 120].astype(np.int32)
        diff = band[:, :, None, :] - pure[None, None, :, :]
        dist_sq = (diff * diff).sum(axis=-1)
        nearest = dist_sq.argmin(axis=-1)
        min_dist = dist_sq.min(axis=-1)

        is_neutral_ink = np.isin(nearest, list(neutral_inks))
        thresh_sq = np.where(is_neutral_ink, 22 * 22, 28 * 28)
        lock_mask = min_dist <= thresh_sq
        out_band = out[y0:y0 + 120]
        out_band[lock_mask] = nearest[lock_mask].astype(np.uint8)

    return _sharpen_edges(
        out,
        orig_pixels,
        orig_pixels.astype(np.float32),
        palette,
        list(neutral_inks),
        edge_range=64,
        text_threshold=168,
        neutral_chroma=28,
    )


def render_indices(image, palette, mode, saturation=None, plan=None, resolution=None):
    """Palette indices (one per pixel) as the display will show them."""
    if saturation is None:
        saturation = MODE_INK_SATURATION.get(mode, MODE_INK_SATURATION[PHOTO])

    res = (
        resolution
        or (plan.get("resolution") if isinstance(plan, dict) else None)
        or image.size
    )

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

    params = (
        plan.get("lut_params")
        if isinstance(plan, dict) and "lut_params" in plan
        else _resolve_lut_params(resolution=res)
    )

    image = image.convert("RGB")
    orig_pixels = np.asarray(image, dtype=np.uint8) if mode == HYBRID else None

    if mode == PHOTO:
        image = _smooth_flat_areas(image, strength=params["denoise_strength"])
        if PHOTO_CONTRAST != 1.0:
            image = ImageEnhance.Contrast(image).enhance(PHOTO_CONTRAST)

    if mode in (PHOTO, VIVID):
        image = _sharpen_photo_details(
            image,
            radius=params["sharpen_radius"],
            percent=params["sharpen_percent"],
            threshold=params["sharpen_threshold"],
        )

    image = image.filter(get_spectra6_3dlut(mode=mode, params=params))

    pure = np.array(palette["pure"], dtype=np.float32)
    perceived = perceived_palette(palette, resolution=res)
    quant = perceived * saturation + pure * (1.0 - saturation)

    flat_pal = [int(round(v)) for v in np.asarray(quant).flatten()]
    flat_pal += flat_pal[:3] * (256 - len(quant))
    pal_image = Image.new("P", (1, 1))
    pal_image.putpalette(flat_pal)

    quantized = image.quantize(len(quant), palette=pal_image, dither=Image.Dither.FLOYDSTEINBERG)
    indices = np.asarray(quantized, dtype=np.uint8)

    if mode == HYBRID and orig_pixels is not None:
        indices = _lock_hybrid_ui(indices, orig_pixels, palette)

    return indices


def to_pure_palette_image(indices, palette):
    """Palette ("P") image in the display's pure colours: the driver sends it without dithering."""
    image = Image.fromarray(indices.astype(np.uint8), "P")
    image.putpalette([int(v) for colour in palette["pure"] for v in colour])
    return image


def perceived_palette(palette, resolution=None):
    """The inks as the eye sees them: panel-calibrated paper-normalized Spectra 6 targets."""
    if len(palette.get("measured", [])) == len(CALIBRATED_PERCEIVED):
        return _panel_profile(resolution)["perceived"].copy()
    measured = np.array(palette["measured"], dtype=np.float32)
    white = measured[_white_index(palette)]
    return np.clip(measured / np.maximum(white, 1) * 255.0, 0, 255)


def simulate(image, palette, mode, saturation=None, resolution=None):
    """What the panel will look like (calibrated ink colours, normalised to the paper white)."""
    res = resolution or image.size
    indices = render_indices(image, palette, mode, saturation=saturation, resolution=res)
    return Image.fromarray(perceived_palette(palette, resolution=res).astype(np.uint8)[indices], "RGB")
