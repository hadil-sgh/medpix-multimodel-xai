"""Box 3 - Route decision: short path (cheap) vs long path (full evidence gathering)."""
from . import config as C


def route(fp, threshold=C.ROUTE_CONF_THRESHOLD):
    """Short path only when the first-pass model is confident AND image/text models agree."""
    reasons = []
    if fp.p_location < threshold:
        reasons.append(f"low location confidence ({fp.p_location:.2f} < {threshold:.2f})")
    if not fp.agree:
        reasons.append(f"image says {fp.image_location}, history says {fp.text_location}")
    return ("long" if reasons else "short"), (reasons or ["confident and image/text agree"])
