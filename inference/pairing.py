import re
import math
import difflib

# Our four metrics
LABELS = ["SPEED", "FLOW", "POWER", "PI"]

# Precompiled regexes
NUMERIC_TOKEN = re.compile(r"\d+\.\d+|\d+")
# For cleaning OCR‐noise in labels
CONFUSE_MAP   = str.maketrans({"0":"O", "1":"I", "5":"S"})

def normalize_label(txt: str):
    """
    Correct common OCR confusions in `txt` and map to one of LABELS (or None).
    """
    t = txt.strip().upper().translate(CONFUSE_MAP)
    # Direct substring checks
    if "SPEED" in t or "RPM" in t:
        return "SPEED"
    if "FLOW" in t or "LPM" in t or "L/MIN" in t:
        return "FLOW"
    if "POWER" in t or t.endswith("W"):
        return "POWER"
    if t == "PI" or "P.I." in t or "PULSATILITY" in t:
        return "PI"
    # Fuzzy match for minor OCR typos
    match = difflib.get_close_matches(t, LABELS, n=1, cutoff=0.6)
    return match[0] if match else None

def normalize_value(txt: str):
    """
    Extract the first numeric token from `txt` (e.g. "30W" → "30", "2.0LPM" → "2.0").
    Returns '' if no number found.
    """
    m = NUMERIC_TOKEN.search(txt)
    return m.group(0) if m else ""

def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def pair_labels_and_values(recognized_items):
    """
    recognized_items: list of (raw_txt, center, box)
    Returns: { 'SPEED': str|None, 'FLOW': str|None, 'POWER':str|None, 'PI': str|None }
    """
    labels = []  # [(key, center)]
    values = []  # [(float_val, center)]
    for raw_txt, center, _ in recognized_items:
        key = normalize_label(raw_txt)
        if key:
            labels.append((key, center))
        else:
            val_str = normalize_value(raw_txt)
            if val_str:
                try:
                    values.append((float(val_str), center))
                except:
                    pass

    result = {k: None for k in LABELS}
    used_idx = set()

    # For each label, pick the nearest numeric value
    for key, lctr in labels:
        best_i, best_d = None, float("inf")
        for i, (val, vctr) in enumerate(values):
            if i in used_idx:
                continue
            d = euclidean_distance(lctr, vctr)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None:
            used_idx.add(best_i)
            # store as string to preserve formatting "3000" vs "3.0"
            result[key] = str(values[best_i][0])

    return result
