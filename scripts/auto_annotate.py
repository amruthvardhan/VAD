#!/usr/bin/env python3
import os, cv2, easyocr, math, re

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMGS_DIR    = os.path.join(BASE_DIR, "data", "imgs")
ANN_DIR     = os.path.join(BASE_DIR, "data", "ann")
OCR_CONF    = 0.3     # drop words <30% confidence
LEFT_RATIO  = 0.5     # only tokens in left half of image

# clear old annotations
if os.path.isdir(ANN_DIR):
    for f in os.listdir(ANN_DIR):
        if f.lower().endswith(".txt"):
            os.remove(os.path.join(ANN_DIR, f))
os.makedirs(ANN_DIR, exist_ok=True)
print(f">> Writing new annotations into {ANN_DIR}")

# init EasyOCR GPU reader
reader = easyocr.Reader(['en'], gpu=True)

NUMERIC_RE     = re.compile(r'^\d+(\.\d+)?$')
UNIT_SUFFIX_RE = re.compile(r'(RPM|LPM|W)$', re.IGNORECASE)

def is_label(t):
    t = t.upper().strip()
    return any(k in t for k in ("SPEED","FLOW","POWER")) or t=="PI"

def is_numeric(t):
    t = t.upper().strip()
    t = UNIT_SUFFIX_RE.sub("", t)
    return bool(NUMERIC_RE.fullmatch(t))

def euclid(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ─── MAIN ─────────────────────────────────────────────────────────────────────
for fn in sorted(os.listdir(IMGS_DIR)):
    if not fn.lower().endswith((".png",".jpg",".jpeg")):
        continue

    img_path = os.path.join(IMGS_DIR, fn)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping unreadable {fn}")
        continue

    h, w = img.shape[:2]
    x_limit = w * LEFT_RATIO

    # 1) full-image EasyOCR word detection
    ocr_results = reader.readtext(img, detail=1)

    # 2) filter & split tokens
    candidates = []
    for poly, raw_txt, conf in ocr_results:
        if conf < OCR_CONF:
            continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        cx = (x1 + x2) / 2
        if cx > x_limit:
            continue

        for token in re.split(r'\s+', raw_txt.strip().upper()):
            if not token:
                continue
            if is_label(token) or is_numeric(token):
                cy = (y1 + y2) / 2
                candidates.append((token, (cx, cy), (x1,y1,x2,y2)))

    # 3) pair labels → nearest numeric
    labels = [(t,ctr,box) for (t,ctr,box) in candidates if is_label(t)]
    values = []
    for t,ctr,box in candidates:
        if is_numeric(t):
            num = UNIT_SUFFIX_RE.sub("", t)
            try:
                val = float(num)
                values.append((val,ctr,box))
            except:
                pass

    used, pairs = set(), []
    for lbl, lctr, lbox in labels:
        best_i, best_d = None, float("inf")
        for i,(v,vctr,vbox) in enumerate(values):
            if i in used:
                continue
            d = euclid(lctr, vctr)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None:
            used.add(best_i)
            pairs.append((lbl, lbox, str(values[best_i][0]), values[best_i][2]))

    # 4) write exactly 8 lines
    out_path = os.path.join(ANN_DIR, os.path.splitext(fn)[0] + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for lbl,(x1,y1,x2,y2),val,(vx1,vy1,vx2,vy2) in pairs:
            f.write(f"{x1} {y1} {x2} {y2} {lbl}\n")
            f.write(f"{vx1} {vy1} {vx2} {vy2} {val}\n")
    print(f"Wrote {out_path} ({len(pairs)*2} lines)")
