#!/usr/bin/env python3
import os
import sys
import argparse

import cv2
import torch
import numpy as np

# --- make /content/VAD a package ---
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from .preprocess import preprocess_frame
from .pairing    import pair_labels_and_values, normalize_label, normalize_value
from .utils      import load_detector_model, load_recognizer_model, CSVLogger, visualize_detections
from recognition.dataset import CHARS

def log_for_image(name, frame, det, rec, logger, conf_thresh, top_k, display):
    # 1) preprocess + detect
    img    = preprocess_frame(frame)
    device = next(det.parameters()).device
    t      = torch.from_numpy(img[:,:,::-1].transpose(2,0,1)/255.0).float().to(device)
    with torch.no_grad():
        out = det([t])[0]
    boxes  = out["boxes"].cpu().numpy().astype(int)
    scores = out["scores"].cpu().numpy()
    print(f"[DEBUG] {name}: {len(boxes)} raw boxes, scores={np.round(scores,3).tolist()}")

    # 2) filter by score, then keep top_k
    keep = np.where(scores >= conf_thresh)[0]
    boxes, scores = boxes[keep], scores[keep]
    order = np.argsort(-scores)[:top_k]
    boxes, scores = boxes[order], scores[order]
    print(f"[DEBUG] → after thresh {conf_thresh} & top_k {top_k}: {len(boxes)} boxes")

    # 3) OCR each box
    recognized = []
    for (x1,y1,x2,y2), sc in zip(boxes, scores):
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        H, W = 32, max(1, int(gray.shape[1] * 32 / gray.shape[0]))
        gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_LINEAR)
        inp  = torch.from_numpy(gray/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            outl = rec(inp)
        logits = outl[0] if isinstance(outl, tuple) else outl
        preds  = logits.argmax(dim=2).squeeze(1).cpu().numpy().tolist()

        # collapse repeats & blanks
        chars, prev = [], 0
        for p in preds:
            if p != prev and p != 0:
                chars.append(CHARS[p-1])
            prev = p
        raw = "".join(chars)
        print(f"[DEBUG]  crop {(x1,y1,x2,y2)} → raw OCR = '{raw}'")
        recognized.append((raw, ((x1+x2)/2, (y1+y2)/2), (x1,y1,x2,y2)))

    # 4) pair & normalize
    vals = pair_labels_and_values(recognized)
    for k,v in vals.items():
        if v is not None:
            vals[k] = normalize_value(str(v))
    vals["image"] = name
    logger.log(vals)

    # 5) optional display
    if display:
        vis = visualize_detections(frame, boxes, scores=scores)
        for raw,(cx,cy),_ in recognized:
            lab = normalize_label(raw)
            txt = lab if lab else normalize_value(raw)
            cv2.putText(vis, txt, (int(cx),int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        info = "  ".join(f"{k}={v}" for k,v in vals.items() if k!="image")
        cv2.putText(vis, info, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.imshow("LVAD OCR", vis)
        cv2.waitKey(1)

if __name__=="__main__":
    p = argparse.ArgumentParser(description="LVAD OCR Inference")
    p.add_argument("--detector",   required=True, help="path to detector .pth")
    p.add_argument("--recognizer", required=True, help="path to recognizer .pth")
    p.add_argument("--device",     choices=["cuda","cpu"], default="cuda")
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--csv",        default="lvad_results.csv")
    p.add_argument("--conf_thresh",type=float, default=0.6,
                   help="Only keep detections ≥ this score")
    p.add_argument("--top_k",      type=int,   default=8,
                   help="Then only keep the top K boxes per image")
    p.add_argument("--display",    action="store_true")
    args = p.parse_args()

    os.chdir(ROOT)  # ensure relative paths resolve
    det    = load_detector_model(args.detector,   device=args.device)
    rec    = load_recognizer_model(args.recognizer, device=args.device)
    logger = CSVLogger(args.csv)

    imgs = sorted(f for f in os.listdir(args.image_dir)
                  if f.lower().endswith((".png",".jpg",".jpeg")))
    print(f"Processing {len(imgs)} images from '{args.image_dir}'")

    for fn in imgs:
        frm = cv2.imread(os.path.join(args.image_dir, fn))
        if frm is None:
            print(f"Warning: couldn't read {fn}")
            continue
        log_for_image(fn, frm, det, rec, logger,
                      conf_thresh=args.conf_thresh,
                      top_k=args.top_k,
                      display=args.display)

    logger.close()
    if args.display:
        cv2.destroyAllWindows()
    print("Done. CSV saved to", args.csv)
