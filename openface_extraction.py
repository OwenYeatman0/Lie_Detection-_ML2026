import os
from pathlib import Path

try:
    PROJECT = Path(__file__).resolve().parent
except NameError:
    PROJECT = Path.cwd()
os.chdir(PROJECT)


import cv2
import numpy as np
import pandas as pd
import torch
from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor



VIDEO_DIR = PROJECT / "videos"
OUT_DIR = PROJECT / "features" / "frames"
SAMPLE_FPS = 5
DEVICE = "cpu"
EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}



EMOTIONS = ["neutral", "happy", "sad", "suprise", "fear", "disgust", "anger", "contempt"]


def process_frame(frame, idx, fps, detector, mtl, tmp_path):
    row = {"frame" : idx, "time_s": idx / fps, "face_found": 0}
    cv2.imwrite(tmp_path, frame)
    try:
        face, dets = detector.get_face(tmp_path)
    except Exception as e:
        print(f"    frame {idx}: detector error: {e}")
        face, dets = None, None
    if face is None or dets is None or len(dets) == 0:
        return row
    
    d = dets[int(np.argmax(dets[:, 4]))]
    x1, y1, x2, y2, conf = [float(v) for v in d[:5]]
    with torch.no_grad():
        emo, gaze, aus = mtl.predict(face)
        
    probs = torch.softmax(emo, dim=1)[0].tolist()
    gaze = gaze.reshape(-1).tolist()
    aus = aus.reshape(-1).tolist()
     
    row.update({"face_found": 1, "det_conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    for i, p in enumerate(probs):
        row[f"e,o_p{i}"] = p
        row["gaze_yaw"], row["gaze_pitch"] = gaze[0], gaze[1]
    for i, a in enumerate(aus):
        row[f"au_{i}"] = a
    return row
        
    




def main():
    videos = sorted(p for p in VIDEO_DIR.rglob("*") if p.suffix.lower() in EXTS)
    if not videos:
        print(f"No videos found in {VIDEO_DIR}. Expected videos/<subject>/<label>_*.mp4")
        return OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    detector = FaceDetector(model_path ="./weights/Alignment_RetinaFace.pth", device=DEVICE)
    mtl = MultitaskPredictor(model_path = "./weights//MTL_backbone.pth")
    tmp_path = str(PROJECT / "_tmp_frame.jpg")
    
    for n, vp in enumerate(videos, 1):
        subject = vp.parent.name
        out = OUT_DIR / f"{subject}__{vp.stem}.csv"
        if out.exists():
            print(f"[{n}/{len(videos)}] skip (done): {vp.name}")
            continue
        print(f"[{n}/{len(videos)}] skip (done): {vp.name}")
        
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(round(fps / SAMPLE_FPS)), 1)
        rows, idx = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rows.append(process_frame(frame, idx, fps, detector, mtl, tmp_path))
                idx += 1
            cap.release()
            
            pd.DataFrame(rows).to_csv(out, index=False)
            found = sum(r["face_found"] for r in rows)
            print(f"{len(rows)} frames sampled, face found in {found}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print("Done. Next: run biuild_dataset.py")
            
if __name__ == "__main__":
    main()
    