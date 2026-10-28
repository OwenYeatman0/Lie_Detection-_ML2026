from pathlib import Path

import numpy as np
import pandas as pd

try:
    PROJECT = Path(__file__).resolve().parent
except:
    PROJECT = Path.cwd()
    
    
FRAMES_DIR = PROJECT / "features" / "frames"
AUDIO_DIR = PROJECT / "features" / "audio"
OUT_CSV = PROJECT / "features" / "dataset.csv"
USE_BASELINE = True
USE_AUDIO = True
MIN_FACE_RATIO = 0.5



FEATS = (
    [f"emo_p{i}" for i in range(8)] +
    [f"gaze_yaw", "gaze_pitch"] +
    [f"au_{i}" for i in range(8)] +
    ["head_move", "size_change"]
    
    )
    

AUDIO_NO_BASELINE = {
    "aud_t_n_words", "aud_t_n_pauses", "aud_t_n_long_pauses",
    "aud_t_n_fillers", "aud_t_speech_span"
    }


def label_from_name(video_name):
    name = video_name.lower()
    for lab in ("truth", "lie", "baseline"):
        if name.startswith(lab):
            return lab
    return None

def load_video(csv_path):
    subject, video = csv_path.stem.split("__", 1)
    df = pd.read_csv(csv_path)
    for col in ["x1", "x2", "y1", "y2"]:
        if col not in df:
            df[col] = np.nan
    w = df["x2"] - df["x1"]
    cx, cy = (df["x1"] + df["x2"]) / 2, (df["y1"] + df["y2"]) / 2
    df["head_move"] = np.hpoy(cx.diff(), cy.diff()) / w
    df["size_change"] = w.diff().abs() / w
    
    for f in FEATS:
        if f not in df:
            df[f] = np.nan 
    return subject, video, df

def aggregate(df):
    out = {}
    t = df["time_s"].to_numpy(float)
    for f in FEATS:
        v = df[f].to_numpy(float)
        m = ~np.isnan(v)
        if m.sum() < 3:
            continue
        vv, tt = v[m], t[m]
        out[f + "_mean"] = vv.mean()
        out[f + "_std"] = vv.std()
        out[f + "_min"] = vv.min()
        out[f + "_max"] = vv.max()
        out[f + "_slope"] = np.polyfit(tt, vv, 1)[0] if np.ptp(tt) > 0 else 0.0
        out[f + "_absdiff"] = np.abs(np.diff(vv)).mean()
    return out


def load_audio():
    feats = {}
    if not USE_AUDIO or not AUDIO_DIR.exists():
        return feats
    for c in sorted(AUDIO_DIR.glob("*.csv")):
        df = pd.read_csv(c)
        if df.empty:
            continue
        row = df.iloc[0]
        cols = [k for k in df.columns if k.startswith("aud__")]
        feats[(str(row["subject"]), str(row["video"]))] = row[cols].astype(float)
    return feats


def main():
    csvs = sorted(FRAMES_DIR.glob("*.csv"))
    if not csvs:
        print(f"No frame CSVs in {FRAMES_DIR}. Run extract_features.py first")
        return
    videos = []
    for c in csvs:
        subject, video, df = load_video(c)
        lab = label_from_name(video)
        if lab is None:
            print(f"skip (name must start with truth/lie/baseline): {video}")
            continue
        videos.append({"subject": subject, "video": video, "label": lab, "df": df})
        
        audio = load_audio()
        print(f"audio features found for {len(audio)} videos" if audio else "no audio features (face only)")
        
    subjects = {v["subject"] for v in videos}
    
    face_base, audio_base = {}, {}
    for subj in subjects:
        base = [v for v in videos if v["subject"] == subj and v["label"] == "baseline"]
        if base:
            face_base[subj] = pd.concat(b["df"] for b in base)[FEATS].mean()
        a = [audio[(subj, b["video"])] for b in base if (subj, b["video"]) in audio]
        if a:
            audio_base[subj] = pd.concat(a, axis=1).mean(axis=1)
    rows = []
    for v in videos:
        if v["label"] == "baseline":
            continue
        subj = v["subject"]
        df = v["df"].copy()
        ratio = df["face_found"].mean() if len(df) else 0.0
        if ratio < MIN_FACE_RATIO:
            print(f"drop (face in only {ratio:.0%} of frames) : {subj}/{v['video']}")
            continue
        if USE_BASELINE:
            if subj not in face_base:
                print(f"drop (subject has no baseline video): {subj}/{v['video']}")
                continue
            df[FEATS] = df[FEATS] - face_base[subj]
        row = {
            "subject": subj,
            "video": v["video"],
            "label": v["label"],
            "y": 1 if v["label"] == "lie" else 0,
            "n_frames": len(df),
            "face_ratio": ratio,
            "has_audio": 0,
            }
        row.update(aggregate(df))
        
        a = audio.get((subj, v['video']))
        if a is not None:
            a = a.copy()
            if USE_BASELINE:
                norm_cols = [k for k in a.index if k not in AUDIO_NO_BASELINE]
                if subj in audio_base:
                    a[norm_cols] = a[norm_cols] - audio_base[subj][norm_cols]
                else:
                    a[norm_cols] = np.nan
                row.update(a.to_dict())
                row["has_audio"] = 1
            rows.append(row)
        data = pd.DataFrame(rows)
        OUT_CSV.parent.mkdir(parents=True, exist_ok = True)
        data.to_csv(OUT_CSV, index = False)
        print(f"Sved {OUT_CSV}")
        print(f"{len(data)} videos, {data['subject'].nunique()} subjects,"
              f"{int(data['y'].sum())} lies/ {int((1-data['y']).sum())} truths,"
              f"{int(data['has_audio'].sum())} with audio ")
        if audio and USE_BASELINE:
            missing = sorted(s for s in subjects if s not in audio_base)
            if missing:
                print(f"note: no audio baseline for subjects {missing}; their audio features were left empty")
        print("Next: run evaluate.py")
        
                 
    
    
if __name__ == "__main__":
    main()
    

    
    
