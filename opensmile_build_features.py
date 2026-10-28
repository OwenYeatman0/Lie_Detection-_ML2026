import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd





# ensure that we are in the correct project folder
try:
    PROJECT = Path(__file__).resolve().parent
except NameError:
    PROJECT = Path.cwd()
    
    
    
    
# ----- VIDEO AND WHISPER FEATURES -----
VIDEO_DIR = PROJECT / "videos"
OUT_DIR = PROJECT / "features" / "audio"
EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

WHISPER_MODEL = "base.en" # all models in english
WHISPER_LANG = "en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"


PAUSE_MIN = 0.25 # pause beeds to be > 0.25 seconds

LONG_PAUSE = 0.75 # define a long pause as .75 seconds


# Ensure whisper does not automatically remove filler words

FILLERS = {"um", "uh", "umm", "er", "erm", "hmm", "hm", "mm", "ah"}

WHISPER_PROMPT = "Umm, let me think. Uh, so, like you know, hmm, okay." # sample prompt



# Set the whisper and opensmile variables that we will be testing
TIMING_KEYS = [
    "n_words", "response_latency", "speech_span", "speech_rate", 
    "n_pauses", "mean_pause", "max_pause", "pause_ratio", "n_long_pauses",
    "n_fillers", "filler_rate", "mean_word_prob",
    ]
    
def timing_features(words):
    """
    Gets the audio features required to make inferences about lies
    
    returns:
    """
    f = {k: np.nan for k in TIMING_KEYS}
    f["n_words"] = len(words)
    if not words:
        return {"aud_t_" + k: v for k, v in f.items()}
    
    starts = np.array([w["start"] for w in words], float)
    ends = np.array([w["end"] for w in words], float)
    span = ends[-1] - starts[0]
    
    f["response_latency"] = starts[0]
    f["speech_span"] = span
    f["speech_rate"] = len(words) / span if span > 0 else np.nan
    
    # finds the gaps between words and determines pauses > 0.25
    gaps = starts[1:] - ends[:-1]
    pauses = gaps[gaps >= PAUSE_MIN]
    f["n_pauses"] = len(pauses)
    f["mean_pause"] = float(pauses.max()) if len(pauses) else 0.0 # avg pause
    f["max_pause"] = float(pauses.max()) if len(pauses) else 0.0 # longest pause
    f["pause_ratio"] = float(pauses.sum() / span) if span > 0 else np.nan 
    f["n_long_pauses"] = int((gaps >= LONG_PAUSE).sum()) # num pause > 0.75
    
    clean = [re.sub(r"[^a-z]", "", w["word"].lower()) for w in words] 
    n_fill = sum(c in FILLERS for c in clean)
    f["n_fillers"] = n_fill 
    f["filler_rate"] = n_fill / len(words) # amt of filler words per normal word
    f["mean_word_prob"] = float(np.mean([w["prob"] for w in words])) #
    return {"aud_t_ " + k: v for k, v in f.items()}


def extract_wav(ffmpeg_exe, video_path, wav_path):
    """extracts the audio from the video file
    
    returns: Nothing
    Error found if no audio track in video
    """
    cmd = [ffmpeg_exe, "-y", "-loglevel", "error", "-i", str(video_path),
           "-vn", "-ac", "1", "-ar", "16000", str(wav_path)]
    res = subprocess.run(cmd, capture_output = True, text = True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "ffmpeg failed (does the video have an audio track?)")
        
def transcribe(model, wav_path):
    """ creates a transcription of the whole video, breaking it down into segments, 
    finding mode of each word,and  ensuring that words are not missed
        
        returns: list of words 
    """
    segments, _ = model.transcribe(
        str(wav_path), language = WHISPER_LANG, word_timestamps=True,
        initial_prompt = WHISPER_PROMPT, vad_filter = False,
        condtion_on_previous_text = False)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"word" : w.word.strip(), "start": w.start, "end": w.end, "prob": w.probability()})
    return words

def main():
    try:
        import opensmile
        import imageio_ffmpeg
        from faster_whisper import WhisperModel
    except ImportError as e:
        print(f"Missing package: {e}\n Install with:   pip install opensmile faster-whisper imageio-ffmpeg")
        return
    videos = sorted(p for p in VIDEO_DIR.rglob("*") if p.suffix.lower() in EXTS)
    if not videos:
        print(f"No videos found in {VIDEO_DIR}. Expected videos/<subject>/ <label>_*.mp4")
        return(OUT_DIR / "transcripts").mkdir(parents=True, exist_ok = True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    smile = opensmile.Smile(feature_set = opensmile.FeatureSet.eFeMAPSv02,
                            feature_level = opensmile.FeatureLevel.Functionals)
    print(f"Loading Whisper model '{WHISPER_MODEL}' (first run downloads it) ...")
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type = WHISPER_COMPUTE)
    wav_path = PROJECT / "_tmp_audio.wav"
    
    for n, vp in enumerate(videos, 1):
        subject = vp.parent.name
        key = f"{subject}__{vp.stem}"
        out = OUT_DIR / f"{key}.csv"
       
        if out.exists():
            print(f"[{n}/{len(videos)}] skip (done): {vp.name}")
            continue
        print(f"[{n}/{len(videos)}] {subject}/{vp.name}")
        
        # Check that audio extraction is working
        try:
            extract_wav(ffmpeg_exe, vp, wav_path)
        except Exception as e:
            print(f"   audio extraction failed, skipping: {e}")
        
        row = {"subject": subject, "video": vp.stem}
        
        # check that openSMILE is working
        try:
            voice = smile.process_file(str(wav_path)).iloc[0]
            for name, val in voice.items():
                row["aud_v_" + name] = float(val)
        except Exception as e:
            print(f"     openSMILE failed (voice features left empty): {e}")
        
        # append to a transcript data file
        words = transcribe(model, wav_path)
        row.update(timing_features(words))
        (OUT_DIR / "transcripts" / f"{key}.txt").write_text(
            " ".join(w["word"] for w in words), encoding="utf-8")
        
        pd.DataFrame([row]).to_csv(out, index = False)
        print(f"     {len(words)} words transcribed") # total words
        
        
        if wav_path.exists():
            os.remove(wav_path)
        print("Done. Next: run build_dataset.py")
        
        
        
if __name__ == "__main__":
    main()
            

    
    