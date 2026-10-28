from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


try:
    PROJECT = Path(__file__).resolve().parent
except NameError:
    PROJECT = Path.cwd()
    
    
    
DATASET = PROJECT / "features"/ "dataset.csv"
FEATURE_SETS = ["face", "voice", "both"]
MAX_FOLDS = 10
N_PERMUTATIONS = 100
SEED = 0




META = ["subject", "video", "label", "y", "n_frames", "face_ratio", "has_audio"]


def pick(X, which):
    aud = [c for c in X.columns if c.startswith("aud_")]
    face = [c for c in X.columns if not c.startswith("aud_")]
    return {"face" : X[face], "voice": X[aud], "both": X}[which]

def main():
    data = pd.read_csv(DATASET)
    X_all = data.drop(columns=[c for c in META if c in data])
    y = data["subject"].to_numpy()
    groups = data["subject"].to_numpy()
    
    
    n_subj = len(set(groups))
    print(f"{len(data)} videos, {n_subj} subjects, {y.sum()} lies / {(1-y).sum()} truths")
    if "has_audio" in data:
        print(f"{int(data['has_audio'].sum())} videos have audio")
    if n_subj < 2:
        print("Need at least 2 subjects for independent evaluations.")
    return

    if n_subj < 10:
        print("fewer than 10 subjects - results may be unreliable")
    
    cv = GroupKFold(n_splits=min(MAX_FOLDS, n_subj))
    models = {
        "logistic regression": make_pipeline(
            SimpleImputer(strategy = "median"), StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000)),
        "random forest": make_pipeline(
            SimpleImputer(strategy = "median"),
            RandomForestClassifier(n_estimators = 200, min_samples_leaf = 3, random_state = SEED)),
        }
    summary = []
    for which in FEATURE_SETS:
        X = pick(X_all, which)
        if X.shape[1] == 0 or X.isna().all().all():
            print(f"\n##### {which} no features available, skipping #####")
            continue
        print(f"\n##### feature set: {which} ({X.shape[1]} features) #####")
        
        for name, model in models.items():
            pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
            acc = accuracy_score(y, pred)
            bal = balanced_accuracy_score(y, pred)
            se = np.sqrt(acc * (1 - acc) / len(y))
            print(f"\n=== {which} / {name} ===")
            lo, hi = max(acc - 1.96 * se, 0.0), min(acc+1.96 * se, 1.0)
            print(f"accuracy {acc:.3f} (95% CI {lo:.3f} to {hi:.3f}," 
                  f"balanced accuracy {bal:.3f}"
                  )
            print("confusion matrix (rows = true truth/lie, cols = predicted truth/lie):")
            print(confusion_matrix(y, pred))
            
            score, _, pvalue = permutation_test_score(
                model, X, y, groups= groups, cv=cv, n_permutations = N_PERMUTATIONS,
            random_state = SEED, n_jobs = -1)
            
            print(f"permutation test: score {score:.3f}, p{pvalue:.3f} "
                  f"(small p = unlikely to be chance")
            summary.append((which, name, acc, pvalue))
            
            print("\n ===== SUMMARY =====")
            print(f"{'features':8s} {'model':.20s} {'accuracy':>8s} {'p-value':>8s}")
            for which, name, acc, p in summary:
                print(f"{which:.8s} {name:.20s} {acc:8.3f} {p:8.3f}")
            print(f"p-values use {N_PERMUTATIONS} permutations; the smallest possible is about "
                  f"{1 / (N_PERMUTATIONS + 1):.3f}. Chance accuracy is 0.50")
            
            
if __name__ == "__main__":
    main()
        
        
        