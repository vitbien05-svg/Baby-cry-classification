#!/usr/bin/env python3
"""
Trích xuất LPCC (Linear Prediction Cepstral Coefficients)
từ file WAV và lưu feature cho Machine Learning
"""

# =========================
# IMPORT
# =========================
import numpy as np
import librosa
from pathlib import Path

# =========================
# CONFIG
# =========================
CONFIG = {

    "method_name": "lpcc",

    # INPUT AUDIO
    "input_root": Path(__file__).resolve().parent.parent / "data" / "donatecry_copus_split_data" / "test",

    # OUTPUT
    "output_root": Path(__file__).resolve().parent / "data_extract" / "LPCC_test",

    # AUDIO
    "sample_rate": 16000,

    # FRAME
    "frame_ms": 25,
    "hop_ms": 10,

    # LPC
    "lpc_order": 20,
    "num_lpcc": 20,

    # SAVE
    "save_format": "npy"
}

# =========================
# LPC → LPCC
# =========================
def lpc_to_lpcc(lpc, num_lpcc):

    lpcc = np.zeros(num_lpcc)

    lpcc[0] = np.log(lpc[0]**2 + 1e-8)

    for n in range(1, num_lpcc):

        acc = 0.0

        for k in range(1, n):

            acc += (k / n) * lpcc[k] * lpc[n-k]

        lpcc[n] = lpc[n] + acc

    return lpcc


# =========================
# EXTRACT LPCC
# =========================
def extract_lpcc(y, sr, cfg):

    frame_len = int(sr * cfg["frame_ms"] / 1000)
    hop_len   = int(sr * cfg["hop_ms"] / 1000)

    if len(y) < frame_len:
        return None

    frames = librosa.util.frame(
        y,
        frame_length=frame_len,
        hop_length=hop_len
    ).T

    frames = frames * np.hamming(frame_len)

    lpcc_frames = []

    for frame in frames:

        try:

            lpc = librosa.lpc(frame, order=cfg["lpc_order"])

            if np.any(np.isnan(lpc)) or np.any(np.isinf(lpc)):
                continue

            lpcc = lpc_to_lpcc(lpc, cfg["num_lpcc"])

            if np.any(np.isnan(lpcc)) or np.any(np.isinf(lpcc)):
                continue

            lpcc_frames.append(lpcc)

        except:
            continue

    if len(lpcc_frames) == 0:
        return None

    lpcc_frames = np.array(lpcc_frames)

    # pooling → fixed feature
    feature = np.concatenate([

        np.mean(lpcc_frames, axis=0),
        np.std(lpcc_frames, axis=0)

    ])

    return feature


# =========================
# SAVE CONFIG
# =========================
def save_config_txt(cfg, path):

    with open(path, "w", encoding="utf-8") as f:

        f.write("# LPCC parameters\n")
        f.write("="*40 + "\n")

        for k,v in cfg.items():
            f.write(f"{k}: {v}\n")


# =========================
# RUN NUMBER
# =========================
def get_next_run_number(method_root):

    if not method_root.exists():
        return 1

    max_n = 0

    for d in method_root.iterdir():

        if d.is_dir() and d.name.startswith("run_"):

            try:
                n = int(d.name.split("_")[1])
                max_n = max(max_n, n)

            except:
                pass

    return max_n + 1


# =========================
# MAIN
# =========================
def main():

    cfg = CONFIG

    method_root = cfg["output_root"]
    method_root.mkdir(parents=True, exist_ok=True)

    run_num = get_next_run_number(method_root)

    run_dir = method_root / f"run_{run_num}"
    data_dir = run_dir / "data"

    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: run_{run_num}")

    # save config
    config_file = run_dir / "params.txt"
    save_config_txt(cfg, config_file)

    print(f"Config saved: {config_file}")

    input_root = cfg["input_root"]

    total = 0

    for wav_path in input_root.rglob("*.wav"):

        rel = wav_path.relative_to(input_root)

        out_sub = data_dir / rel.parent
        out_sub.mkdir(parents=True, exist_ok=True)

        out_file = out_sub / (rel.stem + "." + cfg["save_format"])

        try:

            y, sr = librosa.load(
                str(wav_path),
                sr=cfg["sample_rate"],
                mono=True
            )

            feat = extract_lpcc(y, sr, cfg)

            if feat is None:
                continue

            np.save(out_file, feat)

            total += 1

        except Exception as e:

            print("Error:", wav_path.name, e)

    print("\n✅ LPCC extraction completed")
    print("Files processed:", total)
    print("Output folder:", run_dir)


# =========================
# RUN
# =========================
if __name__ == "__main__":

    main()