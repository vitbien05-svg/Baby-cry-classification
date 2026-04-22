#!/usr/bin/env python3
"""
Trích xuất MFCC + Delta (bậc 1) + Delta-Delta (bậc 2) từ file âm thanh WAV.
Delta: đạo hàm bậc 1 (velocity). Delta-Delta: đạo hàm bậc 2 (acceleration).
"""

# ============ IMPORT THƯ VIỆN CẦN THIẾT ============
from pathlib import Path
import numpy as np
import librosa

# ============ CONFIG CẤU HÌNH ============
CONFIG = {
    "method_name": "mfcc_delta_2",
    # Đường dẫn nguồn audio (WAV)
    "input_root": Path(__file__).resolve().parent.parent
    / "data"
    / "corpus"
    / "donatecry_augumentation",
    # Thư mục output: extract/data_extract/mfcc_delta_2/
    "output_root": Path(__file__).resolve().parent
    / "data_extract"
    / "mfcc_delta_2_mel_64",
    # Tham số MFCC
    "sample_rate": 16000,
    "n_mfcc": 13,
    "n_fft": 512,
    "hop_length": 256,
    "n_mels": 64,
    "fmin": 0,
    "fmax": 8000,  # None = sr/2
    # Tham số Delta
    "width": 9,  # Số frame mỗi bên để tính delta (mặc định 9)
    # Định dạng lưu
    "save_format": "npy",
}


# ============ CODE ============
def extract_mfcc_delta_delta(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Trích xuất MFCC + Delta + Delta-Delta từ tín hiệu âm thanh."""
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=cfg["n_mfcc"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        fmin=cfg["fmin"],
        fmax=cfg["fmax"],
    )
    width = cfg.get("width", 9)
    delta = librosa.feature.delta(mfcc, width=width, order=1)
    delta_delta = librosa.feature.delta(mfcc, width=width, order=2)
    # Ghép MFCC + Delta + Delta-Delta: (3*n_mfcc, n_frames)
    features = np.concatenate([mfcc, delta, delta_delta], axis=0)
    return features


def save_config_txt(cfg: dict, output_path: Path) -> None:
    """Ghi tham số config ra file txt."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Tham số cấu hình MFCC + Delta + Delta-Delta\n")
        f.write("=" * 50 + "\n")
        for key, value in cfg.items():
            f.write(f"{key}: {value}\n")
        f.write("=" * 50 + "\n")


def get_next_run_number(method_root: Path) -> int:
    """Tìm số thứ tự chạy tiếp theo (run_1, run_2, ...)."""
    if not method_root.exists():
        return 1
    max_n = 0
    for d in method_root.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            try:
                n = int(d.name.split("_")[1])
                max_n = max(max_n, n)
            except (IndexError, ValueError):
                pass
    return max_n + 1


def main():
    cfg = CONFIG
    method_root = cfg["output_root"]
    method_root.mkdir(parents=True, exist_ok=True)

    run_num = get_next_run_number(method_root)
    run_dir = method_root / f"run_{run_num}"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Ghi file config (thêm run_number)
    config_txt = run_dir / "params.txt"
    cfg_to_save = {**cfg, "run_number": run_num}
    save_config_txt(cfg_to_save, config_txt)
    print(f"Lần chạy: run_{run_num}")
    print(f"Đã ghi config: {config_txt}")

    input_root = cfg["input_root"]
    if not input_root.exists():
        print(f"⚠ Không tìm thấy thư mục nguồn: {input_root}")
        print(
            "  Vui lòng chạy organize_and_convert_to_wav.py trước hoặc cập nhật input_root."
        )
        return 1

    total = 0
    for wav_path in input_root.rglob("*.wav"):
        rel = wav_path.relative_to(input_root)
        out_sub = data_dir / rel.parent
        out_sub.mkdir(parents=True, exist_ok=True)
        out_name = rel.stem + f".{cfg['save_format']}"
        out_path = out_sub / out_name

        try:
            y, sr = librosa.load(str(wav_path), sr=cfg["sample_rate"], mono=True)
            features = extract_mfcc_delta_delta(y, sr, cfg)
            np.save(str(out_path), features)
            total += 1
        except Exception as e:
            print(f"  ❌ Lỗi {wav_path.name}: {e}")

    print(f"\n✅ Hoàn tất. Đã trích xuất MFCC+Delta+Delta-Delta cho {total} file.")
    print(f"   Output: {run_dir}")
    print(f"   - params.txt: tham số config")
    print(f"   - data/: các file đã extract")
    return 0


if __name__ == "__main__":
    exit(main())
