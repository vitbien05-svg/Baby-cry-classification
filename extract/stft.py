#!/usr/bin/env python3
"""
Trích xuất STFT (Short-Time Fourier Transform) spectrogram từ file âm thanh WAV.
Đầu ra là magnitude spectrogram (hoặc power/dB tùy config).
"""

# ============ IMPORT THƯ VIỆN CẦN THIẾT ============
from pathlib import Path
import numpy as np
import librosa

# ============ CONFIG CẤU HÌNH ============
CONFIG = {
    "method_name": "stft",
    # Đường dẫn nguồn audio (WAV)
    "input_root": Path(__file__).resolve().parent.parent / "data" / "corpus" / "donatecry_augumentation",
    # Thư mục output: extract/data_extract/stft/
    "output_root": Path(__file__).resolve().parent / "data_extract" / "stft",
    # Tham số STFT
    "sample_rate": 16000,
    "n_fft": 512,
    "hop_length": 256,
    "win_length": 512,
    "window": "hann",  # "hann", "hamming", "blackman", ...
    # Loại scale: "magnitude" | "power" | "db"
    "scale": "magnitude",
    # Định dạng lưu
    "save_format": "npy",
}

# ============ CODE ============
def extract_stft(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Trích xuất STFT spectrogram từ tín hiệu âm thanh."""
    S = np.abs(librosa.stft(
        y,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        win_length=cfg["win_length"],
        window=cfg.get("window", "hann"),
    ))
    scale = cfg.get("scale", "magnitude")
    if scale == "power":
        S = S ** 2
    elif scale == "db":
        S = librosa.power_to_db(S ** 2, ref=np.max)
    # magnitude: giữ nguyên |S|
    return S


def save_config_txt(cfg: dict, output_path: Path) -> None:
    """Ghi tham số config ra file txt."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Tham số cấu hình STFT\n")
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
        print("  Vui lòng chạy organize_and_convert_to_wav.py trước hoặc cập nhật input_root.")
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
            spec = extract_stft(y, sr, cfg)
            np.save(str(out_path), spec)
            total += 1
        except Exception as e:
            print(f"  ❌ Lỗi {wav_path.name}: {e}")

    print(f"\n✅ Hoàn tất. Đã trích xuất STFT cho {total} file.")
    print(f"   Output: {run_dir}")
    print(f"   - params.txt: tham số config")
    print(f"   - data/: các file đã extract")
    return 0


if __name__ == "__main__":
    exit(main())
