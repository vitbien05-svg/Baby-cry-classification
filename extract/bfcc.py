#!/usr/bin/env python3
"""
Trích xuất BFCC (Bark-frequency cepstral coefficients) từ file âm thanh WAV.
BFCC dùng Bark filterbank theo thang critical band của thính giác người.
"""

# ============ IMPORT THƯ VIỆN CẦN THIẾT ============
from pathlib import Path
import numpy as np
import librosa
from scipy.fftpack import dct

# ============ CONFIG CẤU HÌNH ============
CONFIG = {
    "method_name": "bfcc",
    # Đường dẫn nguồn audio (WAV)
    "input_root": Path(__file__).resolve().parent.parent
    / "data"
    / "corpus"
    / "donatecry_augumentation",
    # Thư mục output: extract/data_extract/bfcc/
    "output_root": Path(__file__).resolve().parent / "data_extract" / "bfcc",
    # Tham số BFCC
    "sample_rate": 16000,
    "n_bfcc": 13,
    "n_fft": 512,
    "hop_length": 256,
    "win_length": 512,
    "n_filters": 40,
    "fmin": 0,
    "fmax": None,  # None = sr/2
    # Định dạng lưu
    "save_format": "npy",
}


# ============ CODE ============
def _hz_to_bark(f: np.ndarray) -> np.ndarray:
    """Chuyển tần số Hz sang thang Bark (Traunmüller)."""
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)


def _bark_filterbank(
    sr: int, n_fft: int, n_filters: int, fmin: float, fmax: float
) -> np.ndarray:
    """Tạo Bark filterbank (tam giác, khoảng cách tuyến tính theo Bark)."""
    if fmax is None:
        fmax = sr / 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bark_freqs = _hz_to_bark(freqs)
    bark_min = _hz_to_bark(np.array([fmin]))[0]
    bark_max = _hz_to_bark(np.array([fmax]))[0]
    low_bark = np.linspace(bark_min, bark_max, n_filters + 2)
    bank = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        left, center, right = low_bark[i], low_bark[i + 1], low_bark[i + 2]
        for j, b in enumerate(bark_freqs):
            if b <= left or b >= right:
                continue
            if b < center:
                bank[i, j] = (b - left) / (center - left)
            else:
                bank[i, j] = (right - b) / (right - center)
    return bank


def extract_bfcc(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Trích xuất BFCC từ tín hiệu âm thanh."""
    S = (
        np.abs(
            librosa.stft(
                y,
                n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"],
                win_length=cfg["win_length"],
            )
        )
        ** 2
    )
    bank = _bark_filterbank(
        sr, cfg["n_fft"], cfg["n_filters"], cfg["fmin"], cfg.get("fmax") or sr / 2
    )
    log_filter = np.log(np.dot(bank, S) + 1e-10)
    bfcc = dct(log_filter, type=2, axis=0, norm="ortho")[: cfg["n_bfcc"], :]
    return bfcc


def save_config_txt(cfg: dict, output_path: Path) -> None:
    """Ghi tham số config ra file txt."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Tham số cấu hình BFCC\n")
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
            bfcc = extract_bfcc(y, sr, cfg)
            np.save(str(out_path), bfcc)
            total += 1
        except Exception as e:
            print(f"  ❌ Lỗi {wav_path.name}: {e}")

    print(f"\n✅ Hoàn tất. Đã trích xuất BFCC cho {total} file.")
    print(f"   Output: {run_dir}")
    print(f"   - params.txt: tham số config")
    print(f"   - data/: các file đã extract")
    return 0


if __name__ == "__main__":
    exit(main())
