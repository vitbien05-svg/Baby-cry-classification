#!/usr/bin/env python3
"""
Trích xuất LFCC (Linear-frequency cepstral coefficients) từ file âm thanh WAV.
LFCC dùng linear filterbank thay vì mel filterbank như MFCC.
"""

# ============ IMPORT THƯ VIỆN CẦN THIẾT ============
from pathlib import Path
import numpy as np
import librosa
from scipy.fftpack import dct

# ============ CONFIG CẤU HÌNH ============
CONFIG = {
    "method_name": "lfcc",
    # Đường dẫn nguồn audio (WAV)
    "input_root": Path(__file__).resolve().parent.parent
    / "data"
    / "corpus"
    / "donatecry_augumentation",
    # Thư mục output: extract/data_extract/lfcc/
    "output_root": Path(__file__).resolve().parent / "data_extract" / "lfcc_mel_64",
    # Tham số LFCC
    "sample_rate": 16000,
    "n_lfcc": 13,
    "n_fft": 512,
    "hop_length": 256,
    "n_filters": 64,
    "fmin": 0,
    "fmax": None,  # None = sr/2
    # Định dạng lưu
    "save_format": "npy",  # "npy" hoặc "npz"
}


# ============ CODE ============
def _linear_filterbank(
    sr: int, n_fft: int, n_filters: int, fmin: float, fmax: float
) -> np.ndarray:
    """Tạo linear filterbank (tam giác, khoảng cách tuyến tính theo Hz)."""
    if fmax is None:
        fmax = sr / 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low = np.linspace(fmin, fmax, n_filters + 2)
    bank = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        left, center, right = low[i], low[i + 1], low[i + 2]
        for j, f in enumerate(freqs):
            if f <= left or f >= right:
                continue
            if f < center:
                bank[i, j] = (f - left) / (center - left)
            else:
                bank[i, j] = (right - f) / (right - center)
    return bank


def extract_lfcc(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Trích xuất LFCC từ tín hiệu âm thanh."""
    S = (
        np.abs(
            librosa.stft(
                y,
                n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"],
            )
        )
        ** 2
    )
    bank = _linear_filterbank(
        sr, cfg["n_fft"], cfg["n_filters"], cfg["fmin"], cfg.get("fmax") or sr / 2
    )
    log_filter = np.log(np.dot(bank, S) + 1e-10)
    lfcc = dct(log_filter, type=2, axis=0, norm="ortho")[: cfg["n_lfcc"], :]
    return lfcc


def save_config_txt(cfg: dict, output_path: Path) -> None:
    """Ghi tham số config ra file txt."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Tham số cấu hình LFCC\n")
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
    method = cfg["method_name"]
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
        # Giữ cấu trúc thư mục: class/file.npy
        out_sub = data_dir / rel.parent
        out_sub.mkdir(parents=True, exist_ok=True)
        out_name = rel.stem + f".{cfg['save_format']}"
        out_path = out_sub / out_name

        try:
            y, sr = librosa.load(str(wav_path), sr=cfg["sample_rate"], mono=True)
            lfcc = extract_lfcc(y, sr, cfg)
            np.save(str(out_path), lfcc)
            total += 1
        except Exception as e:
            print(f"  ❌ Lỗi {wav_path.name}: {e}")

    print(f"\n✅ Hoàn tất. Đã trích xuất LFCC cho {total} file.")
    print(f"   Output: {run_dir}")
    print(f"   - params.txt: tham số config")
    print(f"   - data/: các file đã extract")
    return 0


if __name__ == "__main__":
    exit(main())
