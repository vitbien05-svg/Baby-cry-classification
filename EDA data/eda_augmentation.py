"""
=============================================================
EDA - So sanh Train vs Train Augmentation
Baby Cry Classification Project
=============================================================
1. Phan bo so luong file theo class: Train vs Augmentation
2. So sanh Waveform (Time Domain) truoc va sau augmentation
=============================================================
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# ======================== CONFIG ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_ROOT = os.path.join(BASE_DIR, "data", "corpus", "donatecry_copus_split_data")

TRAIN_DIR = os.path.join(SPLIT_ROOT, "train")
AUG_DIR = os.path.join(SPLIT_ROOT, "train_augumentation")
TEST_DIR = os.path.join(SPLIT_ROOT, "test")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = 16000

CLASS_COLORS = {
    "belly_pain": "#FF6B6B",
    "burping": "#4ECDC4",
    "discomfort": "#FFE66D",
    "hungry": "#95E1D3",
    "tired": "#AA96DA"
}


# ======================== TIEN ICH ========================

def count_files_per_class(root):
    """Dem so file .wav trong moi class"""
    counts = {}
    for cls in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls)
        if os.path.isdir(cls_path):
            wav_files = [f for f in os.listdir(cls_path) if f.endswith(".wav")]
            counts[cls] = len(wav_files)
    return counts


def get_aug_methods(aug_root):
    """Lay danh sach cac phuong phap augmentation da dung"""
    methods = set()
    for cls in os.listdir(aug_root):
        cls_path = os.path.join(aug_root, cls)
        if not os.path.isdir(cls_path):
            continue
        for f in os.listdir(cls_path):
            m = re.match(r'aug_\d+_(.+?)_[a-fA-F0-9]', f)
            if m:
                methods.add(m.group(1))
    return sorted(methods)


def find_original_and_augmented_pairs(train_dir, aug_dir, cls_name, n_pairs=3):
    """
    Tim n_pairs cap file: (original trong train, cac ban augmented trong aug)
    Tra ve list cua dict: { 'original': filepath, 'augmented': [(method, filepath), ...] }
    """
    train_cls = os.path.join(train_dir, cls_name)
    aug_cls = os.path.join(aug_dir, cls_name)

    original_files = sorted([f for f in os.listdir(train_cls) if f.endswith(".wav")])
    aug_files = sorted([f for f in os.listdir(aug_cls) if f.startswith("aug_") and f.endswith(".wav")])

    pairs = []
    for orig_name in original_files:
        if len(pairs) >= n_pairs:
            break

        # Tim tat ca file augmented duoc tao tu file original nay
        # Ten file aug co dang: aug_{id}_{Method}_{original_name}
        base_name = orig_name  # vd: "549a46d8-...-bp.wav"
        matching_augs = []
        for af in aug_files:
            if af.endswith(base_name):
                m = re.match(r'aug_\d+_(.+?)_' + re.escape(base_name.split('-')[0]), af)
                if m:
                    method = m.group(1)
                    matching_augs.append((method, os.path.join(aug_cls, af)))

        if matching_augs:
            pairs.append({
                "original": os.path.join(train_cls, orig_name),
                "original_name": orig_name,
                "augmented": matching_augs[:4]  # Lay toi da 4 phien ban aug
            })

    return pairs


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FFFFFF",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


# ======================== CAC HAM VE ========================

def plot_train_vs_aug_distribution(train_counts, aug_counts, test_counts, output_dir):
    """Ve phan bo so luong: Train vs Augmentation vs Test"""
    classes = sorted(train_counts.keys())

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("So sanh phan bo du lieu: Train / Train+Augmentation / Test",
                 fontsize=16, fontweight="bold")

    x = np.arange(len(classes))
    width = 0.35

    # --- Chart 1: Train vs Augmentation (grouped bar) ---
    train_vals = [train_counts.get(c, 0) for c in classes]
    aug_vals = [aug_counts.get(c, 0) for c in classes]
    colors_train = [CLASS_COLORS.get(c, "#888") for c in classes]

    bars1 = axes[0].bar(x - width/2, train_vals, width, label="Train (original)",
                        color=colors_train, edgecolor="#333", linewidth=0.8, alpha=0.7)
    bars2 = axes[0].bar(x + width/2, aug_vals, width, label="Train + Augmentation",
                        color=colors_train, edgecolor="#333", linewidth=0.8, alpha=1.0,
                        hatch="//")

    for bar, val in zip(bars1, train_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, aug_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=15)
    axes[0].set_ylabel("So luong file")
    axes[0].set_title("Train vs Train+Augmentation")
    axes[0].legend(fontsize=9)

    # --- Chart 2: Stacked bar (original vs augmented phan them) ---
    aug_added = [aug_counts.get(c, 0) - train_counts.get(c, 0) for c in classes]
    axes[1].bar(x, train_vals, label="Original", color=colors_train,
               edgecolor="#333", linewidth=0.8, alpha=0.8)
    axes[1].bar(x, aug_added, bottom=train_vals, label="Augmented (them)",
               color=colors_train, edgecolor="#333", linewidth=0.8, alpha=0.4,
               hatch="xx")

    for i, (orig, added) in enumerate(zip(train_vals, aug_added)):
        total = orig + added
        axes[1].text(i, total + 2, f"{total}\n(+{added})",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=15)
    axes[1].set_ylabel("So luong file")
    axes[1].set_title("Thanh phan: Original + Augmented")
    axes[1].legend(fontsize=9)

    # --- Chart 3: Toan bo 3 split ---
    test_vals = [test_counts.get(c, 0) for c in classes]
    w = 0.25
    axes[2].bar(x - w, train_vals, w, label="Train", color="#74B9FF",
               edgecolor="#333", linewidth=0.8)
    axes[2].bar(x, aug_vals, w, label="Train+Aug", color="#00B894",
               edgecolor="#333", linewidth=0.8)
    axes[2].bar(x + w, test_vals, w, label="Test", color="#FD79A8",
               edgecolor="#333", linewidth=0.8)

    for i in range(len(classes)):
        for vals, offset in [(train_vals, -w), (aug_vals, 0), (test_vals, w)]:
            axes[2].text(i + offset, vals[i] + 1, str(vals[i]),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(classes, rotation=15)
    axes[2].set_ylabel("So luong file")
    axes[2].set_title("Train / Train+Aug / Test")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "08_train_vs_augmentation_distribution.png")
    plt.savefig(path)
    plt.show()
    print(f"[OK] Da luu: {os.path.basename(path)}")


def plot_aug_pie(train_counts, aug_counts, output_dir):
    """Ve pie chart: ty le original vs augmented trong tung class"""
    classes = sorted(train_counts.keys())
    n = len(classes)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle("Ty le Original vs Augmented trong moi class",
                 fontsize=16, fontweight="bold")

    for i, cls in enumerate(classes):
        orig = train_counts[cls]
        aug_total = aug_counts[cls]
        added = aug_total - orig

        axes[i].pie(
            [orig, added],
            labels=["Original", "Augmented"],
            autopct=lambda p: f"{p:.0f}%\n({int(p * aug_total / 100)})",
            colors=[CLASS_COLORS.get(cls, "#888"), "#DDDDDD"],
            startangle=90, shadow=True,
            textprops={"fontsize": 9}
        )
        axes[i].set_title(f"{cls}\n(Total: {aug_total})", fontweight="bold", fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, "09_augmentation_ratio_per_class.png")
    plt.savefig(path)
    plt.show()
    print(f"[OK] Da luu: {os.path.basename(path)}")


def plot_waveform_comparison(train_dir, aug_dir, output_dir, n_classes_to_show=3):
    """
    Ve so sanh waveform truoc va sau augmentation.
    Moi class chon 1 file original, ve waveform cua no va cac ban augmented.
    """
    classes = sorted(os.listdir(train_dir))
    classes = [c for c in classes if os.path.isdir(os.path.join(train_dir, c))]

    # Chon cac class co it du lieu nhat (can aug nhieu nhat) de so sanh co y nghia
    train_counts = count_files_per_class(train_dir)
    classes_sorted = sorted(classes, key=lambda c: train_counts.get(c, 0))
    classes_to_show = classes_sorted[:n_classes_to_show]

    for cls in classes_to_show:
        pairs = find_original_and_augmented_pairs(train_dir, aug_dir, cls, n_pairs=2)

        if not pairs:
            print(f"  [SKIP] Khong tim thay cap original-augmented cho class: {cls}")
            continue

        for pair_idx, pair in enumerate(pairs):
            n_aug = len(pair["augmented"])
            n_rows = 1 + n_aug  # 1 original + n augmented

            fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.8 * n_rows))
            fig.suptitle(
                f"So sanh Waveform: Truoc va Sau Augmentation\n"
                f"Class: {cls}  |  File goc: {pair['original_name'][:50]}...",
                fontsize=14, fontweight="bold", y=1.02
            )

            # Ve original
            y_orig, sr = librosa.load(pair["original"], sr=SAMPLE_RATE, mono=True)
            ax = axes[0]
            librosa.display.waveshow(y_orig, sr=sr, ax=ax,
                                     color=CLASS_COLORS.get(cls, "#333"), alpha=0.85)
            ax.set_title(f"ORIGINAL  |  duration={len(y_orig)/sr:.2f}s  |  "
                        f"max_amp={np.max(np.abs(y_orig)):.4f}  |  "
                        f"rms={np.sqrt(np.mean(y_orig**2)):.4f}",
                        fontsize=11, fontweight="bold", color="#2D3436")
            ax.set_ylabel("Amplitude")
            ax.set_xlabel("")

            # Ve tung ban augmented
            aug_colors = ["#0984E3", "#6C5CE7", "#E17055", "#00B894"]
            for j, (method, aug_path) in enumerate(pair["augmented"]):
                y_aug, sr = librosa.load(aug_path, sr=SAMPLE_RATE, mono=True)
                ax = axes[j + 1]
                color = aug_colors[j % len(aug_colors)]
                librosa.display.waveshow(y_aug, sr=sr, ax=ax, color=color, alpha=0.85)
                ax.set_title(
                    f"AUG: {method}  |  duration={len(y_aug)/sr:.2f}s  |  "
                    f"max_amp={np.max(np.abs(y_aug)):.4f}  |  "
                    f"rms={np.sqrt(np.mean(y_aug**2)):.4f}",
                    fontsize=11, fontweight="bold", color=color
                )
                ax.set_ylabel("Amplitude")
                if j < n_aug - 1:
                    ax.set_xlabel("")

            axes[-1].set_xlabel("Thoi gian (s)")

            plt.tight_layout()
            path = os.path.join(output_dir,
                               f"10_waveform_compare_{cls}_sample{pair_idx+1}.png")
            plt.savefig(path)
            plt.show()
            print(f"[OK] Da luu: {os.path.basename(path)}")


def plot_waveform_grid_all_classes(train_dir, aug_dir, output_dir):
    """
    Ve 1 figure lon: moi class 1 hang, cot 1 = Original, cot 2-4 = Augmented.
    De nhin tong quan su khac biet giua cac loai augmentation.
    """
    classes = sorted(os.listdir(train_dir))
    classes = [c for c in classes if os.path.isdir(os.path.join(train_dir, c))]
    n_classes = len(classes)
    n_cols = 4  # 1 original + 3 aug

    fig, axes = plt.subplots(n_classes, n_cols, figsize=(5 * n_cols, 3 * n_classes))
    fig.suptitle("So sanh Time Domain: Original vs Augmented (moi class)",
                 fontsize=16, fontweight="bold", y=1.01)

    for i, cls in enumerate(classes):
        pairs = find_original_and_augmented_pairs(train_dir, aug_dir, cls, n_pairs=1)

        if not pairs:
            for j in range(n_cols):
                axes[i, j].text(0.5, 0.5, "No data", ha="center", va="center")
                axes[i, j].set_title(f"{cls}" if j == 0 else "")
            continue

        pair = pairs[0]

        # Cot 0: Original
        y_orig, sr = librosa.load(pair["original"], sr=SAMPLE_RATE, mono=True)
        ax = axes[i, 0]
        librosa.display.waveshow(y_orig, sr=sr, ax=ax,
                                 color=CLASS_COLORS.get(cls, "#333"), alpha=0.85)
        ax.set_title(f"{cls} - ORIGINAL", fontsize=10, fontweight="bold",
                    color="#2D3436")
        ax.set_ylabel("Amp" if True else "")

        # Cot 1-3: Augmented
        aug_colors = ["#0984E3", "#6C5CE7", "#E17055"]
        for j in range(min(n_cols - 1, len(pair["augmented"]))):
            method, aug_path = pair["augmented"][j]
            y_aug, sr = librosa.load(aug_path, sr=SAMPLE_RATE, mono=True)
            ax = axes[i, j + 1]
            librosa.display.waveshow(y_aug, sr=sr, ax=ax,
                                     color=aug_colors[j % len(aug_colors)], alpha=0.85)
            ax.set_title(f"{method}", fontsize=10, fontweight="bold",
                        color=aug_colors[j % len(aug_colors)])

        # An cac cot thua
        for j in range(len(pair["augmented"]) + 1, n_cols):
            axes[i, j].axis("off")

        # Chi hien xlabel o hang cuoi
        if i < n_classes - 1:
            for j in range(n_cols):
                axes[i, j].set_xlabel("")

    for j in range(n_cols):
        axes[-1, j].set_xlabel("Time (s)")

    plt.tight_layout()
    path = os.path.join(output_dir, "11_waveform_grid_all_classes.png")
    plt.savefig(path)
    plt.show()
    print(f"[OK] Da luu: {os.path.basename(path)}")


def print_summary(train_counts, aug_counts, test_counts, aug_methods):
    """In thong ke tom tat"""
    print("\n" + "=" * 70)
    print("THONG KE - SO SANH TRAIN vs AUGMENTATION")
    print("=" * 70)

    total_train = sum(train_counts.values())
    total_aug = sum(aug_counts.values())
    total_test = sum(test_counts.values())
    total_added = total_aug - total_train

    print(f"\nTong so file:")
    print(f"  Train (original): {total_train}")
    print(f"  Train + Aug:      {total_aug} (+{total_added} file)")
    print(f"  Test:             {total_test}")
    print(f"  Tong tat ca:      {total_train + total_test} -> {total_aug + total_test}")

    print(f"\n{'─' * 60}")
    print(f"  {'Class':15s} | {'Train':>6s} | {'Aug':>6s} | {'Added':>7s} | {'Test':>5s}")
    print(f"{'─' * 60}")
    for cls in sorted(train_counts.keys()):
        t = train_counts[cls]
        a = aug_counts[cls]
        added = a - t
        te = test_counts.get(cls, 0)
        ratio_str = f"+{added}" if added > 0 else "0"
        print(f"  {cls:15s} | {t:6d} | {a:6d} | {ratio_str:>7s} | {te:5d}")

    print(f"\n{'─' * 60}")
    print(f"Cac phuong phap augmentation da dung ({len(aug_methods)}):")
    for m in aug_methods:
        print(f"  - {m}")

    # Kiem tra can bang
    print(f"\n{'─' * 60}")
    print("Kiem tra can bang sau augmentation:")
    aug_vals = list(aug_counts.values())
    min_v, max_v = min(aug_vals), max(aug_vals)
    ratio = max_v / min_v if min_v > 0 else float('inf')
    print(f"  Min class: {min_v} files")
    print(f"  Max class: {max_v} files")
    print(f"  Ty le max/min: {ratio:.1f}x")
    if ratio <= 3:
        print("  -> Du lieu tuong doi can bang sau augmentation!")
    else:
        print("  -> Van con mat can bang, can xem xet them.")

    print("=" * 70 + "\n")


# ======================== MAIN ========================

def main():
    set_plot_style()

    print("BAT DAU PHAN TICH: Train vs Augmentation")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Aug dir:   {AUG_DIR}")
    print(f"Test dir:  {TEST_DIR}")
    print()

    # Kiem tra duong dan
    for d, name in [(TRAIN_DIR, "Train"), (AUG_DIR, "Augmentation"), (TEST_DIR, "Test")]:
        if not os.path.exists(d):
            print(f"[ERROR] Khong tim thay thu muc {name}: {d}")
            sys.exit(1)

    # Dem file
    train_counts = count_files_per_class(TRAIN_DIR)
    aug_counts = count_files_per_class(AUG_DIR)
    test_counts = count_files_per_class(TEST_DIR)
    aug_methods = get_aug_methods(AUG_DIR)

    # In thong ke
    print_summary(train_counts, aug_counts, test_counts, aug_methods)

    # Ve phan bo
    print("[1/4] Ve phan bo Train vs Augmentation...")
    plot_train_vs_aug_distribution(train_counts, aug_counts, test_counts, OUTPUT_DIR)

    print("\n[2/4] Ve ty le augmentation moi class...")
    plot_aug_pie(train_counts, aug_counts, OUTPUT_DIR)

    print("\n[3/4] Ve so sanh waveform chi tiet (truoc vs sau)...")
    plot_waveform_comparison(TRAIN_DIR, AUG_DIR, OUTPUT_DIR, n_classes_to_show=3)

    print("\n[4/4] Ve grid waveform tong quan toan bo class...")
    plot_waveform_grid_all_classes(TRAIN_DIR, AUG_DIR, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("HOAN THANH PHAN TICH AUGMENTATION!")
    print(f"Tat ca bieu do da duoc luu tai: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
