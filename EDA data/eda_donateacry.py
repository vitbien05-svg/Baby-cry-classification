"""
=============================================================
EDA - Donateacry Corpus
Baby Cry Classification Project
=============================================================
Phân tích khám phá dữ liệu (Exploratory Data Analysis) cho
bộ dữ liệu donateacry-corpus đã được làm sạch.

Nội dung:
1. Thống kê số lượng file âm thanh mỗi class
2. Phân bố file âm thanh theo class (bar chart + pie chart)
3. Spectrogram mẫu cho mỗi class (3 mẫu / class)
4. Phân bố thời lượng (duration) của các file
5. Phân tích thêm: waveform, mel-spectrogram, thống kê metadata
   (giới tính, độ tuổi), sample rate, phân bố năng lượng...
=============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')

# ======================== CONFIG ========================
DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "corpus", "donateacry-corpus",
    "donateacry_corpus_cleaned_and_updated_data"
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RATE = 16000  # Tần số lấy mẫu chuẩn

# Bảng mã lý do khóc
REASON_MAP = {
    "hu": "Hungry",
    "bu": "Burping",
    "bp": "Belly Pain",
    "dc": "Discomfort",
    "ti": "Tired",
    "lo": "Lonely",
    "ch": "Cold/Hot",
    "sc": "Scared",
    "dk": "Don't Know"
}

# Bảng mã giới tính
GENDER_MAP = {
    "m": "Male",
    "f": "Female"
}

# Bảng mã độ tuổi
AGE_MAP = {
    "04": "0-4 weeks",
    "48": "4-8 weeks",
    "26": "2-6 months",
    "72": "7m-2 years",
    "22": ">2 years"
}

# Màu sắc đẹp cho từng class
CLASS_COLORS = {
    "belly_pain": "#FF6B6B",
    "burping": "#4ECDC4",
    "discomfort": "#FFE66D",
    "hungry": "#95E1D3",
    "tired": "#AA96DA"
}

# ======================== HÀM TIỆN ÍCH ========================

def parse_filename(filename):
    """Phân tích tên file để lấy metadata (giới tính, tuổi, lý do khóc)"""
    parts = filename.rsplit(".", 1)[0]  # bỏ extension
    segments = parts.split("-")

    # Tên file có dạng: uuid-timestamp-version-gender-age-reason
    # Với iOS: uuid (36 chars) có 4 dấu gạch ngang
    # Với Android: uuid (36 chars) có 4 dấu gạch ngang, timestamp dài hơn

    try:
        # Lấy từ cuối: reason, age, gender
        reason_code = segments[-1]
        age_code = segments[-2]
        gender_code = segments[-3]

        gender = GENDER_MAP.get(gender_code, "Unknown")
        age = AGE_MAP.get(age_code, "Unknown")
        reason = REASON_MAP.get(reason_code, "Unknown")

        return {
            "gender": gender,
            "age": age,
            "reason": reason,
            "gender_code": gender_code,
            "age_code": age_code,
            "reason_code": reason_code
        }
    except (IndexError, KeyError):
        return {
            "gender": "Unknown",
            "age": "Unknown",
            "reason": "Unknown",
            "gender_code": "",
            "age_code": "",
            "reason_code": ""
        }


def load_audio_info(data_root, sr=SAMPLE_RATE):
    """Tải thông tin của tất cả file âm thanh trong dataset"""
    records = []

    for class_name in sorted(os.listdir(data_root)):
        class_path = os.path.join(data_root, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in sorted(os.listdir(class_path)):
            if not filename.endswith(".wav"):
                continue

            filepath = os.path.join(class_path, filename)
            file_size = os.path.getsize(filepath)

            # Load audio để lấy thông tin
            try:
                y, sr_loaded = librosa.load(filepath, sr=sr, mono=True)
                duration = librosa.get_duration(y=y, sr=sr_loaded)

                # Phân tích metadata từ tên file
                meta = parse_filename(filename)

                records.append({
                    "filename": filename,
                    "filepath": filepath,
                    "class": class_name,
                    "duration": duration,
                    "sr": sr_loaded,
                    "n_samples": len(y),
                    "file_size_kb": file_size / 1024,
                    "rms_energy": float(np.sqrt(np.mean(y**2))),
                    "max_amplitude": float(np.max(np.abs(y))),
                    "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y))),
                    "gender": meta["gender"],
                    "age": meta["age"],
                })
            except Exception as e:
                print(f"  ⚠ Lỗi khi xử lý {filename}: {e}")

    return pd.DataFrame(records)


def set_plot_style():
    """Thiết lập style cho matplotlib"""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FFFFFF",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


# ======================== CÁC HÀM VẼ ========================

def plot_class_distribution(df, output_dir):
    """1. Vẽ phân bố số lượng file theo class"""
    class_counts = df["class"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phân bố số lượng file âm thanh theo class", fontsize=16, fontweight="bold")

    # Bar chart
    colors = [CLASS_COLORS.get(c, "#888888") for c in class_counts.index]
    bars = axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor="#333333", linewidth=0.8)

    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                     str(count), ha="center", va="bottom", fontweight="bold", fontsize=12)

    axes[0].set_xlabel("Class (Lý do khóc)")
    axes[0].set_ylabel("Số lượng file")
    axes[0].set_title("Bar Chart - Số lượng file mỗi class")
    axes[0].tick_params(axis='x', rotation=15)

    # Pie chart
    explode = [0.05] * len(class_counts)
    wedges, texts, autotexts = axes[1].pie(
        class_counts.values,
        labels=class_counts.index,
        autopct=lambda p: f"{p:.1f}%\n({int(p * sum(class_counts.values) / 100)})",
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=140,
        textprops={"fontsize": 10}
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    axes[1].set_title("Pie Chart - Tỷ lệ phần trăm mỗi class")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_distribution.png"))
    plt.show()
    print("✅ Đã lưu: 01_class_distribution.png")


def plot_spectrograms(df, data_root, output_dir, n_samples=3):
    """2. Vẽ Spectrogram mẫu cho mỗi class (3 mẫu / class)"""
    classes = sorted(df["class"].unique())
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, n_samples, figsize=(5 * n_samples, 4 * n_classes))
    fig.suptitle(f"Spectrogram mẫu - {n_samples} file mỗi class",
                 fontsize=16, fontweight="bold", y=1.02)

    for i, cls in enumerate(classes):
        cls_files = df[df["class"] == cls].sample(
            n=min(n_samples, len(df[df["class"] == cls])),
            random_state=42
        )

        for j, (_, row) in enumerate(cls_files.iterrows()):
            ax = axes[i, j] if n_classes > 1 else axes[j]

            y, sr = librosa.load(row["filepath"], sr=SAMPLE_RATE, mono=True)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel",
                                           sr=sr, fmax=8000, ax=ax, cmap="magma")

            ax.set_title(f"{cls} (#{j+1})", fontsize=11, fontweight="bold",
                        color=CLASS_COLORS.get(cls, "#333333"))
            if j == 0:
                ax.set_ylabel("Tần số (Hz)")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Thời gian (s)" if i == n_classes - 1 else "")

        # Thêm colorbar cho cột cuối
        fig.colorbar(img, ax=axes[i, :] if n_classes > 1 else axes[:],
                     format="%+2.0f dB", shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_spectrograms_per_class.png"))
    plt.show()
    print("✅ Đã lưu: 02_spectrograms_per_class.png")


def plot_duration_distribution(df, output_dir):
    """3. Phân bố thời lượng của các file âm thanh"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phân bố thời lượng file âm thanh", fontsize=16, fontweight="bold")

    # Histogram tổng thể
    axes[0].hist(df["duration"], bins=30, color="#4ECDC4", edgecolor="#333333",
                alpha=0.85, linewidth=0.8)
    axes[0].axvline(df["duration"].mean(), color="#FF6B6B", linestyle="--",
                   linewidth=2, label=f'Trung bình: {df["duration"].mean():.2f}s')
    axes[0].axvline(df["duration"].median(), color="#FFE66D", linestyle="-.",
                   linewidth=2, label=f'Trung vị: {df["duration"].median():.2f}s')
    axes[0].set_xlabel("Thời lượng (giây)")
    axes[0].set_ylabel("Số lượng file")
    axes[0].set_title("Histogram - Phân bố thời lượng tổng thể")
    axes[0].legend(fontsize=10)

    # Boxplot theo class
    classes = sorted(df["class"].unique())
    data_by_class = [df[df["class"] == cls]["duration"].values for cls in classes]
    colors = [CLASS_COLORS.get(c, "#888888") for c in classes]

    bp = axes[1].boxplot(data_by_class, labels=classes, patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_xlabel("Class (Lý do khóc)")
    axes[1].set_ylabel("Thời lượng (giây)")
    axes[1].set_title("Boxplot - Thời lượng theo class")
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_duration_distribution.png"))
    plt.show()
    print("✅ Đã lưu: 03_duration_distribution.png")


def plot_waveforms(df, output_dir, n_samples=1):
    """4. Vẽ Waveform mẫu cho mỗi class"""
    classes = sorted(df["class"].unique())
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, 1, figsize=(14, 3 * n_classes))
    fig.suptitle("Waveform mẫu cho mỗi class", fontsize=16, fontweight="bold")

    for i, cls in enumerate(classes):
        sample = df[df["class"] == cls].iloc[0]
        y, sr = librosa.load(sample["filepath"], sr=SAMPLE_RATE, mono=True)

        ax = axes[i]
        librosa.display.waveshow(y, sr=sr, ax=ax, color=CLASS_COLORS.get(cls, "#333333"), alpha=0.8)
        ax.set_title(f"Class: {cls}  |  Duration: {sample['duration']:.2f}s",
                    fontweight="bold", fontsize=12)
        ax.set_ylabel("Amplitude")
        if i < n_classes - 1:
            ax.set_xlabel("")

    axes[-1].set_xlabel("Thời gian (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_waveforms_per_class.png"))
    plt.show()
    print("✅ Đã lưu: 04_waveforms_per_class.png")


def plot_energy_distribution(df, output_dir):
    """5. Phân bố năng lượng (RMS Energy) theo class"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phân tích năng lượng âm thanh", fontsize=16, fontweight="bold")

    classes = sorted(df["class"].unique())

    # Violin plot cho RMS Energy
    data_by_class = [df[df["class"] == cls]["rms_energy"].values for cls in classes]
    parts = axes[0].violinplot(data_by_class, showmeans=True, showmedians=True)

    colors = [CLASS_COLORS.get(c, "#888888") for c in classes]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    axes[0].set_xticks(range(1, len(classes) + 1))
    axes[0].set_xticklabels(classes, rotation=15)
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("RMS Energy")
    axes[0].set_title("Violin Plot - RMS Energy theo class")

    # Zero Crossing Rate
    data_zcr = [df[df["class"] == cls]["zero_crossing_rate"].values for cls in classes]
    bp = axes[1].boxplot(data_zcr, labels=classes, patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Zero Crossing Rate")
    axes[1].set_title("Boxplot - Zero Crossing Rate theo class")
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_energy_analysis.png"))
    plt.show()
    print("✅ Đã lưu: 05_energy_analysis.png")


def plot_metadata_distribution(df, output_dir):
    """6. Phân bố metadata: giới tính, độ tuổi"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phân bố Metadata (Giới tính & Độ tuổi)", fontsize=16, fontweight="bold")

    # Giới tính
    gender_counts = df["gender"].value_counts()
    gender_colors = ["#74B9FF", "#FD79A8", "#B2BEC3"]  # Male, Female, Unknown
    axes[0].pie(gender_counts.values, labels=gender_counts.index,
               autopct="%1.1f%%", colors=gender_colors[:len(gender_counts)],
               shadow=True, startangle=90, textprops={"fontsize": 12})
    axes[0].set_title("Phân bố theo Giới tính")

    # Độ tuổi
    age_order = ["0-4 weeks", "4-8 weeks", "2-6 months", "7m-2 years", ">2 years", "Unknown"]
    age_counts = df["age"].value_counts()
    age_counts = age_counts.reindex([a for a in age_order if a in age_counts.index])
    age_colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3", "#AA96DA", "#B2BEC3"]

    bars = axes[1].barh(age_counts.index, age_counts.values,
                       color=age_colors[:len(age_counts)], edgecolor="#333333", linewidth=0.8)

    for bar, count in zip(bars, age_counts.values):
        axes[1].text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    str(count), va="center", fontweight="bold", fontsize=11)

    axes[1].set_xlabel("Số lượng file")
    axes[1].set_title("Phân bố theo Độ tuổi")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_metadata_distribution.png"))
    plt.show()
    print("✅ Đã lưu: 06_metadata_distribution.png")


def plot_gender_by_class(df, output_dir):
    """7. Phân bố giới tính trong từng class"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Phân bố giới tính trong từng class", fontsize=16, fontweight="bold")

    classes = sorted(df["class"].unique())
    cross_tab = pd.crosstab(df["class"], df["gender"])
    cross_tab = cross_tab.reindex(classes)

    cross_tab.plot(kind="bar", stacked=True, ax=ax,
                  color=["#FD79A8", "#74B9FF", "#B2BEC3"],
                  edgecolor="#333333", linewidth=0.8)

    ax.set_xlabel("Class")
    ax.set_ylabel("Số lượng file")
    ax.set_title("Stacked Bar - Giới tính theo class")
    ax.legend(title="Giới tính")
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_gender_by_class.png"))
    plt.show()
    print("✅ Đã lưu: 07_gender_by_class.png")


def print_summary_statistics(df):
    """In bảng thống kê tóm tắt"""
    print("\n" + "=" * 70)
    print("📊 THỐNG KÊ TỔNG QUAN - DONATEACRY CORPUS")
    print("=" * 70)

    print(f"\n📁 Tổng số file âm thanh: {len(df)}")
    print(f"📂 Số lượng class: {df['class'].nunique()}")

    print(f"\n{'─' * 50}")
    print("📋 Số lượng file theo class:")
    print(f"{'─' * 50}")
    class_counts = df["class"].value_counts().sort_index()
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {cls:15s} │ {count:4d} file ({pct:5.1f}%) │ {bar}")

    print(f"\n{'─' * 50}")
    print("⏱  Thống kê thời lượng (giây):")
    print(f"{'─' * 50}")
    print(f"  Min:    {df['duration'].min():.3f}s")
    print(f"  Max:    {df['duration'].max():.3f}s")
    print(f"  Mean:   {df['duration'].mean():.3f}s")
    print(f"  Median: {df['duration'].median():.3f}s")
    print(f"  Std:    {df['duration'].std():.3f}s")

    print(f"\n{'─' * 50}")
    print("🎵 Thống kê kỹ thuật:")
    print(f"{'─' * 50}")
    print(f"  Sample Rate:   {df['sr'].unique()}")
    print(f"  RMS Energy - Mean:   {df['rms_energy'].mean():.6f}")
    print(f"  RMS Energy - Std:    {df['rms_energy'].std():.6f}")
    print(f"  ZCR - Mean:          {df['zero_crossing_rate'].mean():.6f}")
    print(f"  ZCR - Std:           {df['zero_crossing_rate'].std():.6f}")
    print(f"  Max Amplitude - Mean: {df['max_amplitude'].mean():.4f}")

    print(f"\n{'─' * 50}")
    print("👶 Metadata:")
    print(f"{'─' * 50}")
    print(f"  Giới tính: {dict(df['gender'].value_counts())}")
    print(f"  Độ tuổi:   {dict(df['age'].value_counts())}")

    print(f"\n{'─' * 50}")
    print("⏱  Thời lượng trung bình theo class:")
    print(f"{'─' * 50}")
    for cls in sorted(df["class"].unique()):
        cls_df = df[df["class"] == cls]
        print(f"  {cls:15s} │ mean={cls_df['duration'].mean():.3f}s  "
              f"│ std={cls_df['duration'].std():.3f}s  "
              f"│ min={cls_df['duration'].min():.3f}s  "
              f"│ max={cls_df['duration'].max():.3f}s")

    print("\n" + "=" * 70)

    # Cảnh báo mất cân bằng
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count

    if imbalance_ratio > 3:
        print(f"\n⚠️  CẢNH BÁO: Dataset rất MẤT CÂN BẰNG!")
        print(f"   Tỷ lệ max/min = {imbalance_ratio:.1f}x")
        print(f"   Class lớn nhất: {class_counts.idxmax()} ({max_count} files)")
        print(f"   Class nhỏ nhất: {class_counts.idxmin()} ({min_count} files)")
        print(f"   → Cần xem xét các kỹ thuật: SMOTE, data augmentation, "
              f"class weights, undersampling...")
    print()


# ======================== MAIN ========================

def main():
    set_plot_style()

    print("🔊 BẮT ĐẦU PHÂN TÍCH EDA - DONATEACRY CORPUS")
    print(f"📂 Đường dẫn dữ liệu: {DATA_ROOT}")
    print()

    # Kiểm tra đường dẫn
    if not os.path.exists(DATA_ROOT):
        print(f"❌ Không tìm thấy thư mục dữ liệu: {DATA_ROOT}")
        sys.exit(1)

    # Bước 1: Tải thông tin tất cả file
    print("📖 Đang tải và phân tích thông tin file âm thanh...")
    df = load_audio_info(DATA_ROOT)

    if df.empty:
        print("❌ Không tìm thấy file âm thanh nào!")
        sys.exit(1)

    # Lưu dataframe để dùng lại
    csv_path = os.path.join(OUTPUT_DIR, "audio_info.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Đã lưu thông tin vào: {csv_path}")

    # In thống kê tóm tắt
    print_summary_statistics(df)

    # Bước 2: Vẽ phân bố class
    print("\n📊 [1/6] Vẽ phân bố số lượng file theo class...")
    plot_class_distribution(df, OUTPUT_DIR)

    # Bước 3: Vẽ spectrogram
    print("\n📊 [2/6] Vẽ Spectrogram mẫu cho mỗi class...")
    plot_spectrograms(df, DATA_ROOT, OUTPUT_DIR, n_samples=3)

    # Bước 4: Phân bố thời lượng
    print("\n📊 [3/6] Vẽ phân bố thời lượng...")
    plot_duration_distribution(df, OUTPUT_DIR)

    # Bước 5: Waveform
    print("\n📊 [4/6] Vẽ waveform mẫu cho mỗi class...")
    plot_waveforms(df, OUTPUT_DIR)

    # Bước 6: Phân tích năng lượng
    print("\n📊 [5/6] Vẽ phân bố năng lượng...")
    plot_energy_distribution(df, OUTPUT_DIR)

    # Bước 7: Metadata
    print("\n📊 [6/6] Vẽ phân bố metadata...")
    plot_metadata_distribution(df, OUTPUT_DIR)
    plot_gender_by_class(df, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH PHÂN TÍCH EDA!")
    print(f"📂 Tất cả biểu đồ đã được lưu tại: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
