from pathlib import Path
from math import gcd
import csv
import random

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "dataset_raw"
DATA_DIR = ROOT / "dataset"
FIG_DIR = ROOT / "figures"
SPLIT_DIR = ROOT / "splits"

TARGET_SR = 16000
CLIP_LEVEL = 0.999
SILENCE_DB = 35.0
SEED = 42


def max_consecutive_true(mask):
    """Số mẫu liên tiếp dài nhất vượt ngưỡng clipping."""
    best = 0
    current = 0

    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def estimate_edge_silence(y, sr):
    """
    Ước lượng silence đầu/cuối bằng RMS theo frame.
    Đây là phép kiểm tra chất lượng ở phần A, chưa phải endpoint
    detection chính thức của phần C.
    """
    frame_length = round(0.025 * sr)
    hop_length = round(0.010 * sr)

    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))

    starts = range(0, len(y) - frame_length + 1, hop_length)
    rms = np.array([
        np.sqrt(np.mean(y[s:s + frame_length] ** 2) + 1e-12)
        for s in starts
    ])

    rms_db = 20 * np.log10(rms + 1e-12)
    threshold_db = np.max(rms_db) - SILENCE_DB
    active = np.flatnonzero(rms_db >= threshold_db)

    duration = len(y) / sr

    if len(active) == 0:
        return duration, duration

    first_start = active[0] * hop_length / sr
    last_end = (
        active[-1] * hop_length + frame_length
    ) / sr

    leading_silence = first_start
    trailing_silence = max(0.0, duration - last_end)

    return leading_silence, trailing_silence


def standardize_and_analyze(source_path, output_path):
    """
    Đọc WAV, đổi stereo thành mono, resample về 16 kHz,
    ghi ra PCM-16 và trả về số liệu kiểm tra chất lượng.
    """
    info = sf.info(source_path)
    data, sr = sf.read(
        source_path,
        dtype="float32",
        always_2d=True
    )

    original_channels = data.shape[1]
    y = np.mean(data, axis=1)  # Stereo/multi-channel -> mono

    source_peak = float(np.max(np.abs(y))) if len(y) else 0.0
    clip_mask = np.abs(y) >= CLIP_LEVEL
    clip_ratio = float(np.mean(clip_mask)) if len(y) else 0.0
    max_clip_run = max_consecutive_true(clip_mask)

    if sr != TARGET_SR:
        common = gcd(sr, TARGET_SR)
        y = resample_poly(
            y,
            TARGET_SR // common,
            sr // common
        ).astype(np.float32)

    # Tránh tạo clipping mới do dao động nhỏ của bộ lọc resample.
    converted_peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if converted_peak > 0.999:
        y = y * (0.999 / converted_peak)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        output_path,
        y,
        TARGET_SR,
        subtype="PCM_16"
    )

    leading, trailing = estimate_edge_silence(y, TARGET_SR)
    duration = len(y) / TARGET_SR
    peak_dbfs = 20 * np.log10(max(source_peak, 1e-12))

    # Nhiều mẫu liên tiếp chạm full-scale là dấu hiệu clipping mạnh.
    clipping_flag = max_clip_run >= 3 or clip_ratio >= 0.001

    notes = []

    if clipping_flag:
        notes.append("Co nguy co clipping")

    if leading < 0.2:
        notes.append("Silence dau ngan hon 0.2 s")
    elif leading > 0.5:
        notes.append("Silence dau dai hon 0.5 s")

    if trailing < 0.2:
        notes.append("Silence cuoi ngan hon 0.2 s")
    elif trailing > 0.5:
        notes.append("Silence cuoi dai hon 0.5 s")

    if not notes:
        notes.append("Dat yeu cau so bo")

    return {
        "file": output_path.relative_to(ROOT).as_posix(),
        "original_sr": info.samplerate,
        "original_channels": original_channels,
        "original_subtype": info.subtype,
        "duration_s": round(duration, 3),
        "peak_dbfs": round(peak_dbfs, 2),
        "clip_ratio_percent": round(100 * clip_ratio, 4),
        "max_clip_run": max_clip_run,
        "leading_silence_s": round(leading, 3),
        "trailing_silence_s": round(trailing, 3),
        "note": "; ".join(notes),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)


def create_split(label_dirs):
    """
    Split độc lập theo từng từ.
    Với đúng 5 file/từ: 3 train và 2 test.
    """
    rng = random.Random(SEED)
    train_rows = []
    test_rows = []

    for label_dir in label_dirs:
        label = label_dir.name
        files = sorted(label_dir.glob("*.wav"))
        rng.shuffle(files)

        # Giữ ít nhất 3 file train và 2 file test.
        n_test = max(2, round(0.4 * len(files)))
        n_test = min(n_test, len(files) - 3)

        test_files = files[:n_test]
        train_files = files[n_test:]

        for path in train_files:
            train_rows.append({
                "path": path.relative_to(ROOT).as_posix(),
                "label": label,
            })

        for path in test_files:
            test_rows.append({
                "path": path.relative_to(ROOT).as_posix(),
                "label": label,
            })

    write_csv(SPLIT_DIR / "train.csv", train_rows)
    write_csv(SPLIT_DIR / "test.csv", test_rows)

    return train_rows, test_rows


def plot_three_waveforms(label_dirs, report_rows):
    selected = [directory / sorted(
        p.name for p in directory.glob("*.wav")
    )[0] for directory in label_dirs[:3]]

    report_by_file = {
        row["file"]: row for row in report_rows
    }

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 9),
        constrained_layout=True
    )

    for ax, path in zip(axes, selected):
        y, sr = sf.read(path, dtype="float32")
        t = np.arange(len(y)) / sr

        relative_path = path.relative_to(ROOT).as_posix()
        row = report_by_file[relative_path]

        leading = float(row["leading_silence_s"])
        trailing = float(row["trailing_silence_s"])
        duration = len(y) / sr

        ax.plot(t, y, linewidth=0.8)
        ax.axhline(0.999, color="red", linestyle="--",
                   linewidth=0.8, label="Nguong clipping")
        ax.axhline(-0.999, color="red", linestyle="--",
                   linewidth=0.8)

        ax.axvspan(0, leading, color="gray", alpha=0.25,
                   label="Silence uoc luong")
        ax.axvspan(duration - trailing, duration,
                   color="gray", alpha=0.25)

        ax.set_title(
            f"{path.parent.name}: {path.name} | "
            f"peak={row['peak_dbfs']} dBFS"
        )
        ax.set_xlabel("Thoi gian (s)")
        ax.set_ylabel("Bien do")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = FIG_DIR / "A_waveforms.png"
    fig.savefig(output, dpi=200)
    plt.show()

    print("Da luu:", output)


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Khong tim thay thu muc: {RAW_DIR}"
        )

    raw_label_dirs = sorted(
        path for path in RAW_DIR.iterdir()
        if path.is_dir()
    )

    if not 5 <= len(raw_label_dirs) <= 10:
        raise ValueError(
            f"Can 5-10 tu, hien co {len(raw_label_dirs)} thu muc nhan."
        )

    report_rows = []

    for raw_label_dir in raw_label_dirs:
        wav_files = sorted(raw_label_dir.glob("*.wav"))

        if len(wav_files) < 5:
            raise ValueError(
                f"Nhãn '{raw_label_dir.name}' chi co "
                f"{len(wav_files)} file; can it nhat 5."
            )

        output_label_dir = DATA_DIR / raw_label_dir.name

        for source_path in wav_files:
            output_path = output_label_dir / source_path.name
            row = standardize_and_analyze(
                source_path,
                output_path
            )
            report_rows.append(row)

    write_csv(ROOT / "quality_report.csv", report_rows)

    standardized_label_dirs = sorted(
        path for path in DATA_DIR.iterdir()
        if path.is_dir()
    )

    train_rows, test_rows = create_split(
        standardized_label_dirs
    )
    plot_three_waveforms(
        standardized_label_dirs,
        report_rows
    )

    print("\nHoan thanh phan A")
    print("So file:", len(report_rows))
    print("So file train:", len(train_rows))
    print("So file test:", len(test_rows))
    print("Bao cao:", ROOT / "quality_report.csv")
    print("Train split:", SPLIT_DIR / "train.csv")
    print("Test split:", SPLIT_DIR / "test.csv")


if __name__ == "__main__":
    main()