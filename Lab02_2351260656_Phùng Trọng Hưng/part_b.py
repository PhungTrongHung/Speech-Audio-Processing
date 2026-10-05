from pathlib import Path
import csv

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "splits" / "train.csv"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"

FS = 16000
FRAME_MS = 25
HOP_MS = 10

FRAME_LENGTH = round(FS * FRAME_MS / 1000)  # 400
HOP_LENGTH = round(FS * HOP_MS / 1000)     # 160
EPS = 1e-12


def load_audio(path):
    """Đọc WAV và kiểm tra đầu vào của phần B."""
    y, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False
    )

    if y.ndim == 2:
        y = np.mean(y, axis=1)

    if sr != FS:
        raise ValueError(
            f"{path.name}: sampling rate = {sr} Hz, "
            f"không phải {FS} Hz. Hãy chạy phần A trước."
        )

    if len(y) == 0:
        raise ValueError(f"{path.name}: file rỗng.")

    return y


def framing(y, frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH):
    """
    Chia tín hiệu thành các frame chồng lấn.
    Frame cuối được zero-padding nếu thiếu mẫu.
    """
    if len(y) <= frame_length:
        number_of_frames = 1
    else:
        number_of_frames = (
            1
            + int(np.ceil(
                (len(y) - frame_length) / hop_length
            ))
        )

    padded_length = (
        (number_of_frames - 1) * hop_length
        + frame_length
    )

    y_padded = np.pad(
        y,
        (0, padded_length - len(y)),
        mode="constant"
    )

    starts = (
        np.arange(number_of_frames) * hop_length
    )

    indices = (
        starts[:, None]
        + np.arange(frame_length)[None, :]
    )

    frames = y_padded[indices]

    # Thời điểm tại tâm của mỗi frame.
    frame_times = (
        starts + frame_length / 2
    ) / FS

    return frames, frame_times


def short_time_features(y):
    frames, frame_times = framing(y)

    # Hamming window theo yêu cầu của Lab.
    hamming = np.hamming(FRAME_LENGTH)
    windowed_frames = frames * hamming

    # Short-time energy và RMS trên frame đã window.
    energy = np.sum(
        windowed_frames ** 2,
        axis=1
    )

    rms = np.sqrt(
        np.mean(windowed_frames ** 2, axis=1)
    )

    log_energy_db = 10 * np.log10(
        energy + EPS
    )

    # Theo định nghĩa của tài liệu:
    # sgn(x) = 1 nếu x >= 0, ngược lại bằng -1.
    signs = np.where(frames >= 0, 1.0, -1.0)

    sign_changes = np.abs(
        signs[:, 1:] - signs[:, :-1]
    )

    zcr = np.sum(
        sign_changes,
        axis=1
    ) / (2 * FRAME_LENGTH)

    assert len(energy) == len(rms) == len(zcr)
    assert np.all(energy >= 0)
    assert np.all(rms >= 0)
    assert np.all((zcr >= 0) & (zcr <= 1))

    return {
        "times": frame_times,
        "energy": energy,
        "log_energy_db": log_energy_db,
        "rms": rms,
        "zcr": zcr,
    }


def select_three_training_files():
    """
    Tự chọn một file train từ mỗi nhãn,
    lấy ba nhãn đầu tiên.
    """
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            "Không tìm thấy splits/train.csv. "
            "Hãy chạy phần A trước."
        )

    selected_by_label = {}

    with TRAIN_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row["label"]

            if label not in selected_by_label:
                selected_by_label[label] = (
                    ROOT / Path(row["path"])
                )

    if len(selected_by_label) < 3:
        raise ValueError(
            "Cần ít nhất ba nhãn trong train.csv."
        )

    return list(selected_by_label.items())[:3]


def plot_features(path, label, y, features):
    signal_times = np.arange(len(y)) / FS
    frame_times = features["times"]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True
    )

    # 1. Waveform
    axes[0].plot(
        signal_times,
        y,
        linewidth=0.7,
        color="steelblue"
    )
    axes[0].set_ylabel("Biên độ")
    axes[0].set_title(
        f"Từ: {label} - File: {path.name}"
    )

    # 2. Log-energy
    axes[1].plot(
        frame_times,
        features["log_energy_db"],
        color="darkorange",
        marker=".",
        markersize=2
    )
    axes[1].set_ylabel("Energy (dB)")

    # 3. RMS
    axes[2].plot(
        frame_times,
        features["rms"],
        color="green",
        marker=".",
        markersize=2
    )
    axes[2].set_ylabel("RMS")

    # 4. ZCR
    axes[3].plot(
        frame_times,
        features["zcr"],
        color="purple",
        marker=".",
        markersize=2
    )
    axes[3].set_ylabel("ZCR")
    axes[3].set_xlabel("Thời gian (s)")
    axes[3].set_ylim(0, 1)

    for axis in axes:
        axis.grid(alpha=0.25)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    safe_stem = path.stem.replace(" ", "_")
    output_path = (
        FIG_DIR
        / f"B_{label}_{safe_stem}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("Đã lưu:", output_path)


def main():
    selected_files = select_three_training_files()
    summary_rows = []

    print(f"Frame length: {FRAME_LENGTH} mẫu")
    print(f"Hop length: {HOP_LENGTH} mẫu")

    for label, path in selected_files:
        y = load_audio(path)
        features = short_time_features(y)

        plot_features(
            path,
            label,
            y,
            features
        )

        summary_rows.append({
            "file": path.relative_to(ROOT).as_posix(),
            "label": label,
            "duration_s": round(len(y) / FS, 3),
            "number_of_frames": len(features["times"]),
            "min_log_energy_db": round(
                float(np.min(features["log_energy_db"])),
                3
            ),
            "max_log_energy_db": round(
                float(np.max(features["log_energy_db"])),
                3
            ),
            "mean_rms": round(
                float(np.mean(features["rms"])),
                6
            ),
            "mean_zcr": round(
                float(np.mean(features["zcr"])),
                6
            ),
            "max_zcr": round(
                float(np.max(features["zcr"])),
                6
            ),
        })

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULT_DIR / "B_feature_summary.csv"

    with summary_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Đã lưu bảng tổng hợp:", summary_path)
    print("Hoàn thành phần B.")


if __name__ == "__main__":
    main()