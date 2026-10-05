from pathlib import Path
import csv

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
TRIMMED_DIR = ROOT / "dataset_trimmed"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"

FS = 16000
FRAME_MS = 25
HOP_MS = 10

FRAME_LENGTH = round(FS * FRAME_MS / 1000)  # 400
HOP_LENGTH = round(FS * HOP_MS / 1000)     # 160

# Các tham số cần ghi trong báo cáo
TOP_DB = 35.0
NOISE_MARGIN_DB = 6.0
MARGIN_MS = 80
NOISE_FRAMES = 10

EPS = 1e-12


def load_audio(path):
    y, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False
    )

    if y.ndim == 2:
        y = np.mean(y, axis=1)

    if sr != FS:
        raise ValueError(
            f"{path.name}: sr={sr}, cần {FS} Hz. "
            "Hãy chạy lại phần A."
        )

    if len(y) == 0:
        raise ValueError(f"{path.name}: file rỗng.")

    return y


def framing(y):
    """Chia frame, zero-pad frame cuối nếu cần."""
    if len(y) <= FRAME_LENGTH:
        number_of_frames = 1
    else:
        number_of_frames = (
            1
            + int(np.ceil(
                (len(y) - FRAME_LENGTH) / HOP_LENGTH
            ))
        )

    padded_length = (
        (number_of_frames - 1) * HOP_LENGTH
        + FRAME_LENGTH
    )

    padded = np.pad(
        y,
        (0, padded_length - len(y)),
        mode="constant"
    )

    starts = (
        np.arange(number_of_frames) * HOP_LENGTH
    )

    indices = (
        starts[:, None]
        + np.arange(FRAME_LENGTH)[None, :]
    )

    frames = padded[indices]

    frame_times = (
        starts + FRAME_LENGTH / 2
    ) / FS

    return frames, frame_times


def calculate_log_energy(y):
    frames, frame_times = framing(y)

    window = np.hamming(FRAME_LENGTH)
    windowed_frames = frames * window

    energy = np.sum(
        windowed_frames ** 2,
        axis=1
    )

    log_energy_db = 10 * np.log10(
        energy + EPS
    )

    # Làm trơn trên 3 frame để giảm ảnh hưởng của spike đơn lẻ.
    padded_energy = np.pad(
        log_energy_db,
        (1, 1),
        mode="edge"
    )

    smoothed_energy_db = np.convolve(
        padded_energy,
        np.ones(3) / 3,
        mode="valid"
    )

    return (
        log_energy_db,
        smoothed_energy_db,
        frame_times
    )


def endpoint_detection(y):
    """
    Trả về:
      - tín hiệu đã trim
      - vị trí start/end theo sample
      - các đại lượng phục vụ vẽ hình
    """
    (
        log_energy_db,
        smoothed_energy_db,
        frame_times
    ) = calculate_log_energy(y)

    n_edge = min(
        NOISE_FRAMES,
        len(smoothed_energy_db)
    )

    edge_frames = np.concatenate([
        smoothed_energy_db[:n_edge],
        smoothed_energy_db[-n_edge:]
    ])

    noise_floor_db = float(
        np.median(edge_frames)
    )
    peak_energy_db = float(
        np.max(smoothed_energy_db)
    )

    relative_threshold = (
        peak_energy_db - TOP_DB
    )

    noise_threshold = (
        noise_floor_db + NOISE_MARGIN_DB
    )

    threshold_db = max(
        relative_threshold,
        noise_threshold
    )

    # Đảm bảo ngưỡng luôn thấp hơn đỉnh tiếng nói.
    threshold_db = min(
        threshold_db,
        peak_energy_db - 3.0
    )

    active_frames = np.flatnonzero(
        smoothed_energy_db >= threshold_db
    )

    if len(active_frames) == 0:
        # Nếu không phát hiện được speech, giữ nguyên file.
        return {
            "trimmed": y.copy(),
            "start_sample": 0,
            "end_sample": len(y),
            "threshold_db": threshold_db,
            "noise_floor_db": noise_floor_db,
            "peak_energy_db": peak_energy_db,
            "log_energy_db": log_energy_db,
            "smoothed_energy_db": smoothed_energy_db,
            "frame_times": frame_times,
            "status": "no_active_frame",
        }

    first_frame = int(active_frames[0])
    last_frame = int(active_frames[-1])

    start_sample = first_frame * HOP_LENGTH

    end_sample = (
        last_frame * HOP_LENGTH
        + FRAME_LENGTH
    )

    # Giữ thêm margin để bảo vệ phụ âm năng lượng thấp.
    margin_samples = round(
        FS * MARGIN_MS / 1000
    )

    start_sample = max(
        0,
        start_sample - margin_samples
    )

    end_sample = min(
        len(y),
        end_sample + margin_samples
    )

    trimmed = y[start_sample:end_sample]

    return {
        "trimmed": trimmed,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "threshold_db": threshold_db,
        "noise_floor_db": noise_floor_db,
        "peak_energy_db": peak_energy_db,
        "log_energy_db": log_energy_db,
        "smoothed_energy_db": smoothed_energy_db,
        "frame_times": frame_times,
        "status": "ok",
    }


def plot_diagnostic(
    source_path,
    label,
    y,
    result
):
    start_time = result["start_sample"] / FS
    end_time = result["end_sample"] / FS

    signal_times = np.arange(len(y)) / FS
    trimmed = result["trimmed"]
    trimmed_times = np.arange(len(trimmed)) / FS

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        constrained_layout=True
    )

    # Waveform gốc và endpoint
    axes[0].plot(
        signal_times,
        y,
        linewidth=0.7,
        color="steelblue"
    )
    axes[0].axvline(
        start_time,
        color="green",
        linestyle="--",
        label="Start"
    )
    axes[0].axvline(
        end_time,
        color="red",
        linestyle="--",
        label="End"
    )
    axes[0].axvspan(
        start_time,
        end_time,
        color="green",
        alpha=0.1
    )
    axes[0].set_title(
        f"Endpoint detection: {label}/{source_path.name}"
    )
    axes[0].set_ylabel("Biên độ")
    axes[0].legend()

    # Log-energy và threshold
    axes[1].plot(
        result["frame_times"],
        result["log_energy_db"],
        color="lightgray",
        linewidth=1,
        label="Log-energy"
    )
    axes[1].plot(
        result["frame_times"],
        result["smoothed_energy_db"],
        color="darkorange",
        linewidth=1.4,
        label="Log-energy làm trơn"
    )
    axes[1].axhline(
        result["threshold_db"],
        color="red",
        linestyle="--",
        label=(
            f"Ngưỡng = "
            f"{result['threshold_db']:.2f} dB"
        )
    )
    axes[1].axvline(
        start_time,
        color="green",
        linestyle="--"
    )
    axes[1].axvline(
        end_time,
        color="red",
        linestyle="--"
    )
    axes[1].set_ylabel("Energy (dB)")
    axes[1].legend()

    # Waveform sau trim
    axes[2].plot(
        trimmed_times,
        trimmed,
        linewidth=0.7,
        color="purple"
    )
    axes[2].set_title(
        f"Sau trim: {len(trimmed) / FS:.3f} giây"
    )
    axes[2].set_xlabel("Thời gian (s)")
    axes[2].set_ylabel("Biên độ")

    for axis in axes:
        axis.grid(alpha=0.25)

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FIG_DIR
        / f"C_endpoint_{label}_{source_path.stem}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("Đã lưu hình:", output_path)


def write_summary(rows):
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULT_DIR / "C_trim_summary.csv"
    )

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Đã lưu bảng:", output_path)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            "Không tìm thấy thư mục dataset. "
            "Hãy chạy phần A trước."
        )

    label_dirs = sorted(
        directory
        for directory in DATA_DIR.iterdir()
        if directory.is_dir()
    )

    if not label_dirs:
        raise ValueError(
            "Không tìm thấy thư mục nhãn trong dataset."
        )

    summary_rows = []
    diagnostic_examples = []

    for label_dir in label_dirs:
        label = label_dir.name
        wav_files = sorted(
            label_dir.glob("*.wav")
        )

        if not wav_files:
            continue

        # Một file đại diện cho mỗi trong ba nhãn đầu.
        if len(diagnostic_examples) < 3:
            diagnostic_examples.append(
                wav_files[0]
            )

        for source_path in wav_files:
            y = load_audio(source_path)
            result = endpoint_detection(y)

            output_path = (
                TRIMMED_DIR
                / label
                / source_path.name
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            sf.write(
                output_path,
                result["trimmed"],
                FS,
                subtype="PCM_16"
            )

            before_duration = len(y) / FS
            after_duration = (
                len(result["trimmed"]) / FS
            )

            removed_duration = (
                before_duration - after_duration
            )

            removed_percent = (
                100 * removed_duration / before_duration
            )

            summary_rows.append({
                "label": label,
                "file": source_path.name,
                "before_s": round(
                    before_duration, 3
                ),
                "after_s": round(
                    after_duration, 3
                ),
                "removed_s": round(
                    removed_duration, 3
                ),
                "removed_percent": round(
                    removed_percent, 2
                ),
                "start_s": round(
                    result["start_sample"] / FS, 3
                ),
                "end_s": round(
                    result["end_sample"] / FS, 3
                ),
                "noise_floor_db": round(
                    result["noise_floor_db"], 2
                ),
                "threshold_db": round(
                    result["threshold_db"], 2
                ),
                "margin_ms": MARGIN_MS,
                "status": result["status"],
                "output": output_path
                    .relative_to(ROOT)
                    .as_posix(),
            })

    write_summary(summary_rows)

    # Vẽ hình chẩn đoán cho ba từ.
    for source_path in diagnostic_examples:
        label = source_path.parent.name
        y = load_audio(source_path)
        result = endpoint_detection(y)

        plot_diagnostic(
            source_path,
            label,
            y,
            result
        )

    print("\nHoàn thành phần C")
    print("Dữ liệu trim:", TRIMMED_DIR)
    print("Số file đã xử lý:", len(summary_rows))
    print("Margin đang dùng:", MARGIN_MS, "ms")


if __name__ == "__main__":
    main()