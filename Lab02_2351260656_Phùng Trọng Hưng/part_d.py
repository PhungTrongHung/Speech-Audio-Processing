from pathlib import Path
import csv

import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

TRIMMED_DIR = ROOT / "dataset_trimmed"
TRAIN_CSV = ROOT / "splits" / "train.csv"
FEATURE_DIR = ROOT / "features" / "mfcc_trimmed"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"

# Cấu hình baseline của Lab
FS = 16000
FRAME_MS = 25
HOP_MS = 10

WIN_LENGTH = round(FS * FRAME_MS / 1000)  # 400
HOP_LENGTH = round(FS * HOP_MS / 1000)   # 160

N_FFT = 512
N_MELS = 24
N_MFCC = 13
PRE_EMPHASIS = 0.97

# Cepstral Mean Normalization theo từng utterance
USE_CMN = True
EPS = 1e-9


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
            f"{path.name}: sr={sr} Hz, cần {FS} Hz."
        )

    if len(y) == 0:
        raise ValueError(f"{path.name}: file rỗng.")

    # Chuẩn hóa biên độ giống baseline trong tài liệu.
    peak = np.max(np.abs(y))

    if peak > EPS:
        y = y / peak

    # center=False cần tín hiệu dài ít nhất n_fft.
    if len(y) < N_FFT:
        y = np.pad(
            y,
            (0, N_FFT - len(y)),
            mode="constant"
        )

    return y.astype(np.float32)


def pre_emphasis(y, alpha=PRE_EMPHASIS):
    """
    y'[n] = y[n] - alpha*y[n-1]
    """
    emphasized = np.empty_like(y)
    emphasized[0] = y[0]

    emphasized[1:] = (
        y[1:] - alpha * y[:-1]
    )

    return emphasized


def mfcc_feature(y):
    """
    Trả về ma trận MFCC có shape (T, 13):
      T: số frame
      13: số hệ số MFCC trên mỗi frame
    """
    emphasized = pre_emphasis(y)

    mfcc = librosa.feature.mfcc(
        y=emphasized,
        sr=FS,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window="hamming",
        center=False
    )

    # Librosa trả về shape (13, T).
    if USE_CMN:
        mfcc = (
            mfcc
            - np.mean(
                mfcc,
                axis=1,
                keepdims=True
            )
        )

    # Đổi về (T, 13) để DTW so sánh từng hàng/frame.
    mfcc = mfcc.T.astype(np.float32)

    if mfcc.ndim != 2:
        raise ValueError(
            f"MFCC phải có 2 chiều, nhận được {mfcc.shape}"
        )

    if mfcc.shape[1] != N_MFCC:
        raise ValueError(
            f"MFCC phải có {N_MFCC} hệ số/frame, "
            f"nhận được shape {mfcc.shape}"
        )

    if not np.all(np.isfinite(mfcc)):
        raise ValueError(
            "MFCC chứa NaN hoặc infinity."
        )

    return mfcc


def process_all_files():
    wav_files = sorted(
        TRIMMED_DIR.glob("*/*.wav")
    )

    if not wav_files:
        raise FileNotFoundError(
            "Không tìm thấy WAV trong dataset_trimmed. "
            "Hãy kiểm tra kết quả phần C."
        )

    summary_rows = []

    for wav_path in wav_files:
        label = wav_path.parent.name

        y = load_audio(wav_path)
        mfcc = mfcc_feature(y)

        output_path = (
            FEATURE_DIR
            / label
            / f"{wav_path.stem}.npy"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(output_path, mfcc)

        summary_rows.append({
            "label": label,
            "file": wav_path.name,
            "duration_s": round(len(y) / FS, 3),
            "number_of_frames": mfcc.shape[0],
            "number_of_coefficients": mfcc.shape[1],
            "mfcc_shape": str(mfcc.shape),
            "pre_emphasis": PRE_EMPHASIS,
            "n_fft": N_FFT,
            "n_mels": N_MELS,
            "n_mfcc": N_MFCC,
            "cmn": USE_CMN,
            "feature_path": (
                output_path
                .relative_to(ROOT)
                .as_posix()
            ),
        })

        print(
            f"{label}/{wav_path.name}: "
            f"MFCC shape = {mfcc.shape}"
        )

    return summary_rows


def save_summary(rows):
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULT_DIR / "D_mfcc_summary.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Đã lưu bảng:", output_path)


def select_two_training_examples():
    """
    Chọn hai file train thuộc hai từ khác nhau.
    Không dùng file test để chọn hình minh họa.
    """
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            "Không tìm thấy splits/train.csv."
        )

    selected = {}

    with TRAIN_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row["label"]

            if label in selected:
                continue

            original_path = Path(row["path"])

            trimmed_path = (
                TRIMMED_DIR
                / label
                / original_path.name
            )

            if trimmed_path.exists():
                selected[label] = trimmed_path

            if len(selected) == 2:
                break

    if len(selected) < 2:
        raise ValueError(
            "Không chọn được hai file train "
            "thuộc hai nhãn khác nhau."
        )

    return list(selected.items())


def plot_mfcc_heatmaps(examples):
    feature_items = []

    for label, wav_path in examples:
        y = load_audio(wav_path)
        mfcc = mfcc_feature(y)

        feature_items.append(
            (label, wav_path, mfcc)
        )

    # Dùng cùng thang màu để dễ so sánh hai từ.
    all_values = np.concatenate([
        mfcc.ravel()
        for _, _, mfcc in feature_items
    ])

    vmin = np.percentile(all_values, 2)
    vmax = np.percentile(all_values, 98)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        layout="constrained"
    )

    last_image = None

    for axis, (label, wav_path, mfcc) in zip(
        axes,
        feature_items
    ):
        number_of_frames = mfcc.shape[0]

        end_time = max(
            number_of_frames * HOP_LENGTH / FS,
            HOP_LENGTH / FS
        )

        # mfcc đang có shape (T, 13), cần chuyển
        # thành (13, T) để trục ngang là thời gian.
        last_image = axis.imshow(
            mfcc.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                0,
                end_time,
                0,
                N_MFCC - 1
            ],
            cmap="magma",
            vmin=vmin,
            vmax=vmax
        )

        axis.set_title(
            f"Từ: {label} - {wav_path.name} - "
            f"shape={mfcc.shape}"
        )
        axis.set_xlabel("Thời gian (s)")
        axis.set_ylabel("Chỉ số hệ số MFCC")
        axis.set_yticks(range(N_MFCC))

    fig.colorbar(
        last_image,
        ax=axes,
        label="Giá trị MFCC sau CMN"
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FIG_DIR / "D_mfcc_two_words.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("Đã lưu heatmap:", output_path)


def main():
    print("Cấu hình MFCC:")
    print("  Sampling rate:", FS)
    print("  Frame:", FRAME_MS, "ms")
    print("  Hop:", HOP_MS, "ms")
    print("  Hamming window:", WIN_LENGTH, "mẫu")
    print("  NFFT:", N_FFT)
    print("  Mel filters:", N_MELS)
    print("  MFCC:", N_MFCC)
    print("  Pre-emphasis:", PRE_EMPHASIS)
    print("  CMN:", USE_CMN)

    summary_rows = process_all_files()
    save_summary(summary_rows)

    examples = select_two_training_examples()
    plot_mfcc_heatmaps(examples)

    print("\nHoàn thành phần D.")
    print("MFCC đã lưu tại:", FEATURE_DIR)


if __name__ == "__main__":
    main()