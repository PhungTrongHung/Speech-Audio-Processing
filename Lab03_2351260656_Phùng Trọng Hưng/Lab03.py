# ============================================================
# CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
# LAB 3 - NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM
#
# Pipeline:
# WAV
# -> trim silence
# -> pre-emphasis
# -> MFCC 13
# -> Delta 13
# -> Delta-Delta 13
# -> CMVN
# -> Whole-word Gaussian HMM
# -> Likelihood / Viterbi
# -> Recognition
# -> Confusion Matrix
# -> DTW baseline
# -> Lexicon
# -> Bigram LM re-ranking
# -> Robustness
# -> Experiments E1-E6
# ============================================================


# ============================================================
# 0. IMPORT THƯ VIỆN
# ============================================================

from pathlib import Path
from collections import Counter

import csv
import math
import time
import warnings

import numpy as np
import librosa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM

try:
    from hmmlearn.hmm import GMMHMM
except Exception:
    GMMHMM = None

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import joblib


warnings.filterwarnings("ignore")


# ============================================================
# 1. CẤU HÌNH
# ============================================================

DATASET_DIR = Path("dataset")
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# 5 từ theo Lab 3
LABELS = ["mo", "dong", "len", "xuong", "dung"]

SR = 16000

# Cấu hình HMM chính
N_STATES = 5
N_ITER = 80

# Chạy toàn bộ E1-E6 sẽ lâu hơn.
# Khi cần kiểm tra pipeline nhanh có thể đổi thành False.
RUN_EXPERIMENTS = True

# Bài mở rộng mục 9
RUN_GMM_EXTENSION = False

# SNR dùng cho robustness test
NOISE_SNR_DB = 20

# Ngữ cảnh minh họa cho Language Model
PREVIOUS_WORD_FOR_LM = "mo"


# ============================================================
# 2. KIỂM TRA DATASET
# ============================================================

def check_dataset():
    print("\n" + "=" * 70)
    print("KIỂM TRA CẤU TRÚC DATASET")
    print("=" * 70)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {TRAIN_DIR.resolve()}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {TEST_DIR.resolve()}"
        )

    total_train = 0
    total_test = 0

    for label in LABELS:
        train_path = TRAIN_DIR / label
        test_path = TEST_DIR / label

        if not train_path.exists():
            raise FileNotFoundError(
                f"Thiếu thư mục train của lớp '{label}': {train_path}"
            )

        if not test_path.exists():
            raise FileNotFoundError(
                f"Thiếu thư mục test của lớp '{label}': {test_path}"
            )

        n_train = len(list(train_path.glob("*.wav")))
        n_test = len(list(test_path.glob("*.wav")))

        total_train += n_train
        total_test += n_test

        print(
            f"{label:8s}: "
            f"train = {n_train:3d} | "
            f"test = {n_test:3d}"
        )

        if n_train == 0:
            raise RuntimeError(
                f"Lớp {label} không có file train WAV."
            )

        if n_test == 0:
            print(
                f"[WARNING] Lớp {label} chưa có file test."
            )

    print("-" * 70)
    print("Tổng train:", total_train)
    print("Tổng test :", total_test)


# ============================================================
# 3. TRÍCH ĐẶC TRƯNG MFCC
#    MFCC 13 + Delta 13 + Delta2 13 = 39 chiều
# ============================================================

def safe_delta(feature, order=1):
    """
    librosa.feature.delta mặc định width=9.
    Hàm này tự giảm width nếu utterance quá ngắn.
    """

    n_frames = feature.shape[1]

    if n_frames < 3:
        return np.zeros_like(feature)

    width = min(9, n_frames)

    # width phải là số lẻ
    if width % 2 == 0:
        width -= 1

    width = max(width, 3)

    return librosa.feature.delta(
        feature,
        width=width,
        order=order,
        mode="nearest"
    )


def compute_features_from_audio(
    y,
    sr=SR,
    use_dynamic=True,
    cmvn=True
):
    """
    Nhận waveform và trả về:
        F: T x 39 nếu use_dynamic=True
        F: T x 13 nếu use_dynamic=False
    """

    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        raise ValueError("Waveform rỗng.")

    # Endpoint detection đơn giản bằng trim
    y, _ = librosa.effects.trim(y, top_db=30)

    if len(y) < int(0.03 * sr):
        raise ValueError("Audio quá ngắn sau khi trim silence.")

    # Pre-emphasis
    y = librosa.effects.preemphasis(
        y,
        coef=0.97
    )

    # MFCC 13 chiều
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=512,
        win_length=int(0.025 * sr),   # 25 ms
        hop_length=int(0.010 * sr),   # 10 ms
        window="hamming",
        n_mels=26,
        fmin=20,
        fmax=sr / 2
    )

    if use_dynamic:
        d1 = safe_delta(mfcc, order=1)
        d2 = safe_delta(mfcc, order=2)

        # (39, T) -> (T, 39)
        F = np.vstack([
            mfcc,
            d1,
            d2
        ]).T
    else:
        # (13, T) -> (T, 13)
        F = mfcc.T

    # Loại NaN/inf nếu có
    if not np.all(np.isfinite(F)):
        F = np.nan_to_num(
            F,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    # Utterance-level CMVN
    if cmvn:
        mu = F.mean(
            axis=0,
            keepdims=True
        )

        sd = F.std(
            axis=0,
            keepdims=True
        ) + 1e-8

        F = (F - mu) / sd

    return F.astype(np.float64)


def extract_feat(
    path,
    sr=SR,
    use_dynamic=True,
    cmvn=True
):
    """
    Đọc file WAV rồi trích đặc trưng.
    """

    y, _ = librosa.load(
        path,
        sr=sr,
        mono=True
    )

    return compute_features_from_audio(
        y,
        sr=sr,
        use_dynamic=use_dynamic,
        cmvn=cmvn
    )


# ============================================================
# 4. KIỂM TRA SHAPE ĐẶC TRƯNG
# ============================================================

def inspect_feature_shapes(n_files=3):
    print("\n" + "=" * 70)
    print("KIỂM TRA SHAPE MFCC + DELTA + DELTA2")
    print("=" * 70)

    count = 0

    for label in LABELS:
        files = sorted(
            (TRAIN_DIR / label).glob("*.wav")
        )

        for wav in files:
            F = extract_feat(wav)

            print(
                f"{wav} -> shape = {F.shape}"
            )

            if F.shape[1] != 39:
                raise RuntimeError(
                    f"Sai chiều đặc trưng tại {wav}. "
                    f"Nhận {F.shape[1]}, cần 39."
                )

            count += 1

            if count >= n_files:
                return


# ============================================================
# 5. HEATMAP MFCC / DELTA / DELTA2
# ============================================================

def plot_feature_heatmap(wav_path, label):
    F = extract_feat(
        wav_path,
        use_dynamic=True
    )

    mfcc = F[:, 0:13].T
    d1 = F[:, 13:26].T
    d2 = F[:, 26:39].T

    components = [
        ("MFCC", mfcc),
        ("Delta MFCC", d1),
        ("Delta-Delta MFCC", d2)
    ]

    for name, data in components:
        plt.figure(figsize=(10, 4))

        plt.imshow(
            data,
            aspect="auto",
            origin="lower"
        )

        plt.colorbar()

        plt.xlabel("Frame")
        plt.ylabel("Coefficient")

        plt.title(
            f"{name} - {label} - {wav_path.name}"
        )

        filename = (
            f"heatmap_{label}_"
            f"{name.lower().replace(' ', '_')}.png"
        )

        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / filename,
            dpi=150
        )
        plt.close()


def generate_heatmaps():
    print("\n" + "=" * 70)
    print("VẼ HEATMAP CHO ÍT NHẤT 2 TỪ")
    print("=" * 70)

    plotted = 0

    for label in LABELS:
        files = sorted(
            (TRAIN_DIR / label).glob("*.wav")
        )

        if files:
            plot_feature_heatmap(
                files[0],
                label
            )

            print(
                f"Đã tạo heatmap: {label}"
            )

            plotted += 1

        if plotted >= 2:
            break


# ============================================================
# 6. TẠO LEFT-TO-RIGHT GAUSSIAN HMM
# ============================================================

def make_lr_hmm(
    n_states=5,
    n_iter=80,
    covariance_type="diag"
):
    """
    Left-to-right topology:

    state 0 -> state 0 / state 1
    state 1 -> state 1 / state 2
    ...
    final   -> final

    startprob = [1,0,...,0]

    params='mc':
    - m: means
    - c: covariance

    Transition matrix không được Baum-Welch thay đổi,
    nhờ đó giữ đúng topology left-to-right.
    """

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=1e-3,
        min_covar=1e-3,

        # hmmlearn khởi tạo means/covars
        init_params="mc",

        # Chỉ cập nhật means/covars,
        # giữ transition cố định
        params="mc",

        random_state=0
    )

    # P(state đầu)
    start = np.zeros(n_states)
    start[0] = 1.0

    # Ma trận transition
    A = np.zeros(
        (n_states, n_states)
    )

    for i in range(n_states - 1):
        A[i, i] = 0.7
        A[i, i + 1] = 0.3

    # State cuối self-loop
    A[-1, -1] = 1.0

    model.startprob_ = start
    model.transmat_ = A

    return model


# ============================================================
# 7. KIỂM TRA TOPOLOGY
# ============================================================

def check_lr_topology(model, atol=1e-8):
    """
    Trả True nếu không có transition lùi
    hoặc skip state.
    """

    A = model.transmat_
    n = len(A)

    for i in range(n):
        for j in range(n):

            allowed = (
                j == i or
                (i < n - 1 and j == i + 1)
            )

            if not allowed:
                if abs(A[i, j]) > atol:
                    return False

    return True


# ============================================================
# 8. HUẤN LUYỆN MỘT HMM CHO MỖI TỪ
# ============================================================

def train_models(
    n_states=5,
    use_dynamic=True,
    covariance_type="diag",
    verbose=True,
    save_models=False
):
    models = {}

    start_time = time.perf_counter()

    for label in LABELS:
        feats = []
        lengths = []

        files = sorted(
            (TRAIN_DIR / label).glob("*.wav")
        )

        if len(files) == 0:
            raise RuntimeError(
                f"Không có dữ liệu train cho {label}"
            )

        for wav in files:
            F = extract_feat(
                wav,
                use_dynamic=use_dynamic
            )

            feats.append(F)
            lengths.append(len(F))

        X = np.vstack(feats)

        model = make_lr_hmm(
            n_states=n_states,
            n_iter=N_ITER,
            covariance_type=covariance_type
        )

        model.fit(
            X,
            lengths
        )

        models[label] = model

        if verbose:
            print("\n" + "-" * 70)
            print("MODEL:", label)
            print("-" * 70)

            print(
                "Converged:",
                model.monitor_.converged
            )

            print(
                "Iterations:",
                model.monitor_.iter
            )

            print("startprob_:")

            print(
                np.round(
                    model.startprob_,
                    3
                )
            )

            print("transmat_:")

            print(
                np.round(
                    model.transmat_,
                    3
                )
            )

            print(
                "Left-to-right topology OK:",
                check_lr_topology(model)
            )

        if save_models:
            joblib.dump(
                model,
                MODEL_DIR / f"hmm_{label}.joblib"
            )

    elapsed = time.perf_counter() - start_time

    return models, elapsed


# ============================================================
# 9. MAXIMUM-LIKELIHOOD RECOGNITION
# ============================================================

def recognize_feature(F, models):
    scores = {}

    for label, model in models.items():
        try:
            score = model.score(F)
        except Exception:
            score = -np.inf

        scores[label] = float(score)

    pred = max(
        scores,
        key=scores.get
    )

    return pred, scores


def recognize(path, models, use_dynamic=True):
    F = extract_feat(
        path,
        use_dynamic=use_dynamic
    )

    pred, scores = recognize_feature(
        F,
        models
    )

    return pred, scores


# ============================================================
# 10. BẢNG ACOUSTIC SCORE
# ============================================================

def generate_score_table(
    models,
    n_test_files=3,
    use_dynamic=True
):
    print("\n" + "=" * 70)
    print("ACOUSTIC SCORE CỦA CÁC TEST UTTERANCE")
    print("=" * 70)

    rows = []
    counter = 0

    for true_label in LABELS:
        files = sorted(
            (TEST_DIR / true_label).glob("*.wav")
        )

        for wav in files:
            F = extract_feat(
                wav,
                use_dynamic=use_dynamic
            )

            pred, scores = recognize_feature(
                F,
                models
            )

            ranked = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            top1 = ranked[0][1]

            top2 = (
                ranked[1][1]
                if len(ranked) > 1
                else -np.inf
            )

            margin = top1 - top2

            row = {
                "file": wav.name,
                "true": true_label,
                "prediction": pred,
                "frames": len(F),
                "top1_top2_margin": margin
            }

            for label in LABELS:
                row[label] = scores[label]

            rows.append(row)

            print("\nFile       :", wav)
            print("True       :", true_label)
            print("Prediction :", pred)
            print("Frames     :", len(F))

            for label, score in ranked:
                print(
                    f"  {label:8s}: {score:12.2f}"
                )

            print(
                f"Top-1 - Top-2 margin = {margin:.2f}"
            )

            counter += 1

            if counter >= n_test_files:
                break

        if counter >= n_test_files:
            break

    csv_path = OUTPUT_DIR / "acoustic_scores.csv"

    if rows:
        fieldnames = (
            [
                "file",
                "true",
                "prediction",
                "frames"
            ]
            + LABELS
            + ["top1_top2_margin"]
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)

    print(
        "\nĐã lưu bảng score:",
        csv_path
    )


# ============================================================
# 11. VITERBI ALIGNMENT
# ============================================================

def plot_viterbi(
    wav_path,
    true_label,
    model,
    index
):
    F = extract_feat(wav_path)

    logp, states = model.decode(
        F,
        algorithm="viterbi"
    )

    print("\nViterbi:", wav_path)
    print("log-prob =", logp)
    print("state sequence:")
    print(states)

    # Duration mỗi state
    durations = {}

    for s in np.unique(states):
        durations[int(s)] = int(
            np.sum(states == s)
        )

    print(
        "Số frame ở từng state:",
        durations
    )

    plt.figure(figsize=(10, 4))

    plt.step(
        range(len(states)),
        states,
        where="post"
    )

    plt.xlabel("Frame")
    plt.ylabel("State")

    plt.title(
        f"Viterbi alignment: "
        f"{wav_path.name} vs {true_label}"
    )

    plt.yticks(
        range(model.n_components)
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / f"viterbi_{index}_{true_label}.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()


def generate_viterbi_examples(models):
    print("\n" + "=" * 70)
    print("VITERBI STATE ALIGNMENT")
    print("=" * 70)

    count = 0

    for label in LABELS:
        files = sorted(
            (TEST_DIR / label).glob("*.wav")
        )

        for wav in files:
            plot_viterbi(
                wav,
                label,
                models[label],
                count + 1
            )

            count += 1

            if count >= 2:
                return


# ============================================================
# 12. ĐÁNH GIÁ HMM
# ============================================================

def evaluate_models(
    models,
    use_dynamic=True,
    title="HMM",
    save_name="confusion_matrix_hmm.png"
):
    true_y = []
    pred_y = []

    test_time_total = 0.0
    n_trials = 0

    for label in LABELS:
        files = sorted(
            (TEST_DIR / label).glob("*.wav")
        )

        for wav in files:

            start = time.perf_counter()

            pred, _ = recognize(
                wav,
                models,
                use_dynamic=use_dynamic
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            test_time_total += elapsed
            n_trials += 1

            true_y.append(label)
            pred_y.append(pred)

    if len(true_y) == 0:
        raise RuntimeError(
            "Không có file test."
        )

    acc = accuracy_score(
        true_y,
        pred_y
    )

    cm = confusion_matrix(
        true_y,
        pred_y,
        labels=LABELS
    )

    avg_test_time = (
        test_time_total / n_trials
    )

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Accuracy = {acc * 100:.2f}%"
    )

    print(
        f"Average test time = "
        f"{avg_test_time:.4f} s/trial"
    )

    print("\nConfusion matrix:")

    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / save_name,
        dpi=160
    )

    plt.close()

    return {
        "accuracy": float(acc),
        "cm": cm,
        "true": true_y,
        "pred": pred_y,
        "test_time": avg_test_time
    }


# ============================================================
# 13. DTW BASELINE
#
# Do Lab hiện tại không cung cấp code Lab 2,
# ở đây dùng nearest-neighbour MFCC + DTW:
# một test utterance được so với mọi train utterance,
# label của template có DTW distance nhỏ nhất được chọn.
# ============================================================

def euclidean_cost_matrix(A, B):
    """
    A: T1 x D
    B: T2 x D

    Return:
    T1 x T2 Euclidean distance matrix
    """

    diff = (
        A[:, None, :]
        - B[None, :, :]
    )

    return np.sqrt(
        np.sum(
            diff * diff,
            axis=2
        )
    )


def dtw_distance(A, B):
    cost = euclidean_cost_matrix(
        A,
        B
    )

    n, m = cost.shape

    prev = np.full(
        m + 1,
        np.inf
    )

    prev[0] = 0.0

    for i in range(1, n + 1):

        cur = np.full(
            m + 1,
            np.inf
        )

        for j in range(1, m + 1):

            cur[j] = (
                cost[i - 1, j - 1]
                + min(
                    prev[j],
                    cur[j - 1],
                    prev[j - 1]
                )
            )

        prev = cur

    # Chuẩn hóa đơn giản theo độ dài chuỗi
    return prev[m] / (n + m)


def prepare_dtw_templates():
    print("\n" + "=" * 70)
    print("CHUẨN BỊ TEMPLATE DTW")
    print("=" * 70)

    templates = {}

    for label in LABELS:
        templates[label] = []

        files = sorted(
            (TRAIN_DIR / label).glob("*.wav")
        )

        for wav in files:

            # DTW baseline Lab 2:
            # dùng 13 MFCC tĩnh
            F = extract_feat(
                wav,
                use_dynamic=False
            )

            templates[label].append(
                (wav, F)
            )

        print(
            label,
            "templates =",
            len(templates[label])
        )

    return templates


def recognize_dtw(
    wav_path,
    templates
):
    X = extract_feat(
        wav_path,
        use_dynamic=False
    )

    best_label = None
    best_distance = np.inf

    class_scores = {}

    for label in LABELS:

        distances = []

        for _, template in templates[label]:

            d = dtw_distance(
                X,
                template
            )

            distances.append(d)

        if len(distances) == 0:
            class_distance = np.inf
        else:
            # Nearest template trong lớp
            class_distance = min(
                distances
            )

        class_scores[label] = (
            class_distance
        )

        if class_distance < best_distance:

            best_distance = (
                class_distance
            )

            best_label = label

    return (
        best_label,
        class_scores
    )


def evaluate_dtw(templates):
    print("\n" + "=" * 70)
    print("ĐÁNH GIÁ MFCC + DTW")
    print("=" * 70)

    true_y = []
    pred_y = []

    start_total = time.perf_counter()

    total_trials = 0

    for label in LABELS:

        files = sorted(
            (TEST_DIR / label).glob("*.wav")
        )

        for wav in files:

            pred, _ = recognize_dtw(
                wav,
                templates
            )

            true_y.append(label)
            pred_y.append(pred)

            total_trials += 1

    elapsed = (
        time.perf_counter()
        - start_total
    )

    acc = accuracy_score(
        true_y,
        pred_y
    )

    cm = confusion_matrix(
        true_y,
        pred_y,
        labels=LABELS
    )

    avg_time = (
        elapsed / max(total_trials, 1)
    )

    print(
        f"DTW Accuracy = "
        f"{acc * 100:.2f}%"
    )

    print(
        f"DTW test time = "
        f"{avg_time:.4f} s/trial"
    )

    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "MFCC + DTW baseline"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "confusion_matrix_dtw.png",
        dpi=160
    )

    plt.close()

    return {
        "accuracy": float(acc),
        "cm": cm,
        "test_time": avg_time
    }


# ============================================================
# 14. SO SÁNH HMM VÀ DTW
# ============================================================

def top_confusion_pair(cm):
    cm = np.array(cm).copy()

    np.fill_diagonal(
        cm,
        0
    )

    if cm.max() == 0:
        return "None"

    i, j = np.unravel_index(
        np.argmax(cm),
        cm.shape
    )

    return (
        f"{LABELS[i]} -> "
        f"{LABELS[j]} "
        f"({cm[i, j]})"
    )


def save_dtw_hmm_comparison(
    hmm_result,
    hmm_train_time,
    dtw_result
):
    output = (
        OUTPUT_DIR
        / "dtw_vs_hmm.csv"
    )

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Chi so",
            "DTW Lab 2",
            "HMM Lab 3",
            "Nhan xet"
        ])

        writer.writerow([
            "Accuracy",
            f"{dtw_result['accuracy'] * 100:.2f}%",
            f"{hmm_result['accuracy'] * 100:.2f}%",
            "Cùng train/test split"
        ])

        writer.writerow([
            "So mau train/lop",
            "Dùng toàn bộ template train",
            "Dùng toàn bộ utterance train",
            "Cùng dữ liệu"
        ])

        writer.writerow([
            "Thoi gian train",
            "Không có bước fit model",
            f"{hmm_train_time:.4f}s",
            "DTW lưu template; HMM phải huấn luyện"
        ])

        writer.writerow([
            "Thoi gian test/trial",
            f"{dtw_result['test_time']:.4f}s",
            f"{hmm_result['test_time']:.4f}s",
            ""
        ])

        writer.writerow([
            "Top confusion pair",
            top_confusion_pair(
                dtw_result["cm"]
            ),
            top_confusion_pair(
                hmm_result["cm"]
            ),
            ""
        ])

        writer.writerow([
            "Alignment",
            "Warping path",
            "Viterbi state path",
            ""
        ])

    print(
        "\nĐã lưu bảng DTW-vs-HMM:",
        output
    )


# ============================================================
# 15. LEXICON MINI
# ============================================================

LEXICON_PATH = Path("lexicon.txt")


def create_lexicon_template():
    """
    Lab yêu cầu phone sequence phải do giảng viên/
    phone inventory cung cấp.

    Vì tài liệu không quy định phiên âm tiếng Việt cụ thể,
    chương trình KHÔNG tự bịa phone sequence.
    """

    if LEXICON_PATH.exists():
        return

    content = """# LAB 3 - MINI LEXICON
#
# Format:
# WORD PHONE_1 PHONE_2 ... PHONE_K
#
# Hãy thay bằng phone inventory do giảng viên cung cấp.
#
# Ví dụ FORMAT בלבד:
# WORD PHONE_1 PHONE_2
#
# Không tự sử dụng dòng ví dụ làm phiên âm tiếng Việt.
"""

    LEXICON_PATH.write_text(
        content,
        encoding="utf-8"
    )

    print(
        "\nĐã tạo:",
        LEXICON_PATH
    )

    print(
        "Hãy bổ sung phone sequence "
        "theo inventory do giảng viên cung cấp."
    )


def parse_lexicon(path):
    lexicon = {}

    if not Path(path).exists():
        return lexicon

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) >= 2:

                word = parts[0]

                phones = parts[1:]

                lexicon[word] = phones

    return lexicon


# ============================================================
# 16. BIGRAM LANGUAGE MODEL
# ============================================================

CORPUS_PATH = Path("corpus.txt")


def create_demo_corpus():
    """
    Chỉ tạo corpus MINH HỌA khi chưa có corpus của giảng viên.

    Sinh viên nên thay bằng corpus/lệnh do giảng viên cung cấp.
    """

    if CORPUS_PATH.exists():
        return

    text = """mo len
mo len
mo xuong
mo dung
dong dung
len dung
xuong dung
mo dong
dong mo
"""

    CORPUS_PATH.write_text(
        text,
        encoding="utf-8"
    )

    print(
        "\nĐã tạo corpus minh họa:",
        CORPUS_PATH
    )

    print(
        "Nên thay corpus này bằng corpus "
        "do giảng viên cung cấp nếu có."
    )


def load_corpus(path):
    sentences = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            words = line.strip().split()

            if words:
                sentences.append(words)

    return sentences


def build_ngram_counts(sentences):
    counts1 = Counter()
    counts2 = Counter()

    for words in sentences:

        for word in words:
            counts1[word] += 1

        for prev, cur in zip(
            words[:-1],
            words[1:]
        ):
            counts2[
                (prev, cur)
            ] += 1

    return counts1, counts2


def bigram_prob(
    prev,
    cur,
    counts2,
    counts1,
    V,
    k=0.5
):
    """
    Add-k smoothing.
    """

    numerator = (
        counts2.get(
            (prev, cur),
            0
        )
        + k
    )

    denominator = (
        counts1.get(
            prev,
            0
        )
        + k * V
    )

    return numerator / denominator


def rerank(
    F,
    prev,
    models,
    counts2,
    counts1,
    lm_weight=0.3
):
    """
    score =
        acoustic log likelihood / T
        + lambda_LM * log P(word | previous word)
    """

    T = len(F)

    total = {}

    details = {}

    for word, model in models.items():

        acoustic = (
            model.score(F) / T
        )

        lm_probability = bigram_prob(
            prev,
            word,
            counts2,
            counts1,
            len(models),
            k=0.5
        )

        lm_score = math.log(
            lm_probability
        )

        combined = (
            acoustic
            + lm_weight * lm_score
        )

        total[word] = combined

        details[word] = {
            "acoustic_per_frame": acoustic,
            "lm_log_prob": lm_score,
            "combined": combined
        }

    pred = max(
        total,
        key=total.get
    )

    return (
        pred,
        total,
        details
    )


# ============================================================
# 17. TÌM TEST UTTERANCE CÓ TOP-1/TOP-2 GẦN NHAU
# ============================================================

def find_ambiguous_test(models):
    best = None

    best_gap = np.inf

    for true_label in LABELS:

        for wav in sorted(
            (TEST_DIR / true_label).glob("*.wav")
        ):

            F = extract_feat(wav)

            _, scores = recognize_feature(
                F,
                models
            )

            ranked = sorted(
                scores.values(),
                reverse=True
            )

            if len(ranked) < 2:
                continue

            gap = (
                ranked[0]
                - ranked[1]
            ) / len(F)

            if gap < best_gap:

                best_gap = gap

                best = (
                    wav,
                    true_label,
                    F,
                    scores
                )

    return best


def run_bigram_experiment(models):
    print("\n" + "=" * 70)
    print("BIGRAM LANGUAGE MODEL RE-RANKING")
    print("=" * 70)

    create_demo_corpus()

    corpus = load_corpus(
        CORPUS_PATH
    )

    counts1, counts2 = (
        build_ngram_counts(corpus)
    )

    print("Unigram counts:")
    print(counts1)

    print("\nBigram counts:")
    print(counts2)

    candidate = find_ambiguous_test(
        models
    )

    if candidate is None:

        print(
            "Không tìm được test utterance "
            "để thử LM."
        )

        return

    wav, true_label, F, acoustic_scores = (
        candidate
    )

    print(
        "\nUtterance có top-1/top-2 gần nhất:"
    )

    print("File:", wav)
    print("True:", true_label)

    print(
        "Previous word context:",
        PREVIOUS_WORD_FOR_LM
    )

    output_rows = []

    for lm_weight in [
        0.0,
        0.1,
        0.3,
        1.0
    ]:

        pred, total, details = rerank(
            F,
            PREVIOUS_WORD_FOR_LM,
            models,
            counts2,
            counts1,
            lm_weight=lm_weight
        )

        print(
            f"\nlambda_LM = "
            f"{lm_weight}"
        )

        print(
            "Prediction =",
            pred
        )

        ranked = sorted(
            total.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for word, score in ranked:

            print(
                f"  {word:8s} "
                f"combined={score:10.4f} "
                f"acoustic/frame="
                f"{details[word]['acoustic_per_frame']:10.4f} "
                f"LM="
                f"{details[word]['lm_log_prob']:8.4f}"
            )

        output_rows.append(
            [
                lm_weight,
                wav.name,
                true_label,
                pred
            ]
        )

    with open(
        OUTPUT_DIR / "bigram_reranking.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "lambda_LM",
            "file",
            "true",
            "prediction"
        ])

        writer.writerows(
            output_rows
        )


# ============================================================
# 18. ROBUSTNESS - THÊM NOISE NHẸ
# ============================================================

def add_noise_snr(
    y,
    snr_db=20,
    random_state=0
):
    rng = np.random.default_rng(
        random_state
    )

    signal_power = np.mean(
        y ** 2
    ) + 1e-12

    noise = rng.normal(
        0,
        1,
        len(y)
    )

    noise_power = np.mean(
        noise ** 2
    ) + 1e-12

    desired_noise_power = (
        signal_power
        / (10 ** (snr_db / 10))
    )

    noise = noise * np.sqrt(
        desired_noise_power
        / noise_power
    )

    return (
        y + noise
    ).astype(np.float32)


def evaluate_noise_robustness(
    models,
    snr_db=20
):
    true_y = []
    pred_y = []

    for true_label in LABELS:

        files = sorted(
            (TEST_DIR / true_label).glob("*.wav")
        )

        for wav in files:

            y, _ = librosa.load(
                wav,
                sr=SR,
                mono=True
            )

            noisy = add_noise_snr(
                y,
                snr_db=snr_db,
                random_state=0
            )

            F = compute_features_from_audio(
                noisy,
                sr=SR,
                use_dynamic=True
            )

            pred, _ = recognize_feature(
                F,
                models
            )

            true_y.append(
                true_label
            )

            pred_y.append(
                pred
            )

    acc = accuracy_score(
        true_y,
        pred_y
    )

    cm = confusion_matrix(
        true_y,
        pred_y,
        labels=LABELS
    )

    print("\n" + "=" * 70)

    print(
        f"ROBUSTNESS TEST - "
        f"NOISE {snr_db} dB"
    )

    print("=" * 70)

    print(
        f"Accuracy = "
        f"{acc * 100:.2f}%"
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        f"HMM - noise SNR={snr_db} dB"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"confusion_noise_{snr_db}dB.png",
        dpi=150
    )

    plt.close()

    return float(acc)


# ============================================================
# 19. PHÂN BỐ SCORE ĐÚNG / SAI + REJECT THRESHOLD
# ============================================================

def score_distribution(models):
    correct_scores = []
    wrong_scores = []

    margins = []

    for true_label in LABELS:

        files = sorted(
            (TEST_DIR / true_label).glob("*.wav")
        )

        for wav in files:

            F = extract_feat(wav)

            scores = {
                label:
                model.score(F) / len(F)

                for label, model
                in models.items()
            }

            correct_scores.append(
                scores[true_label]
            )

            incorrect = [
                score
                for label, score
                in scores.items()
                if label != true_label
            ]

            if incorrect:

                wrong_scores.append(
                    max(incorrect)
                )

            ranked = sorted(
                scores.values(),
                reverse=True
            )

            if len(ranked) >= 2:
                margins.append(
                    ranked[0]
                    - ranked[1]
                )

    if not correct_scores:
        return None

    correct_scores = np.asarray(
        correct_scores
    )

    wrong_scores = np.asarray(
        wrong_scores
    )

    margins = np.asarray(
        margins
    )

    if len(wrong_scores) > 0:

        threshold = (
            np.percentile(
                correct_scores,
                5
            )
            +
            np.percentile(
                wrong_scores,
                95
            )
        ) / 2

    else:
        threshold = np.percentile(
            correct_scores,
            5
        )

    margin_threshold = (
        np.percentile(
            margins,
            5
        )
        if len(margins)
        else 0.0
    )

    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION / UNKNOWN REJECT")
    print("=" * 70)

    print(
        "Mean correct score/frame:",
        float(correct_scores.mean())
    )

    if len(wrong_scores):

        print(
            "Mean strongest wrong score/frame:",
            float(wrong_scores.mean())
        )

    print(
        "Gợi ý normalized likelihood threshold:",
        float(threshold)
    )

    print(
        "Gợi ý top1-top2 margin threshold:",
        float(margin_threshold)
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        correct_scores,
        bins=10,
        alpha=0.6,
        label="Correct model"
    )

    if len(wrong_scores):

        plt.hist(
            wrong_scores,
            bins=10,
            alpha=0.6,
            label="Best wrong model"
        )

    plt.axvline(
        threshold,
        linestyle="--",
        label="Suggested threshold"
    )

    plt.xlabel(
        "Log-likelihood / frame"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Correct vs wrong score distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "score_distribution.png",
        dpi=150
    )

    plt.close()

    return {
        "score_threshold":
            float(threshold),

        "margin_threshold":
            float(margin_threshold)
    }


# ============================================================
# 20. UNKNOWN / REJECT DEMO
# ============================================================

def recognize_with_reject(
    path,
    models,
    score_threshold=None,
    margin_threshold=None
):
    F = extract_feat(path)

    scores = {
        label:
        model.score(F) / len(F)

        for label, model
        in models.items()
    }

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top1_label, top1_score = (
        ranked[0]
    )

    top2_score = (
        ranked[1][1]
        if len(ranked) > 1
        else -np.inf
    )

    margin = (
        top1_score
        - top2_score
    )

    if (
        score_threshold is not None
        and top1_score < score_threshold
    ):
        return "unknown", scores

    if (
        margin_threshold is not None
        and margin < margin_threshold
    ):
        return "unknown", scores

    return top1_label, scores


# ============================================================
# 21. THÍ NGHIỆM E1
#     n_states = 3, 5, 7
# ============================================================

def experiment_states():
    print("\n" + "=" * 70)
    print("E1 - n_states = 3, 5, 7")
    print("=" * 70)

    rows = []

    for n_states in [
        3,
        5,
        7
    ]:

        try:

            models, train_time = (
                train_models(
                    n_states=n_states,
                    use_dynamic=True,
                    covariance_type="diag",
                    verbose=False
                )
            )

            result = evaluate_models(
                models,
                use_dynamic=True,
                title=f"E1 - HMM {n_states} states",
                save_name=(
                    f"E1_cm_{n_states}_states.png"
                )
            )

            convergence = all(
                model.monitor_.converged
                for model
                in models.values()
            )

            rows.append([
                n_states,
                result["accuracy"],
                convergence,
                train_time
            ])

            print(
                f"states={n_states}: "
                f"accuracy="
                f"{result['accuracy'] * 100:.2f}% "
                f"converged={convergence}"
            )

        except Exception as e:

            print(
                f"E1 states={n_states} lỗi:",
                e
            )

            rows.append([
                n_states,
                "ERROR",
                False,
                "ERROR"
            ])

    with open(
        OUTPUT_DIR
        / "E1_states.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "n_states",
            "accuracy",
            "all_converged",
            "train_time"
        ])

        writer.writerows(rows)


# ============================================================
# 22. THÍ NGHIỆM E2
#     MFCC 13 VS MFCC + DELTA + DELTA2
# ============================================================

def experiment_dynamic_features():
    print("\n" + "=" * 70)
    print("E2 - MFCC 13 VS 39-D DYNAMIC MFCC")
    print("=" * 70)

    rows = []

    for use_dynamic, name in [
        (False, "MFCC13"),
        (True, "MFCC39")
    ]:

        try:

            models, train_time = (
                train_models(
                    n_states=5,
                    use_dynamic=use_dynamic,
                    covariance_type="diag",
                    verbose=False
                )
            )

            result = evaluate_models(
                models,
                use_dynamic=use_dynamic,
                title=f"E2 - {name}",
                save_name=f"E2_{name}.png"
            )

            rows.append([
                name,
                result["accuracy"],
                train_time
            ])

            print(
                name,
                "accuracy =",
                f"{result['accuracy'] * 100:.2f}%"
            )

        except Exception as e:

            print(
                name,
                "ERROR:",
                e
            )

            rows.append([
                name,
                "ERROR",
                "ERROR"
            ])

    with open(
        OUTPUT_DIR
        / "E2_features.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "feature",
            "accuracy",
            "train_time"
        ])

        writer.writerows(rows)


# ============================================================
# 23. THÍ NGHIỆM E3
#     DIAG VS FULL COVARIANCE
# ============================================================

def experiment_covariance():
    print("\n" + "=" * 70)
    print("E3 - DIAG VS FULL COVARIANCE")
    print("=" * 70)

    rows = []

    for covariance in [
        "diag",
        "full"
    ]:

        try:

            models, train_time = (
                train_models(
                    n_states=5,
                    use_dynamic=True,
                    covariance_type=covariance,
                    verbose=False
                )
            )

            result = evaluate_models(
                models,
                use_dynamic=True,
                title=(
                    f"E3 - covariance={covariance}"
                ),
                save_name=(
                    f"E3_cov_{covariance}.png"
                )
            )

            rows.append([
                covariance,
                result["accuracy"],
                train_time
            ])

            print(
                covariance,
                "accuracy =",
                f"{result['accuracy'] * 100:.2f}%"
            )

        except Exception as e:

            print(
                f"Full/diag experiment lỗi "
                f"({covariance}):",
                e
            )

            rows.append([
                covariance,
                "ERROR",
                "ERROR"
            ])

    with open(
        OUTPUT_DIR
        / "E3_covariance.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "covariance",
            "accuracy",
            "train_time"
        ])

        writer.writerows(rows)


# ============================================================
# 24. E5 - SPEAKER / ENVIRONMENT MISMATCH
# Ở đây dùng noise nhẹ làm mismatch condition.
# ============================================================

def experiment_mismatch(
    models,
    clean_accuracy
):
    noisy_accuracy = (
        evaluate_noise_robustness(
            models,
            snr_db=NOISE_SNR_DB
        )
    )

    accuracy_drop = (
        clean_accuracy
        - noisy_accuracy
    )

    with open(
        OUTPUT_DIR
        / "E5_mismatch.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "condition",
            "accuracy"
        ])

        writer.writerow([
            "clean",
            clean_accuracy
        ])

        writer.writerow([
            f"noise_{NOISE_SNR_DB}dB",
            noisy_accuracy
        ])

        writer.writerow([
            "accuracy_drop",
            accuracy_drop
        ])

    print(
        f"\nAccuracy drop do noise = "
        f"{accuracy_drop * 100:.2f} "
        f"percentage points"
    )


# ============================================================
# 25. MỤC 9 - MỞ RỘNG GMM-HMM
# ============================================================

def make_gmm_lr_hmm(
    n_states=5,
    n_mix=2
):
    if GMMHMM is None:
        raise RuntimeError(
            "Phiên bản hmmlearn này không "
            "có GMMHMM."
        )

    model = GMMHMM(
        n_components=n_states,
        n_mix=n_mix,
        covariance_type="diag",
        n_iter=N_ITER,
        tol=1e-3,
        init_params="mcw",
        params="mcw",
        random_state=0
    )

    start = np.zeros(
        n_states
    )

    start[0] = 1.0

    A = np.zeros(
        (n_states, n_states)
    )

    for i in range(
        n_states - 1
    ):
        A[i, i] = 0.7
        A[i, i + 1] = 0.3

    A[-1, -1] = 1.0

    model.startprob_ = start
    model.transmat_ = A

    return model


def run_gmm_extension():
    print("\n" + "=" * 70)
    print("MỤC 9 - GMM-HMM EXTENSION")
    print("=" * 70)

    if GMMHMM is None:

        print(
            "GMMHMM không khả dụng "
            "trong hmmlearn hiện tại."
        )

        return

    models = {}

    for label in LABELS:

        feats = []
        lengths = []

        for wav in sorted(
            (TRAIN_DIR / label).glob("*.wav")
        ):

            F = extract_feat(wav)

            feats.append(F)

            lengths.append(
                len(F)
            )

        X = np.vstack(feats)

        model = make_gmm_lr_hmm(
            n_states=5,
            n_mix=2
        )

        model.fit(
            X,
            lengths
        )

        models[label] = model

    result = evaluate_models(
        models,
        use_dynamic=True,
        title="Mục 9 - 2-mixture GMM-HMM",
        save_name="extension_gmm_hmm.png"
    )

    print(
        "GMM-HMM accuracy:",
        result["accuracy"]
    )


# ============================================================
# 26. LƯU SUMMARY KẾT QUẢ
# ============================================================

def save_summary(
    hmm_result,
    hmm_train_time,
    dtw_result,
    reject_info
):
    output = (
        OUTPUT_DIR
        / "summary.txt"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "LAB 3 - HMM SPEECH RECOGNITION\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"HMM accuracy: "
            f"{hmm_result['accuracy'] * 100:.2f}%\n"
        )

        f.write(
            f"HMM train time: "
            f"{hmm_train_time:.4f}s\n"
        )

        f.write(
            f"HMM test time/trial: "
            f"{hmm_result['test_time']:.4f}s\n\n"
        )

        f.write(
            f"DTW accuracy: "
            f"{dtw_result['accuracy'] * 100:.2f}%\n"
        )

        f.write(
            f"DTW test time/trial: "
            f"{dtw_result['test_time']:.4f}s\n\n"
        )

        if reject_info:

            f.write(
                f"Suggested normalized "
                f"likelihood threshold: "
                f"{reject_info['score_threshold']:.6f}\n"
            )

            f.write(
                f"Suggested top1-top2 "
                f"margin threshold: "
                f"{reject_info['margin_threshold']:.6f}\n"
            )

    print(
        "\nSummary saved:",
        output
    )


# ============================================================
# 27. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CSE457 - LAB 3")
    print("NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM")
    print("=" * 70)

    # --------------------------------------------------------
    # A. Chuẩn hóa dữ liệu và đặc trưng
    # --------------------------------------------------------

    check_dataset()

    inspect_feature_shapes(
        n_files=3
    )

    generate_heatmaps()

    # --------------------------------------------------------
    # B + C.
    # HMM left-to-right
    # Huấn luyện một model cho mỗi từ
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HUẤN LUYỆN 5 WHOLE-WORD HMM")
    print("=" * 70)

    models, hmm_train_time = (
        train_models(
            n_states=N_STATES,
            use_dynamic=True,
            covariance_type="diag",
            verbose=True,
            save_models=True
        )
    )

    print(
        f"\nHMM training time = "
        f"{hmm_train_time:.4f}s"
    )

    # --------------------------------------------------------
    # D. Maximum likelihood recognition
    # --------------------------------------------------------

    generate_score_table(
        models,
        n_test_files=3
    )

    # Viterbi alignment ít nhất 2 file
    generate_viterbi_examples(
        models
    )

    # --------------------------------------------------------
    # H. Accuracy + confusion matrix sạch
    # --------------------------------------------------------

    hmm_result = evaluate_models(
        models,
        use_dynamic=True,
        title="Whole-word Gaussian HMM",
        save_name="confusion_matrix_hmm.png"
    )

    # --------------------------------------------------------
    # E. DTW baseline
    # --------------------------------------------------------

    templates = (
        prepare_dtw_templates()
    )

    dtw_result = evaluate_dtw(
        templates
    )

    save_dtw_hmm_comparison(
        hmm_result,
        hmm_train_time,
        dtw_result
    )

    # --------------------------------------------------------
    # F. Lexicon mini
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LEXICON")
    print("=" * 70)

    create_lexicon_template()

    lexicon = parse_lexicon(
        LEXICON_PATH
    )

    print(
        "Lexicon hiện tại:"
    )

    if lexicon:
        for word, phones in lexicon.items():
            print(
                word,
                "->",
                phones
            )
    else:
        print(
            "Chưa có phone sequence. "
            "Hãy bổ sung theo inventory "
            "do giảng viên cung cấp."
        )

    # --------------------------------------------------------
    # G / E6. Bigram + MAP re-ranking
    # --------------------------------------------------------

    run_bigram_experiment(
        models
    )

    # --------------------------------------------------------
    # H / E5. Robustness
    # --------------------------------------------------------

    experiment_mismatch(
        models,
        clean_accuracy=hmm_result[
            "accuracy"
        ]
    )

    # Score distribution + reject
    reject_info = score_distribution(
        models
    )

    # --------------------------------------------------------
    # Thí nghiệm bắt buộc E1, E2, E3
    # E4 = DTW vs HMM đã chạy
    # E5 = mismatch đã chạy
    # E6 = bigram đã chạy
    # --------------------------------------------------------

    if RUN_EXPERIMENTS:

        experiment_states()

        experiment_dynamic_features()

        experiment_covariance()

    # --------------------------------------------------------
    # Mục 9 - mở rộng GMM-HMM
    # --------------------------------------------------------

    if RUN_GMM_EXTENSION:

        try:
            run_gmm_extension()

        except Exception as e:

            print(
                "\nGMM extension lỗi:",
                e
            )

    # --------------------------------------------------------
    # Tổng kết
    # --------------------------------------------------------

    save_summary(
        hmm_result,
        hmm_train_time,
        dtw_result,
        reject_info
    )

    print("\n")
    print("=" * 70)
    print("LAB 3 HOÀN THÀNH")
    print("=" * 70)

    print(
        "\nKiểm tra các thư mục:"
    )

    print(
        "models/  -> "
        "các model HMM đã lưu"
    )

    print(
        "outputs/ -> "
        "confusion matrix, "
        "Viterbi plots, "
        "heatmaps, CSV..."
    )

    print(
        "\nCác file quan trọng:"
    )

    print(
        "outputs/confusion_matrix_hmm.png"
    )

    print(
        "outputs/confusion_matrix_dtw.png"
    )

    print(
        "outputs/acoustic_scores.csv"
    )

    print(
        "outputs/dtw_vs_hmm.csv"
    )

    print(
        "outputs/bigram_reranking.csv"
    )

    print(
        "outputs/E1_states.csv"
    )

    print(
        "outputs/E2_features.csv"
    )

    print(
        "outputs/E3_covariance.csv"
    )

    print(
        "outputs/E5_mismatch.csv"
    )

    print(
        "outputs/score_distribution.png"
    )

    print(
        "outputs/summary.txt"
    )


# ============================================================
# CHẠY CHƯƠNG TRÌNH
# ============================================================

if __name__ == "__main__":
    main()