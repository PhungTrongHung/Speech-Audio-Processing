from pathlib import Path
import csv

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from part_d import load_audio, mfcc_feature
from part_e import dtw_distance


ROOT = Path(__file__).resolve().parent

TRAIN_CSV = ROOT / "splits" / "train.csv"
TEST_CSV = ROOT / "splits" / "test.csv"

DATA_DIR = ROOT / "dataset"
TRIMMED_DIR = ROOT / "dataset_trimmed"

RESULT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


EXPERIMENTS = [
    {
        "id": "baseline_trim_mfcc13",
        "name": "Trim + MFCC13",
        "use_trim": True,
        "use_delta": False,
    },
    {
        "id": "e1_no_trim_mfcc13",
        "name": "No trim + MFCC13",
        "use_trim": False,
        "use_delta": False,
    },
    {
        "id": "e2_trim_mfcc13_delta",
        "name": "Trim + MFCC13 + Delta",
        "use_trim": True,
        "use_delta": True,
    },
]


def read_split(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {csv_path}"
        )

    rows = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append({
                "label": row["label"],
                "path": row["path"],
            })

    return rows


def resolve_audio_path(row, use_trim):
    """
    train.csv và test.csv đang chứa đường dẫn tới dataset/.
    Nếu use_trim=True thì đổi sang dataset_trimmed/.
    """
    original_path = Path(row["path"])

    if original_path.is_absolute():
        untrimmed_path = original_path
    else:
        untrimmed_path = ROOT / original_path

    if use_trim:
        audio_path = (
            TRIMMED_DIR
            / row["label"]
            / original_path.name
        )
    else:
        audio_path = untrimmed_path

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {audio_path}"
        )

    return audio_path


def check_data_leakage(train_rows, test_rows):
    train_keys = {
        (row["label"], Path(row["path"]).name)
        for row in train_rows
    }

    test_keys = {
        (row["label"], Path(row["path"]).name)
        for row in test_rows
    }

    overlap = train_keys & test_keys

    if overlap:
        raise ValueError(
            f"Data leakage: {sorted(overlap)}"
        )

    print("Kiểm tra data leakage: đạt.")


def add_delta(mfcc):
    """
    mfcc có shape (T, 13).
    Librosa tính delta theo trục thời gian nên cần chuyển
    tạm thành (13, T).
    """
    number_of_frames = mfcc.shape[0]

    # Width phải là số lẻ và không lớn hơn số frame.
    width = min(9, number_of_frames)

    if width % 2 == 0:
        width -= 1

    if width < 3:
        delta = np.zeros_like(mfcc)
    else:
        delta = librosa.feature.delta(
            mfcc.T,
            width=width,
            order=1,
            axis=-1,
            mode="nearest"
        ).T

    combined = np.hstack([
        mfcc,
        delta
    ])

    # 13 MFCC + 13 delta = 26 chiều.
    if combined.shape[1] != 26:
        raise ValueError(
            f"MFCC+Delta phải có 26 chiều, "
            f"nhận được {combined.shape}"
        )

    return combined.astype(np.float32)


def extract_feature(audio_path, use_delta):
    """
    Giữ nguyên pipeline MFCC của phần D.
    Chỉ thêm delta khi thí nghiệm yêu cầu.
    """
    y = load_audio(audio_path)
    feature = mfcc_feature(y)

    if use_delta:
        feature = add_delta(feature)

    if feature.ndim != 2:
        raise ValueError(
            f"Feature không hợp lệ: {feature.shape}"
        )

    if not np.all(np.isfinite(feature)):
        raise ValueError(
            f"{audio_path.name}: feature chứa NaN/Inf."
        )

    return feature


def build_feature_cache(
    rows,
    use_trim,
    use_delta
):
    """
    Trích feature một lần cho mỗi file trong một thí nghiệm,
    tránh tính lặp lại nhiều lần khi chạy DTW.
    """
    cache = {}

    for row in rows:
        audio_path = resolve_audio_path(
            row,
            use_trim
        )

        key = (
            row["label"],
            audio_path.name
        )

        cache[key] = {
            "path": audio_path,
            "feature": extract_feature(
                audio_path,
                use_delta
            ),
        }

    return cache


def build_templates(
    train_rows,
    feature_cache,
    use_trim
):
    templates = {}

    for row in train_rows:
        audio_path = resolve_audio_path(
            row,
            use_trim
        )

        key = (
            row["label"],
            audio_path.name
        )

        templates.setdefault(
            row["label"],
            []
        ).append(
            feature_cache[key]
        )

    return templates


def recognize_feature(test_feature, templates):
    """
    Điểm của một nhãn là khoảng cách nhỏ nhất
    trong tất cả template của nhãn đó.
    """
    ranking = []

    for label, references in templates.items():
        scores = []

        for reference in references:
            dtw_norm = dtw_distance(
                test_feature,
                reference["feature"]
            )[0]

            scores.append({
                "score": float(dtw_norm),
                "template": reference["path"].name,
            })

        best_reference = min(
            scores,
            key=lambda item: item["score"]
        )

        ranking.append({
            "label": label,
            "score": best_reference["score"],
            "template": best_reference["template"],
        })

    ranking.sort(
        key=lambda item: item["score"]
    )

    prediction = ranking[0]["label"]

    return prediction, ranking


def run_experiment(
    experiment,
    train_rows,
    test_rows,
    labels
):
    print("\n" + "=" * 70)
    print("Thí nghiệm:", experiment["name"])
    print("Endpoint:", experiment["use_trim"])
    print("Delta:", experiment["use_delta"])

    all_rows = train_rows + test_rows

    feature_cache = build_feature_cache(
        all_rows,
        experiment["use_trim"],
        experiment["use_delta"]
    )

    templates = build_templates(
        train_rows,
        feature_cache,
        experiment["use_trim"]
    )

    predictions = []
    y_true = []
    y_pred = []

    for index, row in enumerate(
        test_rows,
        start=1
    ):
        audio_path = resolve_audio_path(
            row,
            experiment["use_trim"]
        )

        key = (
            row["label"],
            audio_path.name
        )

        test_feature = feature_cache[key][
            "feature"
        ]

        prediction, ranking = recognize_feature(
            test_feature,
            templates
        )

        top1, top2, top3 = ranking[:3]

        y_true.append(row["label"])
        y_pred.append(prediction)

        predictions.append({
            "experiment": experiment["id"],
            "experiment_name": experiment["name"],
            "file": audio_path.name,
            "true_label": row["label"],
            "predicted_label": prediction,
            "correct": prediction == row["label"],
            "trim": experiment["use_trim"],
            "delta": experiment["use_delta"],
            "feature_dimension": (
                test_feature.shape[1]
            ),
            "number_of_frames": (
                test_feature.shape[0]
            ),
            "top1_label": top1["label"],
            "top1_score": round(
                top1["score"], 6
            ),
            "top1_template": top1["template"],
            "top2_label": top2["label"],
            "top2_score": round(
                top2["score"], 6
            ),
            "top2_template": top2["template"],
            "top3_label": top3["label"],
            "top3_score": round(
                top3["score"], 6
            ),
            "top3_template": top3["template"],
        })

        print(
            f"[{index}/{len(test_rows)}] "
            f"{audio_path.name:<22} "
            f"true={row['label']:<12} "
            f"pred={prediction:<12} "
            f"score={top1['score']:.4f}"
        )

    correct_count = sum(
        true == pred
        for true, pred in zip(y_true, y_pred)
    )

    accuracy = (
        correct_count / len(y_true)
        if y_true
        else 0.0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print(
        f"Accuracy: "
        f"{correct_count}/{len(y_true)} "
        f"= {accuracy * 100:.2f}%"
    )

    return {
        "experiment": experiment,
        "predictions": predictions,
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(y_true),
        "confusion_matrix": matrix,
    }


def write_csv(output_path, rows):
    if not rows:
        raise ValueError(
            f"Không có dữ liệu để lưu vào {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
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

    print("Đã lưu:", output_path)


def save_all_results(experiment_results):
    all_predictions = []

    for result in experiment_results:
        all_predictions.extend(
            result["predictions"]
        )

    write_csv(
        RESULT_DIR / "G_all_predictions.csv",
        all_predictions
    )

    baseline_accuracy = (
        experiment_results[0]["accuracy"]
    )

    summary_rows = []

    for result in experiment_results:
        experiment = result["experiment"]

        summary_rows.append({
            "experiment": experiment["id"],
            "name": experiment["name"],
            "trim": experiment["use_trim"],
            "delta": experiment["use_delta"],
            "feature_dimension": (
                26 if experiment["use_delta"] else 13
            ),
            "correct": result["correct"],
            "total": result["total"],
            "accuracy": round(
                result["accuracy"], 6
            ),
            "accuracy_percent": round(
                result["accuracy"] * 100,
                2
            ),
            "difference_vs_baseline_pp": round(
                (
                    result["accuracy"]
                    - baseline_accuracy
                ) * 100,
                2
            ),
        })

    write_csv(
        RESULT_DIR / "G_experiment_summary.csv",
        summary_rows
    )

    # Sản phẩm results.csv theo yêu cầu nộp của Lab:
    # dùng cấu hình baseline.
    baseline_rows = []

    for row in experiment_results[0][
        "predictions"
    ]:
        baseline_rows.append({
            "file": row["file"],
            "true_label": row["true_label"],
            "predicted_label": (
                row["predicted_label"]
            ),
            "top1_label": row["top1_label"],
            "top1_score": row["top1_score"],
            "top2_label": row["top2_label"],
            "top2_score": row["top2_score"],
        })

    write_csv(
        ROOT / "results.csv",
        baseline_rows
    )


def plot_confusion_matrices(
    experiment_results,
    labels
):
    fig, axes = plt.subplots(
        1,
        len(experiment_results),
        figsize=(18, 5.5),
        layout="constrained"
    )

    maximum_count = max(
        int(np.max(result["confusion_matrix"]))
        for result in experiment_results
    )

    maximum_count = max(maximum_count, 1)

    last_image = None

    for axis, result in zip(
        axes,
        experiment_results
    ):
        matrix = result["confusion_matrix"]

        last_image = axis.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
            vmax=maximum_count
        )

        axis.set_title(
            f"{result['experiment']['name']}\n"
            f"Accuracy = "
            f"{result['accuracy'] * 100:.2f}%"
        )

        axis.set_xlabel("Nhãn dự đoán")
        axis.set_ylabel("Nhãn thật")

        axis.set_xticks(
            range(len(labels)),
            labels=labels,
            rotation=45,
            ha="right"
        )

        axis.set_yticks(
            range(len(labels)),
            labels=labels
        )

        threshold = maximum_count / 2

        for i in range(len(labels)):
            for j in range(len(labels)):
                value = int(matrix[i, j])

                axis.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if value > threshold
                        else "black"
                    )
                )

    fig.colorbar(
        last_image,
        ax=axes,
        label="Số lượng mẫu"
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FIG_DIR / "G_confusion_matrices.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Đã lưu:", output_path)


def plot_accuracy_comparison(experiment_results):
    names = [
        result["experiment"]["name"]
        for result in experiment_results
    ]

    accuracies = [
        result["accuracy"] * 100
        for result in experiment_results
    ]

    colors = [
        "steelblue",
        "darkorange",
        "seagreen"
    ]

    fig, axis = plt.subplots(
        figsize=(10, 6),
        layout="constrained"
    )

    bars = axis.bar(
        names,
        accuracies,
        color=colors
    )

    axis.set_ylabel("Accuracy (%)")
    axis.set_title(
        "So sánh các cấu hình nhận dạng"
    )
    axis.set_ylim(
        0,
        max(100, max(accuracies) + 10)
    )
    axis.grid(
        axis="y",
        alpha=0.25
    )

    for bar, accuracy in zip(
        bars,
        accuracies
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom"
        )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FIG_DIR / "G_accuracy_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Đã lưu:", output_path)


def print_comparison(experiment_results):
    baseline = experiment_results[0]
    no_trim = experiment_results[1]
    with_delta = experiment_results[2]

    endpoint_gain = (
        baseline["accuracy"]
        - no_trim["accuracy"]
    ) * 100

    delta_gain = (
        with_delta["accuracy"]
        - baseline["accuracy"]
    ) * 100

    print("\n" + "=" * 70)
    print("TỔNG HỢP THÍ NGHIỆM")

    for result in experiment_results:
        print(
            f"{result['experiment']['name']:<28}: "
            f"{result['accuracy'] * 100:.2f}% "
            f"({result['correct']}/{result['total']})"
        )

    print(
        "\nẢnh hưởng endpoint detection: "
        f"{endpoint_gain:+.2f} điểm phần trăm"
    )

    print(
        "Ảnh hưởng Delta: "
        f"{delta_gain:+.2f} điểm phần trăm"
    )

    if endpoint_gain > 0:
        print(
            "Endpoint detection cải thiện accuracy."
        )
    elif endpoint_gain < 0:
        print(
            "Endpoint detection làm giảm accuracy; "
            "hãy kiểm tra khả năng cắt mất âm."
        )
    else:
        print(
            "Endpoint detection chưa làm thay đổi accuracy."
        )

    if delta_gain > 0:
        print(
            "Đặc trưng Delta cải thiện accuracy."
        )
    elif delta_gain < 0:
        print(
            "Delta làm giảm accuracy trên tập dữ liệu này."
        )
    else:
        print(
            "Delta chưa làm thay đổi accuracy."
        )


def main():
    train_rows = read_split(TRAIN_CSV)
    test_rows = read_split(TEST_CSV)

    check_data_leakage(
        train_rows,
        test_rows
    )

    labels = sorted({
        row["label"]
        for row in train_rows + test_rows
    })

    print("Các nhãn:", labels)
    print("Số file train:", len(train_rows))
    print("Số file test :", len(test_rows))

    experiment_results = []

    for experiment in EXPERIMENTS:
        result = run_experiment(
            experiment,
            train_rows,
            test_rows,
            labels
        )

        experiment_results.append(result)

    save_all_results(experiment_results)

    plot_confusion_matrices(
        experiment_results,
        labels
    )

    plot_accuracy_comparison(
        experiment_results
    )

    print_comparison(experiment_results)

    print("\nHoàn thành phần G.")


if __name__ == "__main__":
    main()