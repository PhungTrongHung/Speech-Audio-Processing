from pathlib import Path
import csv

from part_d import load_audio, mfcc_feature
from part_e import dtw_distance


ROOT = Path(__file__).resolve().parent

TRAIN_CSV = ROOT / "splits" / "train.csv"
TEST_CSV = ROOT / "splits" / "test.csv"
TRIMMED_DIR = ROOT / "dataset_trimmed"
RESULT_DIR = ROOT / "results"

# Để None nếu chưa sử dụng reject.
# Ví dụ: REJECT_THRESHOLD = 25.0
REJECT_THRESHOLD = None


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
            label = row["label"]
            original_name = Path(row["path"]).name

            trimmed_path = (
                TRIMMED_DIR
                / label
                / original_name
            )

            if not trimmed_path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy file đã trim: "
                    f"{trimmed_path}. "
                    "Hãy kiểm tra phần C."
                )

            rows.append({
                "label": label,
                "path": trimmed_path,
            })

    return rows


def check_data_leakage(train_rows, test_rows):
    train_paths = {
        row["path"].resolve()
        for row in train_rows
    }

    test_paths = {
        row["path"].resolve()
        for row in test_rows
    }

    overlap = train_paths & test_paths

    if overlap:
        names = "\n".join(
            str(path) for path in sorted(overlap)
        )

        raise ValueError(
            "Phát hiện data leakage. Các file sau "
            f"có trong cả train và test:\n{names}"
        )

    print("Kiểm tra data leakage: đạt.")


def extract_feature(wav_path):
    """
    Pipeline cho cả train và test phải giống nhau:
      load WAV -> normalize -> pre-emphasis -> MFCC -> CMN
    """
    y = load_audio(wav_path)
    feature = mfcc_feature(y)

    if feature.ndim != 2:
        raise ValueError(
            f"{wav_path.name}: feature phải có 2 chiều."
        )

    if feature.shape[1] != 13:
        raise ValueError(
            f"{wav_path.name}: shape={feature.shape}; "
            "cần dạng (T,13)."
        )

    return feature


def build_templates(train_rows):
    """
    templates có dạng:
    {
        label_1: [
            {"path": ..., "feature": ...},
            ...
        ],
        ...
    }
    """
    templates = {}

    for row in train_rows:
        label = row["label"]
        wav_path = row["path"]

        feature = extract_feature(wav_path)

        templates.setdefault(label, []).append({
            "path": wav_path,
            "feature": feature,
        })

        print(
            f"Template: {label}/{wav_path.name}, "
            f"shape={feature.shape}"
        )

    if len(templates) < 3:
        raise ValueError(
            "Cần ít nhất ba nhãn để tạo top-3."
        )

    for label, references in templates.items():
        if not references:
            raise ValueError(
                f"Nhãn {label} không có template."
            )

        print(
            f"Nhãn '{label}': "
            f"{len(references)} template"
        )

    return templates


def recognize(test_path, templates):
    """
    Với mỗi nhãn:
      - tính DTW đến tất cả template của nhãn đó;
      - lấy khoảng cách nhỏ nhất.

    Trả về nhãn dự đoán và danh sách xếp hạng.
    """
    test_feature = extract_feature(test_path)

    label_results = []

    for label, references in templates.items():
        template_scores = []

        for reference in references:
            normalized_cost = dtw_distance(
                test_feature,
                reference["feature"]
            )[0]

            template_scores.append({
                "score": float(normalized_cost),
                "template_path": reference["path"],
            })

        # Template tốt nhất của nhãn.
        best_template = min(
            template_scores,
            key=lambda item: item["score"]
        )

        label_results.append({
            "label": label,
            "score": best_template["score"],
            "best_template": best_template[
                "template_path"
            ],
        })

    # DTW nhỏ hơn nghĩa là giống hơn.
    ranking = sorted(
        label_results,
        key=lambda item: item["score"]
    )

    best_label = ranking[0]["label"]
    best_score = ranking[0]["score"]

    if (
        REJECT_THRESHOLD is not None
        and best_score > REJECT_THRESHOLD
    ):
        prediction = "unknown"
    else:
        prediction = best_label

    return prediction, ranking, test_feature


def evaluate_test_files(test_rows, templates):
    result_rows = []
    number_correct = 0

    for index, row in enumerate(
        test_rows,
        start=1
    ):
        true_label = row["label"]
        test_path = row["path"]

        prediction, ranking, test_feature = recognize(
            test_path,
            templates
        )

        if len(ranking) < 3:
            raise ValueError(
                "Không đủ ba nhãn để tạo top-3."
            )

        top1, top2, top3 = ranking[:3]
        correct = prediction == true_label

        if correct:
            number_correct += 1

        print("\n" + "=" * 65)
        print(
            f"[{index}/{len(test_rows)}] "
            f"{test_path.name}"
        )
        print("Nhãn thật :", true_label)
        print("Dự đoán  :", prediction)
        print("MFCC     :", test_feature.shape)
        print(
            f"Top 1     : {top1['label']:<15} "
            f"{top1['score']:.6f}"
        )
        print(
            f"Top 2     : {top2['label']:<15} "
            f"{top2['score']:.6f}"
        )
        print(
            f"Top 3     : {top3['label']:<15} "
            f"{top3['score']:.6f}"
        )
        print(
            "Kết quả   :",
            "Đúng" if correct else "Sai"
        )

        result_rows.append({
            "file": test_path.name,
            "true_label": true_label,
            "predicted_label": prediction,
            "correct": correct,
            "number_of_frames": (
                test_feature.shape[0]
            ),

            "top1_label": top1["label"],
            "top1_score": round(
                top1["score"], 6
            ),
            "top1_template": (
                top1["best_template"].name
            ),

            "top2_label": top2["label"],
            "top2_score": round(
                top2["score"], 6
            ),
            "top2_template": (
                top2["best_template"].name
            ),

            "top3_label": top3["label"],
            "top3_score": round(
                top3["score"], 6
            ),
            "top3_template": (
                top3["best_template"].name
            ),

            "reject_threshold": (
                REJECT_THRESHOLD
                if REJECT_THRESHOLD is not None
                else "disabled"
            ),
        })

    preliminary_accuracy = (
        number_correct / len(test_rows)
        if test_rows
        else 0.0
    )

    return result_rows, preliminary_accuracy


def save_results(result_rows):
    if not result_rows:
        raise ValueError(
            "Không có kết quả để lưu."
        )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULT_DIR
        / "F_recognition_results.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=result_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(result_rows)

    print("\nĐã lưu:", output_path)


def main():
    train_rows = read_split(TRAIN_CSV)
    test_rows = read_split(TEST_CSV)

    print("Số file train:", len(train_rows))
    print("Số file test :", len(test_rows))

    if len(test_rows) < 5:
        raise ValueError(
            "Phần F yêu cầu kết quả top-3 "
            "cho ít nhất năm file test."
        )

    check_data_leakage(
        train_rows,
        test_rows
    )

    print("\nĐang tạo ngân hàng template...")
    templates = build_templates(train_rows)

    print("\nĐang nhận dạng tập test...")
    result_rows, preliminary_accuracy = (
        evaluate_test_files(
            test_rows,
            templates
        )
    )

    save_results(result_rows)

    print("\n" + "=" * 65)
    print(
        "Độ chính xác kiểm tra sơ bộ: "
        f"{preliminary_accuracy * 100:.2f}%"
    )
    print(
        "Phần G sẽ tính accuracy và "
        "confusion matrix chính thức."
    )
    print("Hoàn thành phần F.")


if __name__ == "__main__":
    main()