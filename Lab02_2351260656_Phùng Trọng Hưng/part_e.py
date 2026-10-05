from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

TRAIN_CSV = ROOT / "splits" / "train.csv"
FEATURE_DIR = ROOT / "features" / "mfcc_trimmed"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"


def local_distance_matrix(X, Y):
    """
    X: shape (N, D)
    Y: shape (M, D)

    Trả về ma trận khoảng cách Euclid shape (N, M).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            "X và Y phải là ma trận hai chiều."
        )

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Số chiều MFCC không giống nhau: "
            f"{X.shape} và {Y.shape}"
        )

    differences = (
        X[:, None, :]
        - Y[None, :, :]
    )

    distances = np.sqrt(
        np.sum(differences ** 2, axis=2)
    )

    return distances


def dtw_distance(X, Y):
    """
    Tự cài đặt DTW bằng dynamic programming.

    Trả về:
      normalized_cost: chi phí chuẩn hóa
      total_cost: tổng chi phí chưa chuẩn hóa
      path: danh sách các cặp frame (i, j)
      local_cost: ma trận khoảng cách Euclid
      accumulated_cost: ma trận chi phí tích lũy
    """
    local_cost = local_distance_matrix(X, Y)

    N, M = local_cost.shape

    if N == 0 or M == 0:
        raise ValueError(
            "Chuỗi MFCC không được rỗng."
        )

    # Thêm một hàng/cột biên vô cực.
    accumulated = np.full(
        (N + 1, M + 1),
        np.inf,
        dtype=np.float64
    )

    accumulated[0, 0] = 0.0

    # Lưu tọa độ ô trước đó để backtracking.
    back_i = np.full(
        (N + 1, M + 1),
        -1,
        dtype=np.int32
    )

    back_j = np.full(
        (N + 1, M + 1),
        -1,
        dtype=np.int32
    )

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            candidates = [
                # Đi chéo: ghép frame mới của cả X và Y
                (
                    accumulated[i - 1, j - 1],
                    i - 1,
                    j - 1
                ),

                # Đi dọc
                (
                    accumulated[i - 1, j],
                    i - 1,
                    j
                ),

                # Đi ngang
                (
                    accumulated[i, j - 1],
                    i,
                    j - 1
                ),
            ]

            best_cost, previous_i, previous_j = min(
                candidates,
                key=lambda item: item[0]
            )

            accumulated[i, j] = (
                local_cost[i - 1, j - 1]
                + best_cost
            )

            back_i[i, j] = previous_i
            back_j[i, j] = previous_j

    # Backtracking từ (N, M) về (0, 0).
    path = []
    i, j = N, M

    while i > 0 or j > 0:
        if i <= 0 or j <= 0:
            raise RuntimeError(
                "Backtracking đi ra ngoài biên."
            )

        path.append((i - 1, j - 1))

        previous_i = int(back_i[i, j])
        previous_j = int(back_j[i, j])

        if previous_i < 0 or previous_j < 0:
            raise RuntimeError(
                "Không tìm thấy predecessor hợp lệ."
            )

        i, j = previous_i, previous_j

    path.reverse()

    total_cost = float(accumulated[N, M])
    normalized_cost = (
        total_cost / max(len(path), 1)
    )

    accumulated_without_border = (
        accumulated[1:, 1:]
    )

    validate_path(path, N, M)

    return (
        normalized_cost,
        total_cost,
        path,
        local_cost,
        accumulated_without_border
    )


def validate_path(path, N, M):
    """Kiểm tra endpoint và tính đơn điệu của DTW path."""
    if not path:
        raise ValueError("DTW path rỗng.")

    if path[0] != (0, 0):
        raise ValueError(
            f"Path không bắt đầu tại (0,0): {path[0]}"
        )

    if path[-1] != (N - 1, M - 1):
        raise ValueError(
            "Path không kết thúc tại "
            f"({N - 1},{M - 1}): {path[-1]}"
        )

    for previous, current in zip(
        path[:-1],
        path[1:]
    ):
        di = current[0] - previous[0]
        dj = current[1] - previous[1]

        if (di, dj) not in {
            (1, 0),
            (0, 1),
            (1, 1)
        }:
            raise ValueError(
                f"Bước DTW không hợp lệ: "
                f"{previous} -> {current}"
            )


def feature_path_from_csv_row(row):
    """
    train.csv đang trỏ đến dataset/<label>/<file>.wav.
    Đổi thành features/mfcc_trimmed/<label>/<file>.npy.
    """
    label = row["label"]
    wav_name = Path(row["path"]).name

    return (
        FEATURE_DIR
        / label
        / f"{Path(wav_name).stem}.npy"
    )


def load_training_features():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            "Không tìm thấy splits/train.csv."
        )

    features_by_label = {}

    with TRAIN_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row["label"]
            feature_path = feature_path_from_csv_row(row)

            if not feature_path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy {feature_path}. "
                    "Hãy chạy lại phần D."
                )

            features_by_label.setdefault(
                label,
                []
            ).append(feature_path)

    return features_by_label


def select_comparison_pairs(features_by_label):
    """
    Chọn:
      - X và Y_same: hai lần nói cùng một từ.
      - X và Y_different: hai từ khác nhau.

    Cùng dùng X để việc so sánh công bằng hơn.
    """
    same_label = None

    for label, paths in features_by_label.items():
        if len(paths) >= 2:
            same_label = label
            break

    if same_label is None:
        raise ValueError(
            "Cần ít nhất hai file train của cùng một từ."
        )

    different_labels = [
        label
        for label in features_by_label
        if label != same_label
        and len(features_by_label[label]) >= 1
    ]

    if not different_labels:
        raise ValueError(
            "Cần ít nhất hai nhãn khác nhau."
        )

    different_label = different_labels[0]

    x_path = features_by_label[same_label][0]
    same_path = features_by_label[same_label][1]
    different_path = (
        features_by_label[different_label][0]
    )

    return {
        "same_label": same_label,
        "different_label": different_label,
        "x_path": x_path,
        "same_path": same_path,
        "different_path": different_path,
    }


def load_mfcc(path):
    mfcc = np.load(path)

    if mfcc.ndim != 2:
        raise ValueError(
            f"{path.name}: MFCC phải có hai chiều."
        )

    if mfcc.shape[1] != 13:
        raise ValueError(
            f"{path.name}: shape={mfcc.shape}; "
            "cần dạng (T,13)."
        )

    if not np.all(np.isfinite(mfcc)):
        raise ValueError(
            f"{path.name}: MFCC chứa NaN hoặc infinity."
        )

    return mfcc


def run_comparison(pair_info):
    X = load_mfcc(pair_info["x_path"])
    Y_same = load_mfcc(pair_info["same_path"])
    Y_different = load_mfcc(
        pair_info["different_path"]
    )

    # Kiểm tra bắt buộc: X so với chính nó phải gần 0.
    self_result = dtw_distance(X, X)
    self_cost = self_result[0]

    if not np.isclose(
        self_cost,
        0.0,
        atol=1e-7
    ):
        raise AssertionError(
            f"DTW(X,X) phải gần 0, nhận được {self_cost}"
        )

    same_result = dtw_distance(
        X,
        Y_same
    )

    different_result = dtw_distance(
        X,
        Y_different
    )

    return (
        X,
        Y_same,
        Y_different,
        same_result,
        different_result,
        self_cost
    )


def plot_dtw_comparison(
    pair_info,
    same_result,
    different_result
):
    same_norm, _, same_path, same_local, _ = (
        same_result
    )

    (
        different_norm,
        _,
        different_path,
        different_local,
        _
    ) = different_result

    # Dùng cùng thang màu để so sánh hai ma trận.
    combined_values = np.concatenate([
        same_local.ravel(),
        different_local.ravel()
    ])

    vmax = max(
        float(np.percentile(
            combined_values,
            95
        )),
        1e-6
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        layout="constrained"
    )

    comparisons = [
        (
            axes[0],
            same_local,
            same_path,
            (
                f"Cùng từ: {pair_info['same_label']}\n"
                f"DTW_norm = {same_norm:.4f}"
            )
        ),
        (
            axes[1],
            different_local,
            different_path,
            (
                f"Khác từ: "
                f"{pair_info['same_label']} - "
                f"{pair_info['different_label']}\n"
                f"DTW_norm = {different_norm:.4f}"
            )
        ),
    ]

    last_image = None

    for axis, local, path, title in comparisons:
        last_image = axis.imshow(
            local,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=0,
            vmax=vmax
        )

        path_array = np.asarray(path)

        # path chứa (i,j):
        # trục x là j của chuỗi Y,
        # trục y là i của chuỗi X.
        axis.plot(
            path_array[:, 1],
            path_array[:, 0],
            color="cyan",
            linewidth=2,
            label="Optimal path"
        )

        # Đường chéo tham chiếu.
        axis.plot(
            [0, local.shape[1] - 1],
            [0, local.shape[0] - 1],
            color="white",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Đường chéo tham chiếu"
        )

        axis.set_title(title)
        axis.set_xlabel("Frame của chuỗi Y")
        axis.set_ylabel("Frame của chuỗi X")
        axis.legend(loc="upper left")

    fig.colorbar(
        last_image,
        ax=axes,
        label="Khoảng cách Euclid cục bộ"
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FIG_DIR / "E_dtw_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Đã lưu hình:", output_path)


def save_results(
    pair_info,
    X,
    Y_same,
    Y_different,
    same_result,
    different_result,
    self_cost
):
    same_norm, same_total, same_path, _, _ = (
        same_result
    )

    (
        different_norm,
        different_total,
        different_path,
        _,
        _
    ) = different_result

    rows = [
        {
            "comparison": "self_check",
            "label_x": pair_info["same_label"],
            "label_y": pair_info["same_label"],
            "file_x": pair_info["x_path"].name,
            "file_y": pair_info["x_path"].name,
            "frames_x": len(X),
            "frames_y": len(X),
            "total_cost": 0.0,
            "path_length": len(X),
            "dtw_norm": round(self_cost, 6),
        },
        {
            "comparison": "same_word",
            "label_x": pair_info["same_label"],
            "label_y": pair_info["same_label"],
            "file_x": pair_info["x_path"].name,
            "file_y": pair_info["same_path"].name,
            "frames_x": len(X),
            "frames_y": len(Y_same),
            "total_cost": round(
                same_total,
                6
            ),
            "path_length": len(same_path),
            "dtw_norm": round(
                same_norm,
                6
            ),
        },
        {
            "comparison": "different_words",
            "label_x": pair_info["same_label"],
            "label_y": pair_info["different_label"],
            "file_x": pair_info["x_path"].name,
            "file_y": pair_info[
                "different_path"
            ].name,
            "frames_x": len(X),
            "frames_y": len(Y_different),
            "total_cost": round(
                different_total,
                6
            ),
            "path_length": len(
                different_path
            ),
            "dtw_norm": round(
                different_norm,
                6
            ),
        },
    ]

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULT_DIR / "E_dtw_comparison.csv"
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


def main():
    features_by_label = load_training_features()
    pair_info = select_comparison_pairs(
        features_by_label
    )

    (
        X,
        Y_same,
        Y_different,
        same_result,
        different_result,
        self_cost
    ) = run_comparison(pair_info)

    same_norm = same_result[0]
    different_norm = different_result[0]

    print("\nKiểm tra DTW")
    print(f"DTW_norm(X, X) = {self_cost:.8f}")
    print(
        f"Cùng từ '{pair_info['same_label']}': "
        f"{same_norm:.6f}"
    )
    print(
        f"Khác từ "
        f"'{pair_info['same_label']}' - "
        f"'{pair_info['different_label']}': "
        f"{different_norm:.6f}"
    )

    if same_norm < different_norm:
        print(
            "Kết quả phù hợp kỳ vọng: "
            "cùng từ có chi phí nhỏ hơn."
        )
    else:
        print(
            "Cảnh báo: cặp khác từ không có chi phí "
            "lớn hơn cặp cùng từ. Hãy kiểm tra lại "
            "file ghi âm, endpoint hoặc MFCC."
        )

    plot_dtw_comparison(
        pair_info,
        same_result,
        different_result
    )

    save_results(
        pair_info,
        X,
        Y_same,
        Y_different,
        same_result,
        different_result,
        self_cost
    )

    print("\nHoàn thành phần E.")


if __name__ == "__main__":
    main()