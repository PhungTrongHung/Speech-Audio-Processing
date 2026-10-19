import sounddevice as sd
import soundfile as sf
from pathlib import Path
import time

SR = 16000
DURATION = 2.0

labels = {
    "mo": "mở",
    "dong": "đóng",
    "len": "lên",
    "xuong": "xuống",
    "dung": "dừng"
}

for label, word in labels.items():

    print("\n==========================")
    print("Từ cần đọc:", word)
    print("==========================")

    for i in range(1, 9):

        if i <= 6:
            split = "train"
        else:
            split = "test"

        folder = Path("dataset") / split / label
        folder.mkdir(parents=True, exist_ok=True)

        filename = folder / f"{label}_{i:02d}.wav"

        input(
            f"\nChuẩn bị đọc '{word}' lần {i}. "
            "Nhấn ENTER để bắt đầu..."
        )

        print("Đang thu...")

        audio = sd.rec(
            int(DURATION * SR),
            samplerate=SR,
            channels=1,
            dtype="int16"
        )

        sd.wait()

        sf.write(
            filename,
            audio,
            SR,
            subtype="PCM_16"
        )

        print("Đã lưu:", filename)

        time.sleep(0.5)

print("\nHOÀN THÀNH DATASET!")