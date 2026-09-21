# CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI

# LAB 1: PHÂN TÍCH VÀ XỬ LÝ TÍN HIỆU ÂM THANH SỐ

**Sinh viên:** 2351260656\
**File âm thanh:** `Lab01_light_noise.mp3`\
**Notebook:** `Lab01_2351260656.ipynb`

------------------------------------------------------------------------

## 1. Mục tiêu bài thực hành

Bài thực hành Lab 1 tập trung vào chuỗi xử lý:

**Audio → biểu diễn số → phân tích miền thời gian → FFT/STFT → lọc số →
lượng tử hóa/mã hóa → đánh giá kết quả**

Các nội dung thực hiện:

-   Đọc và phân tích metadata của file âm thanh.
-   Phân tích waveform, Peak, RMS, Energy.
-   Phân tích phổ tần bằng FFT.
-   Xây dựng spectrogram bằng STFT.
-   So sánh cửa sổ Rectangular và Hamming.
-   Thiết kế FIR low-pass filter bằng tích chập.
-   Thực nghiệm lượng tử hóa, SNR, resampling, bitrate và compression
    ratio.

------------------------------------------------------------------------

# A. Đọc dữ liệu âm thanh và metadata

File âm thanh đầu vào được đọc bằng thư viện `librosa`.

Các bước xử lý:

-   Đọc file MP3.
-   Chuyển tín hiệu về mono.
-   Chuẩn hóa biên độ tín hiệu về miền \[-1,1\].

Thông tin thu được:

  Thông số        Giá trị
  --------------- ---------------------------------
  File            Lab01_light_noise.mp3
  Sampling rate   Fs (được đọc trực tiếp từ file)
  Channels        Mono
  Dạng dữ liệu    Float
  Chuẩn hóa       \[-1,1\]

------------------------------------------------------------------------

# B. Phân tích miền thời gian

Các đại lượng được tính:

## Peak

\[ Peak = max\|x\[n\]\| \]

Dùng để kiểm tra mức biên độ lớn nhất và khả năng clipping.

## RMS

\[ RMS=`\sqrt{\frac{1}{N}\sum x^2[n]}`{=tex} \]

RMS biểu diễn mức năng lượng hiệu dụng của tín hiệu.

## Energy

\[ E=`\sum `{=tex}x\^2\[n\] \]

Kết quả được biểu diễn qua:

-   Waveform toàn bộ tín hiệu.
-   Waveform đoạn 1 giây đầu.

Hình ảnh:

    figures/waveform.png

Nhận xét:

-   Waveform cho thấy sự thay đổi biên độ theo thời gian.
-   RMS và Energy phản ánh mức năng lượng trung bình của tín hiệu.
-   Peak được sử dụng để kiểm tra nguy cơ clipping.

------------------------------------------------------------------------

# C. Phân tích miền tần số bằng FFT

FFT được thực hiện trên đoạn tín hiệu 1 giây đầu với cửa sổ Hamming.

Hai giá trị NFFT được khảo sát:

  NFFT   Khoảng cách bin tần số
  ------ ------------------------
  2048   Fs/2048
  8192   Fs/8192

Công thức:

\[ `\Delta `{=tex}f=`\frac{F_s}{N_{FFT}}`{=tex} \]

Kết quả:

-   Khi tăng NFFT, khoảng cách giữa các bin tần số giảm.
-   NFFT lớn giúp phổ biểu diễn mịn hơn.
-   Tuy nhiên không tạo thêm thông tin vật lý mới.

Các đỉnh phổ được tìm bằng cách chọn các giá trị magnitude lớn nhất.

Hình ảnh:

    figures/fft.png

------------------------------------------------------------------------

# D. STFT và Spectrogram

Spectrogram được xây dựng bằng STFT với:

  Tham số        Giá trị
  -------------- ---------
  Frame length   25 ms
  Hop size       10 ms
  Window         Hamming

Biểu diễn:

\[ S\_{dB}=20log\_{10}(\|X\|+`\epsilon`{=tex}) \]

Hình ảnh:

    figures/spectrogram.png

Nhận xét:

-   Spectrogram thể hiện sự thay đổi năng lượng theo thời gian.
-   Vùng sáng thể hiện thành phần năng lượng lớn.
-   Frame ngắn cho độ phân giải thời gian tốt hơn.
-   Frame dài cho độ phân giải tần số tốt hơn.

------------------------------------------------------------------------

# E. Thí nghiệm cửa sổ

So sánh:

-   Rectangular window.
-   Hamming window.

Kết quả:

-   Rectangular có spectral leakage lớn hơn.
-   Hamming làm giảm side-lobe và leakage.
-   Tuy nhiên Hamming làm main-lobe rộng hơn, có thể giảm khả năng phân
    biệt các đỉnh gần nhau.

Hình ảnh:

    figures/window_compare.png

------------------------------------------------------------------------

# F. Lọc số FIR

Bộ lọc FIR low-pass được xây dựng bằng moving average:

-   Số tap: 101
-   Hệ số:

\[ h\[n\]=`\frac{1}{101}`{=tex} \]

Tín hiệu sau lọc được tạo bằng phép tích chập:

\[ y\[n\]=x\[n\]\*h\[n\] \]

File âm thanh sau lọc:

    audio/filtered_lowpass.wav

Đáp ứng tần số:

    figures/filter_response.png

Nhận xét:

-   Bộ lọc làm suy giảm các thành phần tần số cao.
-   Âm thanh sau lọc có xu hướng giảm thành phần high-frequency.
-   Đáp ứng tần số phù hợp với đặc tính low-pass.

------------------------------------------------------------------------

# G. Lượng tử hóa, Resampling và Coding

## 1. Quantization

Các mức lượng tử khảo sát:

-   4 bit
-   8 bit
-   16 bit

Công thức SNR:

\[ SNR=10log\_{10} `\frac{\sum x^2[n]}`{=tex}
{`\sum`{=tex}(x\[n\]-x_q\[n\])\^2} \]

Kết quả SNR được tính trực tiếp từ tín hiệu.

File xuất:

    audio/quantized_4bit.wav
    audio/quantized_8bit.wav
    audio/quantized_16bit.wav

Nhận xét:

-   Tăng số bit lượng tử làm giảm sai số lượng tử.
-   SNR tăng khi số bit tăng.
-   SNR lớn không hoàn toàn quyết định chất lượng nghe.

------------------------------------------------------------------------

## 2. Resampling

Tín hiệu được giảm sampling rate:

-   16 kHz
-   8 kHz

File:

    audio/resampled_16000Hz.wav
    audio/resampled_8000Hz.wav

Nhận xét:

-   Sampling rate thấp làm giảm dải tần có thể biểu diễn.
-   8 kHz phù hợp hơn cho tiếng nói hơn là nhạc.

------------------------------------------------------------------------

## 3. Bitrate và Compression Ratio

PCM bitrate:

\[ R\_{PCM}=F_s `\times `{=tex}Bits `\times `{=tex}Channels \]

Trong notebook:

-   Bit depth = 16 bit
-   Channels = 1

So sánh với MP3:

\[ Compression Ratio= `\frac{R_{PCM}}{R_{MP3}}`{=tex} \]

Kết quả được tính với MP3 128 kbps.

------------------------------------------------------------------------

# Câu hỏi phân tích

## 1. Vì sao Fs = 44.1 kHz chỉ biểu diễn đến 22.05 kHz?

Theo định lý Nyquist:

\[ F_s `\geq `{=tex}2F\_{max} \]

Do đó:

\[ F\_{max}=`\frac{F_s}{2}`{=tex} \]

Với:

\[ F_s=44100Hz \]

ta có:

\[ F\_{max}=22050Hz \]

------------------------------------------------------------------------

## 2. Tăng NFFT từ 2048 lên 8192

Thay đổi:

-   Khoảng cách bin tần số giảm.
-   Phổ nhìn mịn hơn.

Không thay đổi:

-   Độ phân giải vật lý phụ thuộc chủ yếu vào độ dài frame.

------------------------------------------------------------------------

## 3. Vì sao Hamming giảm leakage?

Hamming giảm biên độ side-lobe nên giảm spectral leakage.

Tuy nhiên main-lobe rộng hơn làm các đỉnh gần nhau khó tách hơn.

------------------------------------------------------------------------

## 4. Ảnh hưởng của FIR

FIR có đáp ứng tần số xác định bởi:

\[ H(e\^{j`\omega`{=tex}}) \]

Bộ lọc low-pass giữ tần số thấp và suy giảm tần số cao.

------------------------------------------------------------------------

## 5. Quan hệ giữa bit lượng tử và SNR

Khi tăng số bit:

-   Số mức lượng tử tăng.
-   Sai số lượng tử giảm.
-   SNR tăng.

------------------------------------------------------------------------

## 6. Nghe tốt hơn không luôn đồng nghĩa SNR lớn hơn

Ví dụ:

-   Một bộ lọc loại bỏ nhiễu ngoài dải nghe có thể cải thiện cảm nhận dù
    SNR thay đổi ít.
-   Một tín hiệu có SNR cao nhưng mất thành phần tần số quan trọng có
    thể nghe kém tự nhiên.

------------------------------------------------------------------------

# Kết luận

Bài thực hành đã thực hiện đầy đủ pipeline xử lý âm thanh số:

-   Đọc và chuẩn hóa tín hiệu.
-   Phân tích miền thời gian.
-   Phân tích FFT/STFT.
-   Thử nghiệm cửa sổ.
-   Thiết kế FIR filter.
-   Lượng tử hóa, resampling và đánh giá bitrate.

Kết quả cho thấy các tham số xử lý như sampling rate, frame length,
window, số bit lượng tử và bộ lọc có ảnh hưởng trực tiếp đến biểu diễn
và chất lượng tín hiệu âm thanh.
