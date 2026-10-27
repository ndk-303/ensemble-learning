# SỔ TAY HƯỚNG DẪN & TÍNH TOÁN CHẠY TAY CHI TIẾT: BAGGING, BOOSTING, STACKING
*(Tài liệu chuẩn mực phương pháp giải toán Học máy - Machine Learning dành cho sinh viên)*

---

## MỤC LỤC
1. [TỔNG QUAN VỀ HỌC KẾT HỢP (ENSEMBLE LEARNING)](#1-tong-quan-ve-hoc-ket-hop-ensemble-learning)
2. [THUẬT TOÁN 1: BAGGING (BOOTSTRAP AGGREGATING)](#2-thuat-toan-1-bagging-bootstrap-aggregating)
   - 2.1 Cơ sở toán học & Các bước thuật toán
   - 2.2 Ví dụ chạy tay chi tiết từng bước
   - 2.3 Đánh giá sai số Out-of-Bag (OOB Error)
3. [THUẬT TOÁN 2: BOOSTING (ADABOOST - ADAPTIVE BOOSTING)](#3-thuat-toan-2-boosting-adaboost---adaptive-boosting)
   - 3.1 Cơ sở toán học & Các bước thuật toán
   - 3.2 Ví dụ chạy tay chi tiết (Bấm máy tính bỏ túi từng bước)
   - 3.3 Tổng hợp mô hình & Minh chứng 100% độ chính xác
   - 3.4 Mở rộng: Giải thuật SAMME cho bài toán Đa lớp
4. [THUẬT TOÁN 3: STACKING (STACKED GENERALIZATION)](#4-thuat-toan-3-stacking-stacked-generalization)
   - 4.1 Cơ sở toán học & Cơ chế Out-Of-Fold (OOF) chống rò rỉ dữ liệu
   - 4.2 Ví dụ chạy tay chi tiết 2 tầng (Level-0 & Level-1)
   - 4.3 Quá trình suy luận trên mẫu kiểm thử mới (Inference)
5. [BẢNG SO SÁNH TỔNG HỢP 3 THUẬT TOÁN](#5-bang-so-sanh-tong-hop-3-thuat-toan)
6. [BÍ KÍP & BẪY KINH ĐIỂN CẦN TRÁNH KHI LÀM BÀI THI](#6-bi-kip--bay-kinh-dien-can-tranh-khi-lam-bai-thi)

---

## 1. TỔNG QUAN VỀ HỌC KẾT HỢP (ENSEMBLE LEARNING)

Trong học máy, sai số dự đoán tổng quát của một mô hình được phân rã theo công thức **Bias-Variance Decomposition**:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

* **Bias (Độ lệch):** Sai số do giả định mô hình quá đơn giản, không học hết quy luật của dữ liệu $\rightarrow$ Gây ra **Underfitting**.
* **Variance (Phương sai):** Sự nhạy cảm của mô hình đối với sự biến thiên ngẫu nhiên của tập dữ liệu huấn luyện $\rightarrow$ Gây ra **Overfitting**.
* **Noise (Nhiễu):** Sai số không thể tránh khỏi của dữ liệu thực tế.

Ba giải thuật Ensemble giải quyết bài toán theo ba hướng tiếp cận cốt lõi:
1. **Bagging:** Huấn luyện **song song** các mô hình phức tạp (High Variance) trên các tập con mẫu ngẫu nhiên $\rightarrow$ **Giảm Variance**.
2. **Boosting:** Huấn luyện **tuần tự** các mô hình rất đơn giản (High Bias) sửa sai cho nhau $\rightarrow$ **Giảm Bias**.
3. **Stacking:** Huấn luyện **phân tầng** đa dạng cấu trúc mô hình, dùng một mô hình Meta để học cách dung hòa điểm mạnh/yếu $\rightarrow$ **Tối ưu hóa khả năng khái quát hóa**.

---

## 2. THUẬT TOÁN 1: BAGGING (BOOTSTRAP AGGREGATING)

### 2.1 Cơ sở toán học & Các bước thuật toán

Bagging kết hợp hai ý tưởng: **Bootstrap Sampling** (lấy mẫu có hoàn lại) và **Aggregating** (tổng hợp dự đoán).

#### Thuật toán từng bước:
* **Đầu vào:** Tập dữ liệu huấn luyện $D = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$, số lượng mô hình cơ sở $T$, thuật toán học cơ sở $A$ (thường là Cây quyết định sâu - Deep Decision Tree).
* **Vòng lặp:** Với mỗi $t = 1, 2, \dots, T$:
  1. **Bước 1 (Bootstrap Sampling):** Rút ngẫu nhiên đều $n$ mẫu **CÓ HOÀN LẠI** từ tập $D$. Ta thu được tập huấn luyện con $D_t$ ($|D_t| = n$). Một số mẫu xuất hiện nhiều lần, một số mẫu không xuất hiện.
  2. **Bước 2 (Fit Model):** Huấn luyện mô hình cơ sở $h_t = A(D_t)$ trên tập $D_t$.
* **Đầu ra (Aggregating):**
  * **Bài toán Phân loại (Classification):** Biểu quyết đa số (Majority Voting):
    $$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} \sum_{t=1}^T \mathbb{I}(h_t(x) = k)$$
  * **Bài toán Hồi quy (Regression):** Tính trung bình cộng:
    $$\hat{y} = \frac{1}{T} \sum_{t=1}^T h_t(x)$$

#### Định lý Out-Of-Bag (OOB):
Xác suất một mẫu dữ liệu cụ thể **không** được chọn sau $n$ lần rút là:
$$P(\text{không được chọn}) = \left(1 - \frac{1}{n}\right)^n$$
Khi kích thước tập dữ liệu $n$ đủ lớn ($n \to \infty$):
$$\lim_{n \to \infty} \left(1 - \frac{1}{n}\right)^n = \frac{1}{e} \approx 0.368 \ (36.8\%)$$
$\rightarrow$ Khoảng **36.8%** mẫu dữ liệu ban đầu không nằm trong tập $D_t$ của mô hình $h_t$. Các mẫu này được gọi là **Out-Of-Bag (OOB)** và được tận dụng làm tập kiểm thử tự nhiên để đánh giá mô hình mà không cần chia tập Validation riêng!

---

### 2.2 Ví dụ chạy tay chi tiết từng bước

Giả sử ta có tập dữ liệu huấn luyện 1 chiều $D$ gồm $n = 6$ mẫu:

| Chỉ số ($i$) | Giá trị đặc trưng ($X$) | Nhãn thực tế ($y$) |
|:---:|:---:|:---:|
| 1 | 1.0 | 0 |
| 2 | 2.0 | 0 |
| 3 | 3.0 | 1 |
| 4 | 5.0 | 0 |
| 5 | 6.0 | 1 |
| 6 | 7.0 | 1 |

Ta cần xây dựng mô hình Bagging gồm **$T = 3$ cây quyết định**. Mỗi cây là một Decision Stump tìm ngưỡng cắt $\theta$ sao cho số mẫu phân loại sai trên tập con của nó là nhỏ nhất.

---

#### CÂY 1 ($t = 1$):
* **Bước 1.1:** Lấy ngẫu nhiên $n=6$ mẫu có hoàn lại từ $D$. Giả sử ta bốc được các chỉ số sau:
  $$\text{Chỉ số được chọn}: \{1, 1, 2, 3, 5, 6\}$$
* Tập dữ liệu con $D_1$ gồm:
  $$(1.0, 0), (1.0, 0), (2.0, 0), (3.0, 1), (6.0, 1), (7.0, 1)$$
  *(Lưu ý: Mẫu 4 bị sót lại, thuộc tập OOB của Cây 1).*
* **Bước 1.2:** Huấn luyện Cây 1:
  - Dữ liệu $X$ sắp xếp: $[1.0, 1.0, 2.0]$ (nhãn 0) và $[3.0, 6.0, 7.0]$ (nhãn 1).
  - Điểm cắt hoàn hảo chia tách lớp 0 và lớp 1 là giữa $X = 2.0$ và $X = 3.0$, chọn $\theta_1 = 2.5$.
  - **Quy tắc Cây 1 ($h_1$):**
    $$h_1(x) = \begin{cases} 0 & \text{nếu } x \le 2.5 \\ 1 & \text{nếu } x > 2.5 \end{cases}$$
  - Số mẫu sai trên $D_1$: $0/6$ ($0\%$).

---

#### CÂY 2 ($t = 2$):
* **Bước 2.1:** Rút mẫu ngẫu nhiên lần 2, bốc được các chỉ số:
  $$\text{Chỉ số được chọn}: \{2, 3, 4, 4, 5, 6\}$$
* Tập dữ liệu con $D_2$ gồm:
  $$(2.0, 0), (3.0, 1), (5.0, 0), (5.0, 0), (6.0, 1), (7.0, 1)$$
  *(Lưu ý: Mẫu 1 bị sót lại, thuộc tập OOB của Cây 2).*
* **Bước 2.2:** Huấn luyện Cây 2:
  - Các mẫu có nhãn 0: $X = 2.0, 5.0, 5.0$.
  - Các mẫu có nhãn 1: $X = 3.0, 6.0, 7.0$.
  - Dữ liệu bị xen kẽ ($2.0 \to 0, 3.0 \to 1, 5.0 \to 0, 6.0 \to 1, 7.0 \to 1$).
  - Ta thử các ngưỡng cắt:
    - Nếu cắt tại $\theta = 5.5$ ($X \le 5.5 \to 0$, $X > 5.5 \to 1$):
      - Đoán 0 cho $\{2.0, 3.0, 5.0, 5.0\}$ $\rightarrow$ sai ở mẫu $3.0$ ($y=1$).
      - Đoán 1 cho $\{6.0, 7.0\}$ $\rightarrow$ đúng.
      - Tổng sai: $1/6$ mẫu.
  - Ngưỡng tối ưu là $\theta_2 = 5.5$.
  - **Quy tắc Cây 2 ($h_2$):**
    $$h_2(x) = \begin{cases} 0 & \text{nếu } x \le 5.5 \\ 1 & \text{nếu } x > 5.5 \end{cases}$$

---

#### CÂY 3 ($t = 3$):
* **Bước 3.1:** Rút mẫu ngẫu nhiên lần 3, bốc được các chỉ số:
  $$\text{Chỉ số được chọn}: \{1, 2, 3, 4, 4, 6\}$$
* Tập dữ liệu con $D_3$ gồm:
  $$(1.0, 0), (2.0, 0), (3.0, 1), (5.0, 0), (5.0, 0), (7.0, 1)$$
  *(Lưu ý: Mẫu 5 bị sót lại, thuộc tập OOB của Cây 3).*
* **Bước 3.2:** Huấn luyện Cây 3:
  - Tương tự $D_2$, ngưỡng cắt tối ưu cho sai số nhỏ nhất trên tập này là $\theta_3 = 5.5$ (chỉ sai đúng mẫu $X=3.0$).
  - **Quy tắc Cây 3 ($h_3$):**
    $$h_3(x) = \begin{cases} 0 & \text{nếu } x \le 5.5 \\ 1 & \text{nếu } x > 5.5 \end{cases}$$

---

#### DỰ ĐOÁN & BIỂU QUYẾT ĐA SỐ (AGGREGATING):

Giả sử ta cần dự đoán nhãn cho hai điểm dữ liệu kiểm thử mới:
- Điểm $A$: $X_A = 3.2$
- Điểm $B$: $X_B = 6.2$

| Điểm test | Dự đoán Cây 1 ($h_1$) | Dự đoán Cây 2 ($h_2$) | Dự đoán Cây 3 ($h_3$) | Số phiếu Lớp 0 | Số phiếu Lớp 1 | Quyết định cuối cùng ($\hat{y}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$X_A = 3.2$** | $3.2 > 2.5 \rightarrow \mathbf{1}$ | $3.2 \le 5.5 \rightarrow \mathbf{0}$ | $3.2 \le 5.5 \rightarrow \mathbf{0}$ | **2 phiếu** ($h_2, h_3$) | 1 phiếu ($h_1$) | **Lớp 0** |
| **$X_B = 6.2$** | $6.2 > 2.5 \rightarrow \mathbf{1}$ | $6.2 > 5.5 \rightarrow \mathbf{1}$ | $6.2 > 5.5 \rightarrow \mathbf{1}$ | 0 phiếu | **3 phiếu** ($h_1, h_2, h_3$) | **Lớp 1** |

---

### 2.3 Đánh giá sai số Out-of-Bag (OOB Error)

Để tính OOB Error, ta chỉ cho một cây dự đoán trên những mẫu **không** tham gia huấn luyện cây đó:
* **Mẫu 1 ($X=1.0, y=0$):** OOB đối với **Cây 2**.
  - $h_2(1.0) = 0 \rightarrow$ **Đúng** ($0 == 0$).
* **Mẫu 4 ($X=5.0, y=0$):** OOB đối với **Cây 1**.
  - $h_1(5.0) = 1 \rightarrow$ **Sai** ($1 \neq 0$).
* **Mẫu 5 ($X=6.0, y=1$):** OOB đối với **Cây 3**.
  - $h_3(6.0) = 1 \rightarrow$ **Đúng** ($1 == 1$).

$$\text{OOB Error} = \frac{\text{Số lần đoán sai}}{\text{Tổng số lượt đánh giá OOB}} = \frac{1}{3} \approx 33.33\%$$

---

## 3. THUẬT TOÁN 2: BOOSTING (ADABOOST - ADAPTIVE BOOSTING)

### 3.1 Cơ sở toán học & Các bước thuật toán

Khác với Bagging, **AdaBoost** huấn luyện các mô hình một cách **tuần tự (Sequential)**. Mỗi mô hình mới sinh ra tập trung toàn bộ "sự chú ý" vào những mẫu mà các mô hình trước đó đã làm sai bằng cách tăng trọng số của các mẫu đó lên.

#### Công thức toán học (Áp dụng cho nhãn $y \in \{-1, +1\}$):

1. **Khởi tạo:** Gán trọng số ban đầu đều nhau cho mọi mẫu:
   $$w_1(i) = \frac{1}{n}, \quad \forall i = 1, \dots, n$$

2. **Vòng lặp tuần tự:** Với mỗi vòng $t = 1, 2, \dots, T$:
   * **Bước 2.1:** Huấn luyện mô hình yếu $h_t(x)$ sao cho sai số có trọng số $\epsilon_t$ là nhỏ nhất:
     $$\epsilon_t = \sum_{i: y_i \neq h_t(x_i)} w_t(i)$$
   * **Bước 2.2:** Tính **hệ số tin cậy (trọng số của mô hình)** $\alpha_t$:
     $$\alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$$
     - Nếu $\epsilon_t \to 0$ (mô hình rất tốt) $\rightarrow \alpha_t \to +\infty$.
     - Nếu $\epsilon_t = 0.5$ (đoán ngẫu nhiên) $\rightarrow \alpha_t = \frac{1}{2}\ln(1) = 0$.
     - Nếu $\epsilon_t > 0.5$ (kém hơn đoán mò) $\rightarrow$ Đảo dấu mô hình hoặc dừng sớm.
   * **Bước 2.3:** Cập nhật trọng số của từng mẫu dữ liệu:
     $$w_{t+1}(i) = \frac{w_t(i) \cdot \exp\Big( -\alpha_t \cdot y_i \cdot h_t(x_i) \Big)}{Z_t}$$
     Trong đó:
     - Nếu mẫu đoán **đúng** ($y_i \cdot h_t(x_i) = +1$): Nhân thêm $e^{-\alpha_t} < 1 \rightarrow$ **Giảm trọng số**.
     - Nếu mẫu đoán **sai** ($y_i \cdot h_t(x_i) = -1$): Nhân thêm $e^{+\alpha_t} > 1 \rightarrow$ **Tăng trọng số**.
     - $Z_t$ là hằng số chuẩn hóa để $\sum_{i=1}^n w_{t+1}(i) = 1$:
       $$Z_t = \sum_{i=1}^n w_t(i) \exp\Big(-\alpha_t y_i h_t(x_i)\Big) = 2\sqrt{\epsilon_t(1 - \epsilon_t)}$$

3. **Mô hình kết hợp cuối cùng (Ensemble Decision):**
   $$H(x) = \text{sign}\left( \sum_{t=1}^T \alpha_t \cdot h_t(x) \right)$$

---

### 3.2 Ví dụ chạy tay chi tiết (Bấm máy tính bỏ túi từng bước)

Xét bài toán trong không gian 2 chiều $(X_1, X_2)$ gồm $n = 5$ mẫu, nhãn $y \in \{-1, +1\}$:

| Chỉ số ($i$) | Tọa độ $X_1$ | Tọa độ $X_2$ | Nhãn thực tế ($y$) |
|:---:|:---:|:---:|:---:|
| **1** | 1.0 | 4.0 | **+1** |
| **2** | 2.0 | 1.0 | **+1** |
| **3** | 3.0 | 3.0 | **-1** |
| **4** | 4.0 | 1.0 | **-1** |
| **5** | 5.0 | 4.0 | **+1** |

Weak learner được dùng là **Decision Stump** (đường phân chia thẳng đứng $X_1 \le \theta$ hoặc nằm ngang $X_2 \le \theta$).

---

#### KHỞI TẠO ($t = 1$):
Trọng số ban đầu phân bổ đều cho 5 mẫu:
$$w_1 = \left[ \frac{1}{5}, \frac{1}{5}, \frac{1}{5}, \frac{1}{5}, \frac{1}{5} \right] = [0.2, 0.2, 0.2, 0.2, 0.2]$$

---

#### VÒNG LẶP 1 ($t = 1$):

1. **Tìm mô hình $h_1$:**
   Ta xét đường cắt thẳng đứng theo trục $X_1$ tại $\theta = 2.5$:
   $$h_1(x) = \begin{cases} +1 & \text{nếu } X_1 \le 2.5 \\ -1 & \text{nếu } X_1 > 2.5 \end{cases}$$
   Kiểm tra dự đoán:
   - Mẫu 1 ($X_1=1 \le 2.5$): Đoán $+1$ $\rightarrow$ Đúng ($y_1 = +1$).
   - Mẫu 2 ($X_1=2 \le 2.5$): Đoán $+1$ $\rightarrow$ Đúng ($y_2 = +1$).
   - Mẫu 3 ($X_1=3 > 2.5$): Đoán $-1$ $\rightarrow$ Đúng ($y_3 = -1$).
   - Mẫu 4 ($X_1=4 > 2.5$): Đoán $-1$ $\rightarrow$ Đúng ($y_4 = -1$).
   - Mẫu 5 ($X_1=5 > 2.5$): Đoán $-1$ $\rightarrow$ **SAI** (Thực tế $y_5 = +1$).

2. **Tính sai số $\epsilon_1$:**
   Chỉ có mẫu 5 bị sai:
   $$\epsilon_1 = w_1(5) = 0.2$$

3. **Tính trọng số mô hình $\alpha_1$:**
   $$\alpha_1 = \frac{1}{2} \ln\left( \frac{1 - 0.2}{0.2} \right) = \frac{1}{2} \ln\left( \frac{0.8}{0.2} \right) = \frac{1}{2} \ln(4) = \ln(2) \approx \mathbf{0.6931}$$

4. **Cập nhật trọng số mẫu:**
   - Hệ số giảm trọng số (mẫu đúng): $e^{-\alpha_1} = e^{-\ln 2} = \frac{1}{2} = 0.5$
   - Hệ số tăng trọng số (mẫu sai): $e^{+\alpha_1} = e^{+\ln 2} = 2.0$

   Trọng số chưa chuẩn hóa $w'_2(i) = w_1(i) \cdot e^{-\alpha_1 y_i h_1(x_i)}$:
   - Mẫu 1: $0.2 \times 0.5 = 0.1$
   - Mẫu 2: $0.2 \times 0.5 = 0.1$
   - Mẫu 3: $0.2 \times 0.5 = 0.1$
   - Mẫu 4: $0.2 \times 0.5 = 0.1$
   - Mẫu 5: $0.2 \times 2.0 = 0.4$

   Tổng hằng số chuẩn hóa $Z_1$:
   $$Z_1 = 0.1 + 0.1 + 0.1 + 0.1 + 0.4 = 0.8$$
   *(Hoặc tính theo công thức: $Z_1 = 2\sqrt{0.2 \times 0.8} = 2\sqrt{0.16} = 2 \times 0.4 = 0.8$).*

   Chuẩn hóa $w_2(i) = \frac{w'_2(i)}{Z_1}$:
   - $w_2(1) = 0.1 / 0.8 = \mathbf{0.125} = 1/8$
   - $w_2(2) = 0.1 / 0.8 = \mathbf{0.125} = 1/8$
   - $w_2(3) = 0.1 / 0.8 = \mathbf{0.125} = 1/8$
   - $w_2(4) = 0.1 / 0.8 = \mathbf{0.125} = 1/8$
   - $w_2(5) = 0.4 / 0.8 = \mathbf{0.500} = 4/8$

> **Nhận xét của sinh viên:** Trọng số của mẫu 5 tăng vọt từ $0.2 \to 0.5$ (chiếm trọn 50% tổng trọng số toàn tập) vì nó là mẫu duy nhất bị phân loại sai!

---

#### VÒNG LẶP 2 ($t = 2$):

1. **Tìm mô hình $h_2$:**
   Hiện tại mẫu 5 có trọng số $0.5$, mô hình buộc phải dự đoán đúng mẫu 5 nếu không sai số sẽ $\ge 0.5$.
   Xét đường cắt nằm ngang theo trục $X_2$ tại $\theta = 3.5$:
   $$h_2(x) = \begin{cases} +1 & \text{nếu } X_2 > 3.5 \\ -1 & \text{nếu } X_2 \le 3.5 \end{cases}$$
   Kiểm tra dự đoán:
   - Mẫu 1 ($X_2=4.0 > 3.5$): Đoán $+1$ $\rightarrow$ Đúng ($y_1 = +1$).
   - Mẫu 2 ($X_2=1.0 \le 3.5$): Đoán $-1$ $\rightarrow$ **SAI** (Thực tế $y_2 = +1$).
   - Mẫu 3 ($X_2=3.0 \le 3.5$): Đoán $-1$ $\rightarrow$ Đúng ($y_3 = -1$).
   - Mẫu 4 ($X_2=1.0 \le 3.5$): Đoán $-1$ $\rightarrow$ Đúng ($y_4 = -1$).
   - Mẫu 5 ($X_2=4.0 > 3.5$): Đoán $+1$ $\rightarrow$ Đúng ($y_5 = +1$).

2. **Tính sai số $\epsilon_2$:**
   Chỉ có mẫu 2 bị sai:
   $$\epsilon_2 = w_2(2) = 0.125 = \frac{1}{8}$$

3. **Tính trọng số mô hình $\alpha_2$:**
   $$\alpha_2 = \frac{1}{2} \ln\left( \frac{1 - 1/8}{1/8} \right) = \frac{1}{2} \ln\left( \frac{7/8}{1/8} \right) = \frac{1}{2} \ln(7) \approx \mathbf{0.9730}$$

4. **Cập nhật trọng số mẫu:**
   - Hệ số giảm: $e^{-\alpha_2} = \frac{1}{\sqrt{7}} \approx 0.3780$
   - Hệ số tăng: $e^{+\alpha_2} = \sqrt{7} \approx 2.6458$

   Hằng số chuẩn hóa:
   $$Z_2 = 2\sqrt{\epsilon_2(1 - \epsilon_2)} = 2\sqrt{\frac{1}{8} \times \frac{7}{8}} = 2 \times \frac{\sqrt{7}}{8} = \frac{\sqrt{7}}{4} \approx 0.6614$$

   Tính trọng số chuẩn hóa mới $w_3(i)$:
   - Với mẫu đúng ($i = 1, 3, 4$):
     $$w_3(i) = \frac{\frac{1}{8} \times \frac{1}{\sqrt{7}}}{\frac{\sqrt{7}}{4}} = \frac{1}{8} \times \frac{1}{\sqrt{7}} \times \frac{4}{\sqrt{7}} = \frac{4}{56} = \frac{\mathbf{1}}{\mathbf{14}} \approx 0.0714$$
   - Với mẫu sai ($i = 2$):
     $$w_3(2) = \frac{\frac{1}{8} \times \sqrt{7}}{\frac{\sqrt{7}}{4}} = \frac{1}{8} \times \sqrt{7} \times \frac{4}{\sqrt{7}} = \frac{4}{8} = \frac{1}{2} = \frac{\mathbf{7}}{\mathbf{14}} = \mathbf{0.5000}$$
   - Với mẫu 5 (đúng ở vòng này):
     $$w_3(5) = \frac{\frac{4}{8} \times \frac{1}{\sqrt{7}}}{\frac{\sqrt{7}}{4}} = \frac{4}{8} \times \frac{1}{\sqrt{7}} \times \frac{4}{\sqrt{7}} = \frac{16}{56} = \frac{\mathbf{4}}{\mathbf{14}} \approx 0.2857$$

   Kiểm tra tổng: $\frac{1}{14} + \frac{7}{14} + \frac{1}{14} + \frac{1}{14} + \frac{4}{14} = \frac{14}{14} = 1.0$ (Khớp tuyệt đối!).

> **Nhận xét của sinh viên:** Mẫu 2 giờ đây trở thành mẫu có trọng số cao nhất ($0.5$), buộc vòng lặp tiếp theo phải tập trung giải cứu mẫu 2!

---

#### VÒNG LẶP 3 ($t = 3$):

1. **Tìm mô hình $h_3$:**
   Ta chọn đường cắt nằm ngang theo trục $X_2$ tại $\theta = 4.5$:
   $$h_3(x) = \begin{cases} +1 & \text{nếu } X_2 \le 4.5 \\ -1 & \text{nếu } X_2 > 4.5 \end{cases}$$
   Kiểm tra dự đoán:
   - Mẫu 1 ($X_2=4.0 \le 4.5$): Đoán $+1$ $\rightarrow$ Đúng ($y_1 = +1$).
   - Mẫu 2 ($X_2=1.0 \le 4.5$): Đoán $+1$ $\rightarrow$ Đúng ($y_2 = +1$).
   - Mẫu 3 ($X_2=3.0 \le 4.5$): Đoán $+1$ $\rightarrow$ **SAI** (Thực tế $y_3 = -1$).
   - Mẫu 4 ($X_2=1.0 \le 4.5$): Đoán $+1$ $\rightarrow$ **SAI** (Thực tế $y_4 = -1$).
   - Mẫu 5 ($X_2=4.0 \le 4.5$): Đoán $+1$ $\rightarrow$ Đúng ($y_5 = +1$).

2. **Tính sai số $\epsilon_3$:**
   Mô hình sai ở mẫu 3 và 4:
   $$\epsilon_3 = w_3(3) + w_3(4) = \frac{1}{14} + \frac{1}{14} = \frac{2}{14} = \frac{1}{7} \approx 0.1429$$

3. **Tính trọng số mô hình $\alpha_3$:**
   $$\alpha_3 = \frac{1}{2} \ln\left( \frac{1 - 1/7}{1/7} \right) = \frac{1}{2} \ln\left( \frac{6/7}{1/7} \right) = \frac{1}{2} \ln(6) \approx \mathbf{0.8959}$$

---

### 3.3 Tổng hợp mô hình & Minh chứng 100% độ chính xác

Hàm quyết định tổng thể của Ensemble AdaBoost là:
$$H(x) = \text{sign}\Big( \alpha_1 h_1(x) + \alpha_2 h_2(x) + \alpha_3 h_3(x) \Big)$$
Với:
$$\alpha_1 \approx 0.6931, \quad \alpha_2 \approx 0.9730, \quad \alpha_3 \approx 0.8959$$

Ta lập bảng tổng hợp số điểm (Score) cho từng mẫu:

| Mẫu ($i$) | $y$ thực | $h_1(x)$ | $h_2(x)$ | $h_3(x)$ | Điểm tích lũy: $S = \sum \alpha_t h_t(x)$ | Dự đoán $H(x) = \text{sign}(S)$ | Kết luận |
|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---:|
| **1** | **+1** | $+1$ | $+1$ | $+1$ | $+0.6931 + 0.9730 + 0.8959 = \mathbf{+2.5620}$ | **+1** | **CHÍNH XÁC** |
| **2** | **+1** | $+1$ | $-1$ | $+1$ | $+0.6931 - 0.9730 + 0.8959 = \mathbf{+0.6160}$ | **+1** | **CHÍNH XÁC** |
| **3** | **-1** | $-1$ | $-1$ | $+1$ | $-0.6931 - 0.9730 + 0.8959 = \mathbf{-0.7702}$ | **-1** | **CHÍNH XÁC** |
| **4** | **-1** | $-1$ | $-1$ | $+1$ | $-0.6931 - 0.9730 + 0.8959 = \mathbf{-0.7702}$ | **-1** | **CHÍNH XÁC** |
| **5** | **+1** | $-1$ | $+1$ | $+1$ | $-0.6931 + 0.9730 + 0.8959 = \mathbf{+1.1758}$ | **+1** | **CHÍNH XÁC** |

> **KẾT QUẢ NGOẠN MỤC:**
> Dù mỗi mô hình đơn lẻ (Decision Stump) đều có sai sót, nhưng khi kết hợp lại thông qua trọng số $\alpha_t$, mô hình Ensemble đạt độ chính xác **5/5 (100%)**!

---

### 3.4 Mở rộng: Giải thuật SAMME cho bài toán Đa lớp (Multi-class)

Trong bài toán đa lớp ($K > 2$ lớp, ví dụ bài toán nhận diện 10 chữ số $0..9$):
* Một mô hình ngẫu nhiên chỉ đạt độ chính xác $1/K = 10\%$, tức tỷ lệ sai số là $90\%$.
* Nếu dùng công thức AdaBoost nhị phân cũ, $\epsilon_t > 0.5$ sẽ khiến $\alpha_t < 0$ và thuật toán đổ vỡ!
* **Giải pháp SAMME (Zhu et al., 2009):** Bổ sung thêm hạng tử $\ln(K - 1)$:
  $$\alpha_t = \ln\left( \frac{1 - \epsilon_t}{\epsilon_t} \right) + \ln(K - 1)$$
  - Điều kiện để $\alpha_t > 0$ chỉ còn là $(1 - \epsilon_t) > \frac{1}{K}$.
  - Điều này có nghĩa: **Chỉ cần một mô hình yếu đoán tốt hơn việc đoán mò ($> 10\%$) là đã đóng góp tích cực vào tập thể!**

---

## 4. THUẬT TOÁN 3: STACKING (STACKED GENERALIZATION)

### 4.1 Cơ sở toán học & Cơ chế Out-Of-Fold (OOF) chống rò rỉ dữ liệu

Khác với Bagging và Boosting thường dùng cùng một loại mô hình cơ sở (Homogeneous), **Stacking** kết hợp các mô hình có bản chất thuật toán hoàn toàn khác nhau (Heterogeneous) như Cây quyết định, KNN, SVM, Logistic Regression.

Kiến trúc Stacking gồm 2 tầng:
* **Tầng 0 (Base Learners):** Gồm $M$ mô hình cơ sở khác nhau ($h^{(1)}, h^{(2)}, \dots, h^{(M)}$).
* **Tầng 1 (Meta Learner):** Một mô hình cấp cao (thường là Logistic Regression hoặc Linear Regression) học cách tổng hợp các dự đoán từ Tầng 0.

```
       [Tập Dữ Liệu Huấn Luyện Gốc]
               │
      ┌────────┼────────┐
      ▼        ▼        ▼
   [Base 1] [Base 2] [Base 3]       (TẦNG 0: Đa dạng hóa thuật toán)
      │        │        │
      └────────┼────────┘
               ▼
     [Meta-Features (Z)]             (Tạo từ K-Fold OOF)
               │
               ▼
        [Meta-Learner]               (TẦNG 1: Học trọng số kết hợp)
               │
               ▼
       [Dự Đoán Cuối Cùng]
```

#### NGUY CƠ RÒ RỈ DỮ LIỆU (DATA LEAKAGE):
Nếu ta huấn luyện các Base Learners trên tập $D$, sau đó dùng chính các Base Learners này dự đoán lại trên $D$ để làm dữ liệu cho Meta-Learner $\rightarrow$ Các Base Learners (đặc biệt là Cây hoặc KNN với $k=1$) sẽ bị Overfitting, cho kết quả dự đoán đúng $100\%$. Meta-Learner sẽ tin tưởng mù quáng vào mô hình bị Overfit và mất hoàn toàn khả năng khái quát hóa trên tập kiểm thử!

#### GIẢI PHÁP: K-Fold Out-Of-Fold (OOF):
1. Chia tập train $D$ thành $K$ phần (folds) bằng nhau.
2. Với mỗi fold $k \in \{1, \dots, K\}$:
   - Huấn luyện Base Learners trên $K-1$ fold còn lại.
   - Dự đoán trên fold $k$ bị giữ lại.
   - Điền kết quả dự đoán vào ma trận đặc trưng mới $Z$ tại đúng vị trí của fold $k$.
3. Sau khi quét hết $K$ folds, ma trận $Z$ có đúng $n$ hàng. Ta huấn luyện Meta-Learner trên $(Z, y)$.
4. Cuối cùng, huấn luyện lại toàn bộ Base Learners trên **100% tập dữ liệu train $D$** để sẵn sàng cho suy luận.

---

### 4.2 Ví dụ chạy tay chi tiết 2 tầng (Level-0 & Level-1)

Xét tập dữ liệu 1 chiều gồm $n = 6$ mẫu, nhãn $y \in \{0, 1\}$:

| Chỉ số ($i$) | $X$ | Nhãn thực tế ($y$) | Phân chia Fold ($K=3$) |
|:---:|:---:|:---:|:---:|
| 1 | 1.0 | 0 | **Fold 1** (Mẫu kiểm định OOF đợt 1) |
| 2 | 2.0 | 0 | **Fold 1** (Mẫu kiểm định OOF đợt 1) |
| 3 | 3.0 | 1 | **Fold 2** (Mẫu kiểm định OOF đợt 2) |
| 4 | 5.0 | 0 | **Fold 2** (Mẫu kiểm định OOF đợt 2) |
| 5 | 6.0 | 1 | **Fold 3** (Mẫu kiểm định OOF đợt 3) |
| 6 | 7.0 | 1 | **Fold 3** (Mẫu kiểm định OOF đợt 3) |

Hai mô hình cơ sở ở Tầng 0:
* **Mô hình A:** $1$-Nearest Neighbor (1-NN) dựa trên khoảng cách $|x_i - x_j|$.
* **Mô hình B:** Decision Stump (ngưỡng cắt tối ưu hóa sai số trên tập train).

---

#### ĐỢT 1: Giữ lại Fold 1 $\{1, 2\}$, Huấn luyện trên Fold 2 + 3 $\{3, 4, 5, 6\}$
* Dữ liệu huấn luyện:
  $$(3.0, 1), (5.0, 0), (6.0, 1), (7.0, 1)$$
* **Mô hình A (1-NN):**
  - Dự đoán cho Mẫu 1 ($X=1.0$): Điểm gần nhất trong tập train là $X=3.0$ (khoảng cách $|1.0 - 3.0| = 2.0$). Nhãn của điểm này là $1$. $\rightarrow \hat{y}_A^{(1)} = 1$.
  - Dự đoán cho Mẫu 2 ($X=2.0$): Điểm gần nhất là $X=3.0$ (khoảng cách $|2.0 - 3.0| = 1.0$). Nhãn là $1$. $\rightarrow \hat{y}_A^{(2)} = 1$.
* **Mô hình B (Decision Stump):**
  - Tìm ngưỡng cắt tốt nhất trên $\{3.0, 5.0, 6.0, 7.0\}$:
  - Ngưỡng tối ưu: $X \le 5.5 \to 0$, $X > 5.5 \to 1$ (chỉ sai mẫu $3.0$).
  - Dự đoán cho Mẫu 1 ($X=1.0 \le 5.5$): $\rightarrow \hat{y}_B^{(1)} = 0$.
  - Dự đoán cho Mẫu 2 ($X=2.0 \le 5.5$): $\rightarrow \hat{y}_B^{(2)} = 0$.

$\rightarrow$ Điền vào hàng 1 và 2 của ma trận $Z$:
$$Z[1, :] = [1, 0], \quad Z[2, :] = [1, 0]$$

---

#### ĐỢT 2: Giữ lại Fold 2 $\{3, 4\}$, Huấn luyện trên Fold 1 + 3 $\{1, 2, 5, 6\}$
* Dữ liệu huấn luyện:
  $$(1.0, 0), (2.0, 0), (6.0, 1), (7.0, 1)$$
* **Mô hình A (1-NN):**
  - Dự đoán cho Mẫu 3 ($X=3.0$): Điểm gần nhất là $X=2.0$ (khoảng cách $1.0$). Nhãn là $0$. $\rightarrow \hat{y}_A^{(3)} = 0$.
  - Dự đoán cho Mẫu 4 ($X=5.0$): Điểm gần nhất là $X=6.0$ (khoảng cách $1.0$). Nhãn là $1$. $\rightarrow \hat{y}_A^{(4)} = 1$.
* **Mô hình B (Decision Stump):**
  - Ngưỡng phân chia hoàn hảo trên tập này là $\theta = 4.0$ ($X \le 4.0 \to 0$, $X > 4.0 \to 1$).
  - Dự đoán cho Mẫu 3 ($X=3.0 \le 4.0$): $\rightarrow \hat{y}_B^{(3)} = 0$.
  - Dự đoán cho Mẫu 4 ($X=5.0 > 4.0$): $\rightarrow \hat{y}_B^{(4)} = 1$.

$\rightarrow$ Điền vào hàng 3 và 4 của ma trận $Z$:
$$Z[3, :] = [0, 0], \quad Z[4, :] = [1, 1]$$

---

#### ĐỢT 3: Giữ lại Fold 3 $\{5, 6\}$, Huấn luyện trên Fold 1 + 2 $\{1, 2, 3, 4\}$
* Dữ liệu huấn luyện:
  $$(1.0, 0), (2.0, 0), (3.0, 1), (5.0, 0)$$
* **Mô hình A (1-NN):**
  - Dự đoán cho Mẫu 5 ($X=6.0$): Điểm gần nhất là $X=5.0$ (khoảng cách $1.0$). Nhãn là $0$. $\rightarrow \hat{y}_A^{(5)} = 0$.
  - Dự đoán cho Mẫu 6 ($X=7.0$): Điểm gần nhất là $X=5.0$ (khoảng cách $2.0$). Nhãn là $0$. $\rightarrow \hat{y}_A^{(6)} = 0$.
* **Mô hình B (Decision Stump):**
  - Ngưỡng cắt tối ưu: $X \le 2.5 \to 0$, $X > 2.5 \to 1$ (chỉ sai mẫu $5.0$).
  - Dự đoán cho Mẫu 5 ($X=6.0 > 2.5$): $\rightarrow \hat{y}_B^{(5)} = 1$.
  - Dự đoán cho Mẫu 6 ($X=7.0 > 2.5$): $\rightarrow \hat{y}_B^{(6)} = 1$.

$\rightarrow$ Điền vào hàng 5 và 6 của ma trận $Z$:
$$Z[5, :] = [0, 1], \quad Z[6, :] = [0, 1]$$

---

#### TỔNG HỢP MA TRẬN META-FEATURES ($Z$):
Sau khi hoàn thành 3 folds OOF, ta thu được tập dữ liệu huấn luyện mới cho Tầng 1:

| Mẫu ($i$) | Đặc trưng mới $Z_A$ (Dự đoán từ Mô hình A) | Đặc trưng mới $Z_B$ (Dự đoán từ Mô hình B) | Nhãn mục tiêu ($y$) |
|:---:|:---:|:---:|:---:|
| 1 | 1 | 0 | **0** |
| 2 | 1 | 0 | **0** |
| 3 | 0 | 0 | **1** |
| 4 | 1 | 1 | **0** |
| 5 | 0 | 1 | **1** |
| 6 | 0 | 1 | **1** |

#### HUẤN LUYỆN META-LEARNER (TẦNG 1):
Giả sử Meta-Learner là một mô hình tuyến tính có ngưỡng phân lớp:
$$\hat{y} = \mathbb{I}(w_A \cdot Z_A + w_B \cdot Z_B + b > 0)$$
Nhìn vào bảng dữ liệu $Z$:
- Khi $Z_A = 1$ (Mẫu 1, 2, 4), nhãn thực tế $y$ luôn luôn bằng **0**.
- Khi $Z_B = 1$ và $Z_A = 0$ (Mẫu 5, 6), nhãn thực tế $y$ bằng **1**.
$\rightarrow$ Meta-Learner sẽ tự động học được rằng: **"Mô hình A thường xuyên báo động giả, nên gán trọng số âm cho A ($w_A < 0$); Mô hình B đáng tin cậy hơn, gán trọng số dương cho B ($w_B > 0$)"**.

---

### 4.3 Quá trình suy luận trên mẫu kiểm thử mới (Inference)

Khi có một điểm kiểm thử mới $X_{test} = 6.5$:
1. **Bước 1:** Cho 2 mô hình cơ sở (đã được huấn luyện lại trên toàn bộ 6 mẫu ban đầu) dự đoán:
   - $h_A(6.5) = 1$ (điểm gần nhất là $X=7.0, y=1$).
   - $h_B(6.5) = 1$ (vì $6.5 > 2.5$).
2. **Bước 2:** Vector đặc trưng meta mới tạo thành:
   $$z_{test} = [Z_A=1, Z_B=1]$$
3. **Bước 3:** Meta-Learner nhận $z_{test}$ và đưa ra phán quyết cuối cùng $\hat{y}_{test}$.

---

## 5. BẢNG SO SÁNH TỔNG HỢP 3 THUẬT TOÁN

| Tiêu chí | BAGGING | BOOSTING (ADABOOST) | STACKING |
|---|---|---|---|
| **Mục tiêu chính** | **Giảm Variance** (tránh Overfitting) | **Giảm Bias** (tránh Underfitting) | **Tối ưu hóa tổng quát** (kết hợp đa dạng thế mạnh) |
| **Mô hình cơ sở** | Cùng loại, phức tạp (Cây quyết định sâu) | Cùng loại, rất đơn giản (Decision Stump) | Khác loại hoàn toàn (Tree, KNN, LogReg, SVM...) |
| **Cách huấn luyện** | **Song song độc lập** (Parallel) | **Tuần tự sửa sai** (Sequential) | **Phân tầng** (Hierarchical / Multi-level) |
| **Cơ chế lấy mẫu** | Bootstrap (lấy mẫu có hoàn lại) | Giữ nguyên tập mẫu, cập nhật trọng số mẫu $w_i$ | K-Fold Out-Of-Fold (OOF) |
| **Cơ chế suy luận** | Biểu quyết đa số / Trung bình cộng | Biểu quyết có trọng số $\alpha_t$ | Mô hình Meta-Learner ra quyết định |
| **Độ nhạy với Nhiễu** | **Rất tốt** (chống nhiễu, hạn chế overfit) | **Nhạy cảm** (dễ gán trọng số cao cho nhiễu/outlier) | Tùy thuộc vào Base Learners & Meta-Learner |

---

## 6. BÍ KÍP & BẪY KINH ĐIỂN CẦN TRÁNH KHI LÀM BÀI THI

### 1. Bẫy trong Bagging:
* **Nhầm lẫn giữa rút mẫu có hoàn lại và không hoàn lại:** Luôn nhớ trong Bagging là **CÓ HOÀN LẠI (with replacement)**, do đó một mẫu có thể xuất hiện nhiều lần trong cùng một cây.
* **Quên tính OOB:** Khi đề bài yêu cầu tính sai số OOB, tuyệt đối không tính dự đoán của những cây đã dùng mẫu đó để huấn luyện!

### 2. Bẫy trong AdaBoost:
* **Nhầm lẫn công thức $\alpha_t$:**
  - AdaBoost nhị phân ($y \in \{-1, +1\}$): $\alpha_t = \mathbf{\frac{1}{2}} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$. Rất nhiều sinh viên quên mất hệ số $\mathbf{1/2}$!
  - Khi tính nhẩm trong phòng thi, nếu $\epsilon_t = 0.2 \rightarrow \frac{1-\epsilon}{\epsilon} = 4 \rightarrow \alpha_t = \frac{1}{2}\ln(4) = \ln(2) \approx 0.693$.
* **Quên bước chuẩn hóa $Z_t$:** Tổng trọng số sau mỗi vòng **BẮT BUỘC** phải bằng $1.0$. Nếu tính ra tổng khác $1$, hãy kiểm tra lại phép chia cho $Z_t$.
* **Sai dấu trong cập nhật trọng số:**
  - Dự đoán ĐÚNG $\rightarrow$ nhân $e^{-\alpha_t}$ (giảm).
  - Dự đoán SAI $\rightarrow$ nhân $e^{+\alpha_t}$ (tăng).

### 3. Bẫy trong Stacking:
* **Rò rỉ dữ liệu (Data Leakage):** Đây là câu hỏi vấn đáp kinh điển! Nếu dùng toàn bộ tập train để vừa huấn luyện Base vừa tạo Meta-features thì mô hình sẽ bị Overfit nặng nề. Phải trả lời đúng từ khóa: **K-Fold Out-Of-Fold (OOF)**.
* **Huấn luyện lại (Refit):** Sau khi đã tạo xong Meta-features và huấn luyện xong Meta-Learner, đừng quên bước **huấn luyện lại các Base Learners trên 100% dữ liệu gốc** để phục vụ việc dự đoán mẫu mới.
