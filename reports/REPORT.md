# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

Ngày chạy: 07/11/2026

Runtime Colab: CPU

Python / PyTorch / Ultralytics: 
Python: 3.13.18
PyTorch: 2.14.1+cpu
Ultralytics: 8.4.145

Checkpoint: yolo11n-cls.pt, yolo11n.pt, yolo11n-seg.pt

Ngưỡng detection/segmentation dùng khi tạo JSON: 0.35 (giữ nguyên, không thay đổi)

Thay đổi so với notebook nguồn: Không

## 1. Phân loại ảnh

- Record hạng 1 (ảnh traffic): class_id=468 ; class_name=cab ; rank=1 ; score=0.510915 ; taxonomy_name=ImageNet-1K
- Record này nói về cả bức ảnh, không phải một vật riêng. Phân loại chỉ gán một nhãn cho toàn ảnh và không có tọa độ, nên không cho biết xe "cab" nằm ở vị trí nào.
- Danh sách tên lớp đi kèm taxonomy/checkpoint (ImageNet-1K), model không tự nghĩ ra tên lớp.
- Cần ghi cả class_id, class_name và taxonomy_name vì cùng một id hoặc một tên có thể mang nghĩa khác ở taxonomy khác. Ở đây id 468 chỉ có nghĩa là "cab" trong ImageNet-1K.
- Ảnh traffic có nhiều đối tượng (xe, đường, biển báo), nên guideline phải nói rõ tiêu chí chọn một lớp, ví dụ chọn chủ thể chính, chiếm diện tích lớn nhất hoặc nổi bật nhất, và cách xử lý khi hai vật ngang nhau.
- Không được lấy score của model làm ground truth. Score chỉ là mức "tin" của model (ở đây khoảng 0.51, tức model không chắc chắn), và kể cả khi score cao model vẫn có thể sai. Ground truth là nhãn do người gán theo guideline.

## 2. Phát hiện vật thể

- Record (ảnh kitchen): class_name=bowl ; score=0.719062 ; bbox_xyxy=[32.65, 342.12, 100.16, 384.93] ; bbox_width=67.51 ; bbox_height=42.81
- Góc trên-trái của box ở (32.65, 342.12) và góc dưới-phải ở (100.16, 384.93), tính bằng pixel với gốc (0,0) ở góc trên-trái của ảnh. Box rộng 67.51 px (100.16 − 32.65) và cao 42.81 px (384.93 − 342.12).
- Ngưỡng 0.20: 17 prediction ; ngưỡng 0.60: 6 prediction (ngưỡng mặc định 0.35: 11 prediction).
- Khi hạ ngưỡng từ 0.35 xuống 0.20, số prediction tăng từ 11 lên 17, xuất hiện thêm các vật có score thấp như spoon. Reviewer phải duyệt thêm 6 hộp, loại các hộp sai (false positive), hộp trùng nhau và hộp lệch.
- Quy tắc box: box ôm sát phần nhìn thấy của vật, không chừa nhiều nền; mỗi vật một box riêng; không gộp nhiều vật vào một box.
- Trên ảnh kitchen, cái cốc ở trên bàn là chỗ khó vì dễ bị nhầm với bát. Nếu vật bị che một phần hoặc cắt ở mép ảnh và tình huống này lặp lại nhiều, tôi sẽ viết thành quy tắc trong guideline (chỉ vẽ box quanh phần nhìn thấy). Với tình huống mơ hồ chưa có quy tắc, ví dụ không rõ vật là cốc hay bát, tôi sẽ đưa lên hỏi Lab Coach (escalation) thay vì tự đoán, vì mỗi người đoán một kiểu sẽ làm nhãn không nhất quán.

## 3. Phân đoạn theo từng đối tượng

- Record (ảnh kitchen): instance_id=kitchen-006 ; class_name=bowl ; score=0.528202 ; polygon_point_count=84 ; polygon_xy (4 điểm đầu)=[[29.0, 303.0], [28.0, 304.0], [26.0, 304.0], [25.0, 305.0]]
- Box chỉ là hình chữ nhật bao quanh vật, còn polygon (84 điểm) cho đường viền thật của vật. Nhờ đó biết được hình dạng và diện tích vật chiếm thực tế, không lẫn phần nền ở các góc của box.
- instance_id (kitchen-006) dùng để phân biệt từng vật riêng lẻ trong cùng một ảnh, ví dụ hai cái bát là hai instance khác nhau. Nó không phải class ID vì không cho biết vật thuộc loại gì (cả hai đều là "bowl"), và không phải tracking ID vì không dùng để theo dõi một vật qua nhiều khung hình.
- Quy tắc vẽ biên mask: vẽ theo mép ngoài của vật, không gồm bóng đổ; vật bị che thì chỉ vẽ phần nhìn thấy; mỗi instance một polygon riêng.
- Trên ảnh kitchen, hai cái bát chạm nhau ở góc dưới bên trái nên ranh giới giữa chúng khó xác định. Nếu ranh giới nhìn thấy rõ, tôi dùng quy tắc chung trong guideline (mỗi bát một polygon, vẽ biên theo chỗ tiếp xúc). Nếu không nhìn rõ đường ranh giới, tôi escalation lên Lab Coach thay vì tự đoán.

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| Phân loại ảnh | 1 nhãn cho cả ảnh | Ảnh traffic có nhiều vật; rank 1 là "cab" nhưng score chỉ 0.51 | Chọn 1 lớp theo quy tắc chủ thể chính trong guideline | Nhãn có đúng quy tắc chọn chủ thể chính không |
| Phát hiện vật thể | 1 box + 1 lớp cho mỗi vật | Cái cốc trên bàn dễ nhầm với bát; hạ ngưỡng xuống 0.20 cho nhiều hộp thừa | Vẽ box sát phần nhìn thấy của vật, gán đúng lớp | Box có khít không, có sót hoặc thừa vật không |
| Instance segmentation | 1 polygon + 1 lớp cho mỗi instance | Hai cái bát chạm nhau ở góc dưới trái (kitchen-006 và bát bên cạnh) | Vẽ biên riêng cho từng bát theo quy tắc biên mask | Biên có bám vật không, hai instance có bị dính nhau không |

## 5. An toàn dữ liệu

- Tôi không đưa họ tên, MSSV, email, số điện thoại hay dữ liệu cá nhân vào báo cáo và output, chỉ dùng ảnh mẫu trong phạm vi bài lab.
- Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng lại và báo cho Lab Coach/giảng viên.

## 6. Checklist trước khi nộp link repo

- [v] REPORT.md
- [v] day1_lab_outputs/classification_predictions.json
- [v] day1_lab_outputs/detection_predictions.json
- [v] day1_lab_outputs/segmentation_predictions.json
- [v] day1_lab_outputs/IMAGE_ATTRIBUTION.md
- [v] day1_lab_outputs/visuals/classification_top5.png
- [v] day1_lab_outputs/visuals/detection_predictions.png
- [v] day1_lab_outputs/visuals/segmentation_prediction.png
- [v] Ô validation cuối notebook báo PASS
- [v] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [v] Đã nộp link repository trên LMS