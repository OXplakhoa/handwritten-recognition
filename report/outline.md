# Báo cáo (Report) — MNIST CNN handwriting recognition

## 1. Giới thiệu / Introduction
- Mục tiêu bài tập: nhận dạng chữ số viết tay (MNIST) using CNN.
- Dữ liệu: MNIST, 70k images (60k train / 10k test).

## 2. Dữ liệu và tiền xử lý / Data & Preprocessing
- Mô tả dataset, kích thước ảnh, số lớp.
- Tiền xử lý cơ bản: chuẩn hóa pixel, one-hot labels.
- Chiến lược chia dữ liệu: train/validation/test.

## 3. Kiến trúc mô hình / Model architectures
- Baseline: mô tả ngắn (Conv2D + MaxPooling + dense).
- Tuned: thay đổi (dropout, batchnorm, augmentation) and justification.

## 4. Huấn luyện / Training
- Hyperparameters: batch size, epochs, optimizer, learning rate.
- Early stopping, checkpointing.

## 5. Kết quả / Results
- Training/validation curves.
- Test accuracy trên test set (10k samples).
- Confusion matrix và ví dụ dự đoán.

## 6. Mục thảo luận / Discussion
- Phân tích các lỗi thường gặp, tác động của augmentation.
- Hạn chế và hướng cải tiến.

## 7. Kết luận / Conclusion
- Tổng kết kết quả, bài học rút ra.

## 8. Tài liệu tham khảo / References
- Keras MNIST docs, tutorial links, course references.
