# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Vũ Duy Điệp (2A202602703)
**Khóa:** K4 - Track 3A

---

## Cấu hình chạy

| | Naive Baseline | Production |
|---|---|---|
| Chunking | `chunk_basic` (paragraph, 500 ký tự) | `chunk_hierarchical` (child 256 / parent 2048) |
| Enrichment | — | M5 combined single-call (1 call/chunk, 100 chunk, 0 fallback) |
| Retrieval | Dense `bge-m3` top-3 | BM25 (`underthesea`) + Dense `bge-m3`, RRF (k=60), top-20 |
| Rerank | — | `bge-reranker-v2-m3`, giữ 3 **parent** khác nhau |
| Generator | `gemini-3.5-flash-lite`, temperature 0 | như baseline |
| RAGAS judge | `gemini-3.1-flash-lite` + `gemini-embedding-001` | như baseline (cùng judge để so sánh công bằng) |

Corpus: 26/28 tài liệu được index. 2 PDF scan (`BCTC.pdf`, Nghị định 13/2023) không có text layer nên bị bỏ qua vì chưa có OCR.

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8250 | **0.9733** | +0.1483 |
| Answer Relevancy | 0.7156 | **0.8558** | +0.1402 |
| Context Precision | 0.8000 | **0.8500** | +0.0500 |
| Context Recall | 0.8417 | **0.9250** | +0.0833 |

Cả 20/20 câu được chấm đủ 4 metric ở cả hai lần chạy (`status: ok`, `metric_coverage` = 20).

**Nhận xét:**
- Faithfulness và Answer Relevancy tăng mạnh nhất. Production trả về cả **parent chunk** (section/tài liệu đầy đủ kèm header phiên bản), còn baseline chỉ đưa 3 đoạn ≤ 500 ký tự cắt theo paragraph. LLM có đủ điều kiện/ngoại lệ trong cùng context nên ít phải suy diễn ngoài context, và trả lời đúng trọng tâm hơn.
- Context Recall tăng nhờ hai yếu tố: hybrid search (BM25 bắt từ khóa chính xác như "PVI", "MFA", "tạm ứng", "Junior"; dense bắt cách diễn đạt khác nghĩa), và context là parent nên chứa trọn các dữ kiện mà ground truth cần.
- Context Precision tăng ít nhất (+0.05) và là metric thấp nhất của production. Nguyên nhân: pipeline **luôn** trả đủ 3 parent khác nhau. Câu hỏi chỉ cần 1 tài liệu vẫn bị kèm 1–2 tài liệu không liên quan, và bản chính sách cũ đôi khi được xếp trên bản mới. 4/5 failure dưới đây đều dính lỗi này.

## Latency Breakdown (Production)

| Bước | Thời gian | Ghi chú |
|---|---|---|
| M1 Chunking (26 docs → 100 child) | 0.3 s | CPU |
| M5 Enrichment (100 chunk) | 807.2 s | 8.1 s/chunk, **gồm 4.2 s throttle** cho free-tier 15 RPM. Không throttle thì khoảng 3.9 s/chunk |
| M2 Indexing (BM25 + bge-m3 + Qdrant) | 46.4 s | bge-m3 encode trên CPU |
| **Per query:** Hybrid search | 403.7 ms | BM25 + encode query bge-m3 + Qdrant `query_points` + RRF |
| **Per query:** Rerank (≈20 candidate) | 5068.7 ms | `bge-reranker-v2-m3` (568M) trên **CPU**. Đây là bottleneck lớn nhất |
| **Per query:** LLM generate | 5519.7 ms | gồm 4.2 s throttle, nên LLM thực tế khoảng 1.3 s |
| RAGAS (20 câu × 4 metric) | 846.6 s | judge throttle 4.2 s/call |
| Tổng pipeline | 1920.3 s | baseline: 1016.4 s |

Ý nghĩa: bỏ phần throttle đi thì latency một query vào khoảng 0.4 + 5.1 + 1.3 ≈ **6.8 s**, trong đó rerank chiếm khoảng 75%. Muốn đưa vào production cần chạy reranker trên GPU, giảm số candidate (top-20 → top-10), hoặc dùng FlashRank cho query đơn giản.

---

## Bottom-5 Failures

Thứ tự lấy từ `reports/ragas_report.json → failures` (sắp theo điểm trung bình 4 metric, thấp nhất trước).

### #1 — Multi-hop (nghỉ phép + lương), avg = 0.375
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên = 18 ngày (v2024). Lương Senior (P3–P4): 20–35 triệu VNĐ/tháng.
- **Got:** "…được **18 ngày phép** (15 + 3). Không tìm thấy thông tin về mức lương cụ thể…"
- **Worst metric:** context_precision
- **Error Tree:** Output sai một nửa → Context đúng? **Thiếu**: 3 context là nghỉ phép v2024, nghỉ phép v2023 và nghỉ phép không lương; không có `bang_luong_2024.md` → Query OK? **Không**: một query chứa 2 ý, và vế "nghỉ phép" chiếm ưu thế ở cả BM25, dense lẫn reranker → lỗi ở **Retrieval (query understanding)**.
- **Root cause:** câu hỏi multi-hop được xử lý như một truy vấn đơn. Reranker chấm cả câu, nên mọi tài liệu nghỉ phép đều điểm cao hơn bảng lương và lấp đầy 3 slot top-k. Phần lương mà LLM không trả lời được là do thiếu context; đó không phải hallucination (faithfulness vẫn đúng).
- **Suggested fix:** **Query decomposition.** Dùng LLM tách câu hỏi thành sub-query ("ngày phép 9 năm thâm niên", "khung lương Senior"), retrieve riêng từng sub-query rồi gộp bằng RRF. Ngoài ra, giới hạn tối đa 1 parent cho mỗi "chủ đề" (theo metadata `category`/`source`) để tránh 3 slot cùng một chủ đề.

### #2 — Liên kết 2 tài liệu (lương ↔ phân loại dữ liệu), avg = 0.724
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Dữ liệu **Bí mật (cấp 3)**: cấm chia sẻ với đồng nghiệp, phải mã hóa khi truyền, truy cập theo need-to-know.
- **Got:** "…thông tin lương thuộc cấp độ **Bí mật**." Đúng nhưng thiếu cấp 3 và các yêu cầu xử lý.
- **Worst metric:** context_precision
- **Error Tree:** Output đúng nhưng chưa đủ → Context đúng? **Có**: `ky_luong.md` và `phan_loai_du_lieu.md` đều có mặt, nhưng context thứ 3 là `bang_luong_2024.md` (khung lương, không liên quan) → Query OK? Có → lỗi ở **Rerank cut-off** và **Prompt**.
- **Root cause:** (1) Pipeline luôn trả đủ 3 parent kể cả khi parent thứ 3 có điểm rerank thấp, gây nhiễu nên precision giảm. (2) Prompt chỉ yêu cầu "trả lời dựa trên context" nên LLM trả lời 1 từ, bỏ qua phần mô tả cấp độ có sẵn trong context.
- **Suggested fix:** đặt **ngưỡng điểm rerank** (ví dụ chỉ giữ context có score ≥ 30% score top-1) thay vì cố định top-3. Bổ sung prompt: "trả lời đầy đủ, nêu kèm yêu cầu/điều kiện liên quan có trong context".

### #3 — Numeric reasoning (phí tạm ứng quá hạn), avg = 0.796
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày, quá hạn 5 ngày. Phí 2%/tháng × 15.000.000 = 300.000 VNĐ/tháng (pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** "…bị tính phí **2%/tháng**… (Không có con số cụ thể bằng tiền được nêu trong context…)"
- **Worst metric:** context_recall
- **Error Tree:** Output sai (không ra con số) → Context đúng? **Có**: `tam_ung.md` đứng đầu, chứa "15 ngày" và "2%/tháng" → Query OK? Có → lỗi ở **Generation**.
- **Root cause:** system prompt "Trả lời **CHỈ** dựa trên context" khiến model hiểu là không được tính toán, nên từ chối nhân 2% × 15 triệu. Context recall bị chấm thấp vì các con số 300.000 / 50.000 trong ground truth là giá trị suy ra, không xuất hiện nguyên văn trong context. Context thứ 3 (nghỉ phép năm) cũng là nhiễu.
- **Suggested fix:** sửa prompt: "Được phép tính toán từ số liệu trong context; trình bày phép tính từng bước." Với câu hỏi dạng numeric, có thể thêm few-shot ví dụ tính phí/pro-rata.

### #4 — Version conflict (phép năm v2023 vs v2024), avg = 0.827
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** **15 ngày** theo v2024 hiện hành. 12 ngày là bản v2023 đã bị thay thế.
- **Got:** liệt kê song song "v2023: 12 ngày" và "v2024: 15 ngày", **không nói bản nào đang hiệu lực**.
- **Worst metric:** context_precision
- **Error Tree:** Output mơ hồ → Context đúng? Có cả 2 bản, nhưng **bản cũ v2023 xếp hạng 1**, trên bản v2024. Context thứ 3 (nghỉ ốm) không liên quan → Query OK? Có → lỗi ở **Rerank / thiếu metadata phiên bản** và **Prompt**.
- **Root cause:** hai tài liệu gần như giống hệt nhau về nội dung nên reranker không phân biệt được bản nào mới. Pipeline không dùng metadata `Ngày hiệu lực` / `Phiên bản` có sẵn trong header tài liệu.
- **Suggested fix:** trích `version`, `effective_date`, `status (superseded)` vào metadata ở bước M5. Khi có nhiều phiên bản của cùng một chính sách thì **ưu tiên bản có effective_date mới nhất** (re-sort sau rerank) và gắn nhãn "[ĐÃ THAY THẾ]" cho bản cũ trong context. Thêm vào prompt: "Nếu có nhiều phiên bản, trả lời theo bản hiện hành và ghi chú bản cũ đã bị thay thế."

### #5 — Version + yes/no (MFA), avg = 0.838
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có. Theo v2.0 hiện hành, MFA bắt buộc cho email, VPN và hệ thống nội bộ. Bản v1.0 cũ không yêu cầu MFA.
- **Got:** "Có, tất cả nhân viên bắt buộc kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ." Đúng nhưng thiếu ý so sánh với v1.0.
- **Worst metric:** context_recall
- **Error Tree:** Output đúng nhưng thiếu ý → Context đúng? **Thiếu** `mat_khau_v1.md`. Context 2–3 (WFH, phân loại dữ liệu) chỉ nhắc VPN/bảo mật một cách gián tiếp → Query OK? Có → lỗi ở **Retrieval**.
- **Root cause:** bản v1.0 không chứa từ "MFA", nên cả BM25 lẫn dense đều không kéo nó lên được. Ground truth lại cần thông tin "bản cũ không yêu cầu", nên recall bị trừ. Hai slot còn lại bị lấp bằng các tài liệu chỉ liên quan bề mặt (từ khóa "VPN"), làm precision cũng giảm.
- **Suggested fix:** **version-sibling expansion**: khi một tài liệu có `supersedes`/`superseded_by` trong metadata được retrieve, kéo luôn bản liên quan (v1 ↔ v2) vào context, đồng thời áp dụng ngưỡng rerank (như #2) để bỏ các tài liệu bề mặt.

---

## Case Study (cho presentation)

**Question chọn phân tích:** #1 "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?" Đây là câu tệ nhất (avg 0.375) và cho thấy rõ giới hạn của pipeline single-query.

**Error Tree walkthrough:**
1. **Output đúng?** Đúng một nửa. Phần phép năm đúng (18 ngày, đúng bản v2024 và đúng công thức thâm niên 9÷3). Phần lương trả lời "Không tìm thấy". Model không bịa số, nên faithfulness vẫn ổn: generator hoạt động đúng thiết kế.
2. **Context đúng?** Sai: 3/3 context đều thuộc nhóm "nghỉ phép", không có `bang_luong_2024.md`. Vậy lỗi nằm **trước** bước generation.
3. **Query rewrite OK?** Không có bước rewrite. Câu hỏi gộp 2 ý; trong embedding và BM25, vế "nghỉ phép năm + thâm niên" chiếm nhiều token hơn vế "lương Senior". Reranker cũng chấm theo cả câu, nên tài liệu nghỉ phép luôn thắng.
4. **Fix ở bước:** **Query transformation (trước M2).** Tách câu hỏi thành sub-query, retrieve và rerank riêng từng sub-query, rồi phân bổ slot context cho mỗi sub-query (ví dụ 2 + 1). Kỳ vọng context_recall của câu này tăng từ ≈0.5 lên ≈1.0, và LLM trả lời được cả "20–35 triệu".

**Nếu có thêm 1 giờ, sẽ optimize:**
- Query decomposition cho câu multi-hop (fix #1, ảnh hưởng nhóm câu multi-hop trong test set).
- Metadata `version` / `effective_date` từ M5 và re-sort ưu tiên bản mới (fix #4, #5 và nhóm câu "version").
- Ngưỡng điểm rerank động thay vì top-3 cố định, để tăng context_precision (metric thấp nhất hiện tại).
- Prompt cho phép tính toán từ số liệu trong context (fix #3, nhóm câu "numeric").
- OCR (`pytesseract`, `lang=vie`) cho 2 PDF scan đang bị bỏ qua.
