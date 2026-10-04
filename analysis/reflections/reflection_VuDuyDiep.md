# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Vũ Duy Điệp (2A202602703)
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 05/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 trên toàn corpus tạo **208 chunk (trung bình 99 ký tự, min 6)**, trong khi basic tạo **51 chunk (trung bình 410)**. Lý do: `all-MiniLM-L6-v2` là model tiếng Anh, nên cosine giữa các câu tiếng Việt liền nhau hay rơi dưới 0.85 và văn bản bị cắt vụn. Muốn dùng semantic cho tiếng Việt phải đổi sang embedding đa ngữ (bge-m3) hoặc hạ threshold. |
| Hierarchical (parent-child) | M1 + pipeline | `chunk_hierarchical()`, `run_query()` | 26 tài liệu thành 100 child (≤ 256 ký tự). Tìm trên child, sau đó **trả parent** (dedupe theo `parent_id`, lấy 3 parent khác nhau). Kết hợp với hybrid search, rerank và enrichment, Faithfulness tăng 0.825 → **0.973** và Context Recall 0.842 → **0.925**. Chưa chạy ablation riêng từng thành phần, nên chưa tách được mức đóng góp của riêng parent-return. |
| Structure-aware | M1 | `chunk_structure_aware()` | 106 chunk (trung bình 196), giữ header `##` trong `metadata["section"]` và không cắt trong code fence. Hợp với corpus chính sách vốn chia section rõ ràng. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | `underthesea` tách "nghỉ_phép" rồi `replace("_", " ")` để token khớp với query. RRF (k=60) cộng `1/(k+rank)` nên không cần chuẩn hóa thang điểm BM25 và cosine. Hybrid search mất **404 ms/query** (gồm encode bge-m3 trên CPU). |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` rerank khoảng 20 candidate mất **5069 ms/query trên CPU**, chiếm khoảng 75% latency khi chưa tính throttle. Context Precision 0.80 → **0.85**. Hạn chế: không phân biệt được 2 bản chính sách gần giống nhau (v2023 bị xếp trên v2024, failure #4). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production: F **0.973** / AR **0.856** / CP **0.850** / CR **0.925** (baseline 0.825 / 0.716 / 0.800 / 0.842). **Context Precision thấp nhất** vì pipeline luôn trả đủ 3 parent, kể cả khi chỉ cần 1. `failure_analysis()` map worst metric sang diagnosis và fix theo Diagnostic Tree. |
| Contextual embeddings | M5 | `_enrich_single_call()` | Combined mode: 1 call/chunk trả summary + 3 HyQA + context + metadata dạng JSON. 100 chunk mất 807 s (gồm throttle), 0 fallback. Câu context (ví dụ "Tài liệu chi_phi_expense.md quy định về…") được prepend trước chunk, đưa tên và chủ đề tài liệu vào cả BM25 lẫn vector. Nhờ vậy child 256 ký tự không còn "mồ côi" ngữ cảnh. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### Lỗi 1 — OpenAI key không hợp lệ
- **Exact error:**
  `openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: AQ.Ab8RN*****...ioHA ...'}}`
- **Debug:** gọi thử 1 request `chat.completions` tối thiểu cho từng provider (OpenAI và Gemini) với `max_retries=0` để tách lỗi key khỏi lỗi code. Key trong `OPENAI_API_KEY` thực chất là key Google (tiền tố `AQ.`), nên OpenAI từ chối.
- **Giải quyết:** dùng Gemini qua endpoint OpenAI-compatible (`GEMINI_BASE_URL`). M4 bọc Gemini thành `BaseRagasLLM` để RAGAS 0.1.x dùng được, M5 và bước generation dùng chung `openai.OpenAI(base_url=...)`.

### Lỗi 2 — Hết quota free tier giữa chừng
- **Exact error:**
  `Error code: 429 - RESOURCE_EXHAUSTED ... Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 500, model: gemini-3.5-flash-lite. Please retry in 6h48m16s.`
- **Debug:** đọc `quotaId` trong body lỗi: `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. Quota được tính **theo project và theo từng model**. Tiếp theo, liệt kê model bằng `client.models.list()` rồi gọi thử từng model với `max_retries=0`:
  - `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.0-flash` trả 404 (`no longer available to new users`).
  - `gemini-flash-lite-latest` chỉ là alias của model đã hết quota.
  - `gemini-3.7-flash` / `3.8-flash` trả 503 (quá tải).
  - `gemini-3.5-flash` / `3.6-flash` chỉ có **20 request/ngày** (`quotaValue: '20'`), không đủ cho khoảng 140 call.
- **Bẫy thứ hai — retry storm:** lần chạy đầu, enrichment chỉ đi được 10/100 chunk sau 13 phút. Bật log `httpx` thì thấy chuỗi `429 → Retrying in 0.4s → 0.9s → 1.9s → 4s → 6.6s`. Client OpenAI tự retry rất dày, và mỗi lần retry cũng bị tính vào quota, nên càng retry càng bị chặn.
- **Giải quyết:**
  - Đổi sang key của project khác để có quota mới.
  - Thêm `GEMINI_ENRICH_MODEL` và `GEMINI_EVAL_MODEL` vào `config.py` để tách model theo từng giai đoạn. Generation và enrichment dùng `gemini-3.5-flash-lite`; RAGAS judge dùng `gemini-3.1-flash-lite`, cố định cho cả baseline lẫn production để so sánh công bằng.
  - Chạy baseline và production lệch pha nhau (production bắt đầu khi baseline đã sinh xong câu trả lời) để hai process không cùng gọi một model một lúc.
  - Đặt `answer_relevancy.strictness = 1`: Gemini chỉ trả 1 completion mỗi request, nên `n=3` sẽ tốn 3 request/câu. Cách này giảm từ khoảng 180 xuống khoảng 140 judge call mỗi lần chạy.

### Lỗi 3 — Thinking model trả `content=None`
- **Hiện tượng:** khi dò model, `gemini-3.5-flash` với `max_tokens=10` trả `message.content = None`, `completion_tokens=1`, nhưng `total_tokens=149`.
- **Nguyên nhân:** model "thinking" tiêu hết token budget cho phần suy nghĩ nội bộ trước khi kịp sinh câu trả lời.
- **Giải quyết:** giữ `max_tokens=4096` cho mọi call Gemini. Code đã kiểm tra rỗng (`content(response)` raise `ValueError("Gemini returned an empty response")`) để RAGAS retry thay vì chấm điểm trên chuỗi rỗng.

### Lỗi 4 — Windows console và PDF scan
- `UnicodeEncodeError: 'charmap' codec can't encode character 'ủ'` khi in tiếng Việt ra console Windows (cp1252). Cách sửa: `sys.stdout.reconfigure(encoding="utf-8")` ở đầu mỗi module, hoặc chạy với `PYTHONIOENCODING=utf-8`.
- `⚠️ Bỏ qua BCTC.pdf: PDF scan ảnh, không có text layer (cần OCR)`, và tương tự với Nghị định 13/2023. pypdf chỉ đọc được text layer nên 2/3 file PDF không vào index. Đây là một giới hạn còn tồn tại (xem Phần 3).

### Kiến thức còn thiếu & cách bổ sung
- **RAGAS 0.1.x gọi LLM như thế nào:** đọc source `ragas/metrics/_answer_relevance.py` để biết `strictness` điều khiển `n` completions, và `_context_precision.py` để biết mỗi context tốn 1 call. Nhờ đó ước lượng được số request trước khi chạy, tránh cháy quota lần nữa.
- **Rate limit của Gemini free tier:** đọc trang `ai.google.dev/gemini-api/docs/rate-limits`. Có 2 tầng giới hạn: RPM (request/phút, dẫn tới `GEMINI_REQUEST_INTERVAL=4.2s`) và RPD (request/ngày, tính theo model).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý hỏi đáp chính sách nội bộ (HR / IT / Tài chính) cho doanh nghiệp

#### 1. Hiện trạng
- **Pipeline hiện tại:** naive RAG gồm cắt đoạn theo `\n\n` (khoảng 500 ký tự), dense search 1 vector store, lấy top-3 rồi đưa thẳng vào LLM. Không rerank, không enrichment, không có bộ test đánh giá.
- **Vấn đề đang gặp:**
  - Trả lời theo **chính sách cũ** khi có nhiều phiên bản (v2023 và v2024).
  - Bỏ sót mã/số hiệu chính xác (ví dụ "PVI", "WireGuard") vì dense embedding không bắt từ khóa hiếm.
  - Tài liệu scan (báo cáo tài chính, nghị định) không được index.
  - Không có số liệu để biết thay đổi nào thực sự cải thiện chất lượng.

#### 2. Kế hoạch cải tiến
1. **Chunking — Hierarchical (child 256 / parent 2048) kết hợp header-aware.** Tìm trên child để tăng precision, trả parent để LLM có đủ ngữ cảnh điều kiện/ngoại lệ. Lab cho thấy semantic chunking với `all-MiniLM-L6-v2` (model tiếng Anh) cắt vụn văn bản tiếng Việt (208 chunk, trung bình 99 ký tự), nên **không** dùng nếu chưa đổi sang embedding đa ngữ.
2. **Search — Hybrid BM25 (`underthesea` word-segment) + dense `bge-m3`, gộp bằng RRF (k=60).** BM25 bắt số hiệu và tên riêng, dense bắt paraphrase. RRF không cần chuẩn hóa điểm giữa hai hệ.
3. **Reranking — có, dùng `BAAI/bge-reranker-v2-m3`** (đa ngữ, chạy được trên CPU) để rerank top-20 xuống top-3. Nếu latency vượt SLA thì chuyển sang FlashRank cho truy vấn đơn giản.
4. **Evaluation — RAGAS 4 metrics** trên bộ test 20 câu (6 loại: lookup, version, negation, multi-hop, numeric, ambiguous), mở rộng lên 50 câu. Chạy mỗi lần đổi pipeline, cố định 1 judge model để các lần chạy so sánh được với nhau. Thêm 1 metric tự viết là **"version correctness"**: câu trả lời có trích phiên bản hiện hành hay không.
5. **Enrichment — combined single-call** (summary + HyQA + context + metadata trong 1 request/chunk). Đưa `effective_date` và `version` vào metadata để lọc bỏ bản cũ ở bước search, thay vì trông chờ LLM tự chọn đúng.
6. **OCR cho PDF scan:** dùng `pytesseract` hoặc Document AI với `lang=vie`, rồi đưa qua pipeline chunking như file `.md`.

#### 3. Timeline triển khai
- **Tuần 1:** dựng bộ test 50 câu và chạy RAGAS baseline (đóng băng số liệu làm mốc). Chuyển chunking sang hierarchical.
- **Tuần 2:** hybrid search BM25 + bge-m3 + RRF, cùng reranker bge-reranker-v2-m3. Đo latency từng bước (search / rerank / generate).
- **Tuần 3:** enrichment combined mode + metadata `version` / `effective_date` + filter. OCR cho PDF scan.
- **Tuần 4:** chạy lại RAGAS, phân tích bottom-5 theo Error Tree, sửa prompt/retrieval, rồi đóng gói và triển khai nội bộ.
