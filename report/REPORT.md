# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Võ Đại Phước]
**Nhóm:** [C401 - X5]
**Ngày:** [10/4/2026]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> 2 vector trong không gian đa chiều có góc càng bé (cosine tiến về 1.0)

**Ví dụ HIGH similarity:**
- Sentence A: 
- Sentence B:
- Tại sao tương đồng:

**Ví dụ LOW similarity:**
- Sentence A:
- Sentence B:
- Tại sao khác:

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine chỉ phụ thuộc vào góc giữa 2 vector, trong khi Euclidean distance phụ thuộc vào 
> độ lớn chiều dài của vector
> giúp tính toán trở nên ổn định hơn
> giống như trong bài toán Face recognition

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Lấy 500*N - 50*(N-1) + x = 10k (N chunk thì có N-1 đoạn overlap)
> => 450N + x = 10k-50, lấy (10k-50)/450 làm tròn xuống rồi cộng 1
> Đáp án: 23 

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> 500*N - 100*(N-1) + x = 10k => round_down((10k-100)/(500-100)) + 1 => 25 chunk
> overlap nhiều lên thì các chunk trở nên gần nhau hơn trong không gian nhiều chiều
> dẫn đến khi truy vấn sẽ chính xác hơn
---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Giáo dục, cụ thể VinUni policy

**Tại sao nhóm chọn domain này?**
> Tài liệu thực tế, gần gũi với chương trình đang học

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| | FixedSizeChunker (`fixed_size`) | | | |
| | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> regex: re.split(r'(?<=[.!?]) +'
> bắt dấu chấm có khoảng trắng phía sau

**`RecursiveChunker.chunk` / `_split`** — approach:
> chunk với seperator tại index 0 trước, chunk nào lớn hơn chunk_size thì dùng
> seperator tiếp theo để chunk và cứ tiếp tục quá trình

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents`: đầu tiên phải `_make_record`, mỗi document bao gồm original content, embedding 
> được tạo từ embedding_model, add thêm metadata, id khi được thêm vào collection thì
> lấy số lượng document hiện tại trong collection cộng dần lên
> `search`: tạo embedding cho câu query, sau đó dùng method query của collection để
> try vấn sau sort score theo thứ tự giảm dần, tìm với công thức cosine đã được config 
> trước khi khởi tạo collection, 
> xem thêm tại [EmbeddingStore.__init__](../src/store.py) method

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter`: dùng argument `where` trong method query của collection để thêm match với metadata
> `delete_document`: xóa 1 document phải xóa theo doc_id

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*

### Test Results

```terminal
export OPENAI_API_KEY=your_openai_api_key 
pytest ./tests/test_solution.py -v >> ./report/pytest_results.txt
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
