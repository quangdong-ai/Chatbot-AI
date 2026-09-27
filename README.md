# Chatbot AI

Chatbot AI tra cứu tài liệu nội bộ/pháp lý theo kiến trúc RAG, hỗ trợ hỏi đáp tiếng Việt dựa trên tài liệu đã nạp vào hệ thống. Dự án tập trung vào truy xuất chính xác, dẫn nguồn rõ ràng, xử lý PDF/OCR và đánh giá chất lượng câu trả lời bằng bộ benchmark nội bộ.

## Tính năng chính

- Hỏi đáp tài liệu theo kiến trúc RAG.
- Xử lý tài liệu PDF, TXT, DOCX, XLSX, PPTX.
- Trích xuất nội dung PDF bằng PyMuPDF, OCR bằng Tesseract và hướng mở rộng OCR-VL/PaddleOCR cho tài liệu scan.
- Chunking theo cấu trúc văn bản, bảng, điều/khoản/phụ lục và metadata.
- Hybrid Retrieval kết hợp ChromaDB vector search và BM25.
- Embedding bằng BGE-M3.
- Rerank bằng BGE reranker, ưu tiên `bge-reranker-v2-m3` khi bật cấu hình.
- Tích hợp LLM qua Ollama, mặc định dùng `qwen2.5:latest`.
- Citation verification để kiểm tra câu trả lời có được nguồn hỗ trợ hay không.
- Giao diện web người dùng và trang quản trị.
- API FastAPI cho chat, streaming, upload tài liệu, quản lý bot, export và đánh giá.
- Bộ test và benchmark đánh giá retrieval/generation.

## Kiến trúc thư mục

```text
.
|-- app.py                         # Entry point chạy FastAPI app
|-- ingest.py                      # Pipeline nạp/chunk/index tài liệu
|-- run_benchmark.py               # Chạy benchmark hỏi đáp
|-- botai/
|   |-- answering/                 # Rule extractor, format câu trả lời, citation verifier
|   |-- api/                       # Pydantic schemas
|   |-- core/                      # Config, state runtime
|   |-- models/                    # Khởi tạo LLM, embedding, reranker, retriever
|   |-- retrieval/                 # Hybrid search, fusion, rerank, neighbor docs
|   |-- routes/                    # FastAPI routes
|   |-- security/                  # Auth, session, rate limit
|   `-- services/                  # Database, source, export, index, bot registry
|-- static/                        # Giao diện web/admin/widget
|-- tests/                         # Unit tests
|-- evaluation/                    # Script và kết quả đánh giá
|-- requirements.txt
`-- requirements-ocr-vl.txt
```

## Thành phần AI và thuật toán

| Thành phần | Công nghệ |
|---|---|
| Backend | FastAPI, Uvicorn |
| LLM | Ollama `qwen2.5:latest` |
| Embedding | BGE-M3 local model |
| Vector database | ChromaDB |
| Keyword retrieval | BM25 |
| Hybrid search | RRF / weighted fusion |
| Reranker | BGE reranker large / BGE reranker v2 m3 |
| PDF parser | PyMuPDF |
| OCR fallback | Tesseract |
| OCR-VL mở rộng | PaddleOCR / PaddleOCR-VL stack |
| Database app | SQLite |
| Test | Pytest |

## Cài đặt

Tạo và kích hoạt môi trường Python trước, sau đó cài dependency:

```powershell
pip install -r requirements.txt
```

Nếu dùng OCR-VL/PaddleOCR:

```powershell
pip install -r requirements-ocr-vl.txt
```

Cài Ollama và tải model LLM:

```powershell
ollama pull qwen2.5
```

## Model và dữ liệu không đưa lên GitHub

Các thư mục/file nặng hoặc dữ liệu runtime đã được ignore:

```text
models--*/
chroma_db/
storage/
chatbot_app.db
bm25_retriever.pkl
hf_cache/
paddlex_cache/
```

Vì vậy sau khi clone repo trên máy mới, cần tải lại model local và chạy lại pipeline ingest để tạo index.

## Cấu hình môi trường

Một số biến môi trường quan trọng:

```powershell
$env:BOTAI_LLM_MODEL="qwen2.5:latest"
$env:BOTAI_EMBEDDING_MODEL_PATH="./models--BAAI--bge-m3"
$env:BOTAI_RERANKER_ENABLED="1"
$env:BOTAI_RERANKER_BACKEND="bge-v2-m3"
$env:ADMIN_USERNAME="admin"
$env:ADMIN_PASSWORD="your-password"
$env:ADMIN_TOKEN="your-admin-token"
```

Nếu không đặt `ADMIN_TOKEN`, hệ thống vẫn hỗ trợ đăng nhập bằng user admin khi có `ADMIN_USERNAME` và `ADMIN_PASSWORD`.

## Nạp tài liệu và tạo index

Đặt tài liệu mặc định là `tailieu.pdf` ở thư mục gốc, sau đó chạy:

```powershell
python ingest.py
```

Pipeline sẽ tạo:

```text
chroma_db/
bm25_retriever.pkl
chatbot_app.db
```

## Chạy hệ thống

```powershell
python app.py
```

Mặc định app chạy tại:

```text
http://127.0.0.1:8001
```

Một số trang/API chính:

```text
/                         Giao diện chatbot
/admin.html               Trang quản trị
/api/v1/health            Kiểm tra trạng thái hệ thống
/api/v1/chat              Chat API
/api/v1/chat/stream       Streaming chat API
```

## Chạy test

```powershell
pytest -q
```

## Benchmark

Chạy benchmark:

```powershell
python run_benchmark.py
```

Các script đánh giá chi tiết nằm trong thư mục `evaluation/`, gồm retrieval eval, parsing/OCR benchmark, A/B matrix và kiểm tra golden set.

## Ghi chú bảo mật

- Không commit database runtime, vector index, model local hoặc cache.
- Không hard-code API key/token vào source code.
- Nên đặt repo ở chế độ private nếu tài liệu pháp lý/nội bộ có dữ liệu nhạy cảm.
- Với môi trường production, cần đặt `ADMIN_PASSWORD`/`ADMIN_TOKEN` mạnh và chạy sau reverse proxy có HTTPS.

## Trạng thái dự án

Dự án đã được tách module từ `app.py` lớn thành cấu trúc nhiều package nhỏ để dễ bảo trì:

- `routes`: định tuyến API và trang web.
- `services`: nghiệp vụ hệ thống.
- `retrieval`: truy xuất và rerank.
- `answering`: tạo câu trả lời, rule extractor và kiểm chứng nguồn.
- `models`: khởi tạo model runtime.
- `security`: xác thực và giới hạn tần suất.
