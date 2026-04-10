from __future__ import annotations

import os
import sys
from pathlib import Path
import uuid
from dotenv import load_dotenv

load_dotenv()

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore
from src.chunking import SentenceChunker

MAX_SENTENCES_PER_CHUNK = 5


BENCHMARK_QUERIES = [
    {
        "id": "1",
        "query": "Làm sao để có thể học cùng lúc hai chương trình",
        "gold_answer": """2. Điều kiện để học cùng lúc hai chương trình
a) Ngành đào tạo chính ở chương trình thứ hai phải khác với ngành đào tạo chính
ở chương trình thứ nhất.
b) Sinh viên được đăng ký học chương trình thứ hai sớm nhất khi đã được xếp
trình độ năm thứ hai của chương trình thứ nhất.
c) Tại thời điểm đăng ký, sinh viên phải đáp ứng (i) học lực tính theo điểm trung
bình tích lũy xếp loại khá trở lên (CGPA=2.5 trở lên); (ii) tiêu chí xét đầu vào của
chương trình thứ hai trong năm tuyển sinh; và (iii) không đang trong thời gian bị kỷ luật
tương đương ở mức cảnh cáo trở lên.
d) Trong quá trình sinh viên học cùng lúc hai chương trình, nếu điểm trung bình
tích luỹ của chương trình thứ nhất đạt dưới điểm trung bình hoặc thuộc diện cảnh báo
kết quả học tập thì phải dừng học chương trình thứ hai ở học kỳ tiếp theo; sinh viên sẽ
bị loại khỏi danh sách đã đăng ký học chương trình thứ hai.
e) Sinh viên có nguyện vọng học cùng lúc hai chương trình cần đăng ký chương
trình thứ hai tối thiểu 2 năm trước thời hạn dự kiến tốt nghiệp của chương trình thứ hai."""
    },{
        "id": "2",
        "query": "Có khi nào hệ thống retrieval bị sai không?",
        "gold_answer": """Trong thực tế, retrieval không phải lúc nào cũng đúng. Một số lỗi thường gặp là tài liệu cũ vẫn xếp hạng cao, từ khóa trong câu hỏi không khớp với cách diễn đạt trong tài liệu, hoặc embedding model chưa xử lý tốt nội dung song ngữ. Vì vậy, đội ngũ phát triển nên kiểm thử bằng các truy vấn thực tế, xem trực tiếp các chunk được trả về, và ghi nhận failure cases để cải thiện dữ liệu cũng như chiến lược truy xuất."""
    },{
        "id": "3",
        "query": "khi nào sinh viên được công nhận tốt nghiệp?",
        "gold_answer": """Điều 28. Công nhận tốt nghiệp
1. Sinh viên phải đăng ký tốt nghiệp trong học kỳ tốt nghiệp dự kiến theo các thủ
tục và hướng dẫn của Nhà trường.
2. Sinh viên đã đăng ký tốt nghiệp nhưng không hoàn thành tất cả các yêu cầu
đào tạo vào cuối học kỳ/kỳ tốt nghiệp dự kiến phải đăng ký lại để tốt nghiệp.
3. Sinh viên sau khi hoàn thành chương trình đào tạo được xét và công nhận tốt
nghiệp phải đáp ứng đủ các điều kiện sau:
a) Tích lũy đủ số học phần, khối lượng của chương trình đào tạo trong thời gian
đào tạo quy định của mỗi chương trình đào tạo;
b) Hoàn thành các yêu cầu về Giáo dục đại cương và năng lực tiếng Anh;
c) Hoàn thành các yêu cầu đối với chuyên ngành do các Viện đào tạo quy định;
d) Hoàn thành các học phần bắt buộc đang bị điểm “I - Chưa hoàn thành” trong
bảng điểm;
e) Điểm trung bình tích lũy của toàn khóa học đạt tối thiểu từ 2,00/4,00 trở lên;
f) Cho đến thời điểm xét tốt nghiệp không bị truy cứu trách nhiệm hình sự;
g) Hoàn thành nghĩa vụ khác của sinh viên theo quy định của Nhà trường;
4. Căn cứ đề nghị của Hội đồng xét tốt nghiệp, Hiệu trưởng ký quyết định công
nhận tốt nghiệp cho các sinh viên đủ điều kiện theo quy định.
Hội đồng xét tốt nghiệp do Hiệu trưởng hoặc người được Hiệu trưởng uỷ quyền
làm Chủ tịch, Trưởng phòng Quản lý đào tạo làm Thư ký và các thành viên khác bao
gồm lãnh đạo các đơn vị chuyên môn và Trưởng phòng Phòng Công tác sinh viên."""
    },{
        "id": "4",
        "query": "xếp hạng học lực tại VinUni",
        "gold_answer": """2. Sau mỗi học kỳ, căn cứ vào điểm trung bình tích lũy, sinh viên được xếp hạng
về học lực như sau:
Xếp hạng học lực Xuất sắc 3,60 – 4,00
Sinh viên năm thứ 6 Điểm trung bình tích lũy
Giỏi Khá 3,20 – 3,59
2,50 – 3,19
Trung bình 2,00 – 2,49
Yếu Dưới 2,00"""
    },{
        "id": "5",
        "query": "Làm thế nào để  bảo lưu kết quả?",
        "gold_answer": """Điều 15. Nghỉ học tạm thời hoặc nghỉ ốm
1. Sinh viên muốn xin nghỉ học tạm thời hoặc bảo lưu kết quả đã học có thể gửi
yêu cầu cho Viện trưởng. Sinh viên có thể được nghỉ từ một đến hai học kỳ tùy từng
trường hợp. Sau khi kết thúc thời gian bảo lưu, sinh viên phải liên hệ nhà trường để xin
gia hạn trong trường hợp muốn kéo dài thời gian bảo lưu.
2. Sinh viên được xin nghỉ học tạm thời và bảo lưu kết quả học tập trong các
trường hợp sau:
a) Được điều động vào các lực lượng vũ trang (cần có thư xác thực);
16
b) Được cấp có Thẩm quyền cử đại diện cho Quốc gia tham gia các cuộc thi, giải
đấu quốc tế;
c) Bị ốm hoặc cấp cứu y tế phải điều trị thời gian dài, nhưng phải có giấy xác
nhận của cơ quan y tế. Các giấy tờ trên đều phải được dịch công chứng sang tiếng Anh.
d) Vì lý do cá nhân hoặc gia cảnh khó khăn. Trường hợp này, sinh viên phải học
ít nhất một học kỳ ở trường, không rơi vào các trường hợp bị buộc thôi học quy định tại
Điều 15 của Quy chế này . Thời gian nghỉ học tạm thời vì nhu cầu cá nhân phải được
tính vào thời gian học chính thức tại trường.
3. Đơn xin nghỉ học tạm thời/thôi học cho kỳ học sắp tới sẽ có hiệu lực từ ngày
cuối cùng của học kỳ đang học. Đơn xin nghỉ học tạm thời học kỳ đang học sẽ có hiệu
lực kể từ ngày nộp đơn.
4. Để bảo lưu kết quả học, sinh viên phải chuẩn bị đơn cũng như các giấy tờ minh
chứng cần thiết và được sự cho phép của cố vấn học tập. Sau khi có kết quả cuối cùng,
Phòng Quản lý Đào tạo sẽ xem xét các yêu cầu với Viện và thông báo cho sinh viên."""
    }

]




SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
    "data/team_selection_custom_data.txt"
]

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""

    # chunk the content right after loading it
    chunker = SentenceChunker(max_sentences_per_chunk=MAX_SENTENCES_PER_CHUNK)

    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        for ith, chunk_str in enumerate(chunker.chunk(content)):
            # use uuid for global indexing for all chunks for all files
            # anh use chunk order in metadata
            documents.append(
                Document(
                    id=str(uuid.uuid4()),
                    content=chunk_str,
                    metadata={
                        "chunk_id": ith,
                        "source": str(path), 
                        "extension": path.suffix.lower()
                    },
                )
            )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:900].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def main():
    files = SAMPLE_FILES
    
    print("=== SETTING UP BENCHAMRK ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Benchmark ===")
    
    
    print(f"Query: {query}")
    search_results = store.search(query, top_k=5)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
