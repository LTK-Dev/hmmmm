import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re

# Import các thư viện cần thiết cho FAISS và LangChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- 1. TỐI ƯU HIỆU NĂNG VỚI CACHING ---
@st.cache_resource
def get_embedder():
    """Tải và cache mô hình embedding."""
    print("INFO: Đang tải mô hình embedding...")
    return SentenceTransformerEmbeddings(
        model_name=EMBEDDER_MODEL,
        model_kwargs={'device': 'cpu'}
    )

# --- Data Classes và Enums ---
class SourceType(Enum):
    SCRIPT = "script"
    PRODUCT = "product"
    HYBRID = "hybrid"

@dataclass
class RetrievedInfo:
    content: str
    source_type: SourceType
    score: float
    metadata: Dict[str, Any] = None

@dataclass
class MasterDecision:
    primary_source: SourceType
    confidence: float
    reasoning: str
    selected_info: List[RetrievedInfo]
    response_strategy: str

@dataclass
class TaskRequest:
    query: str
    context: List[str] = None
    metadata: Dict[str, Any] = None

# --- Cấu hình ---
PRODUCT_CSV_FILE = 'EKS.csv'
SCRIPT_CSV_FILE = 'EKS_ques.csv'
PRODUCT_FAISS_PATH = "faiss_index_product"
SCRIPT_FAISS_PATH = "faiss_index_script"
EMBEDDER_MODEL = 'intfloat/multilingual-e5-base'
GENERATIVE_MODEL = 'gemini-2.0-flash'

# --- Master Agent (Thay thế Router) ---
class MasterAgent:
    def __init__(self, model: genai.GenerativeModel, product_store: FAISS, script_store: FAISS):
        self.model = model
        self.product_store = product_store
        self.script_store = script_store
        
        self.evaluation_prompt = """
Bạn là Master Agent của hệ thống EKS - có nhiệm vụ đánh giá và quyết định nguồn thông tin tốt nhất để trả lời khách hàng.

**Câu hỏi từ khách hàng:**
{query}

**THÔNG TIN TỪ KỊCH BẢN Q&A:**
{script_info}

**THÔNG TIN TỪ DATABASE SẢN PHẨM:**
{product_info}

**NHIỆM VỤ CỦA BẠN:**
1. Đánh giá chất lượng và độ phù hợp của mỗi nguồn thông tin
2. Quyết định nguồn nào nên được ưu tiên
3. Đưa ra chiến lược trả lời phù hợp

**QUY TẮC ƯU TIÊN:**
- **KỊCH BẢN Q&A**: Ưu tiên cao nhất nếu có câu trả lời trực tiếp và chính xác
- **SẢN PHẨM**: Sử dụng khi cần thông tin chi tiết, kỹ thuật về sản phẩm
- **KẾT HỢP**: Dùng cả hai nguồn khi cần thông tin toàn diện

Trả lời theo format JSON:
{{
    "primary_source": "script/product/hybrid",
    "confidence": 0.9,
    "reasoning": "Lý do chi tiết về quyết định",
    "response_strategy": "Chiến lược trả lời cụ thể"
}}
"""

        self.response_prompt = """
Bạn là EKS Master Agent - chuyên gia tư vấn hàng đầu về sản phẩm mỹ phẩm EKS.

**Câu hỏi từ khách hàng:**
{query}

**THÔNG TIN ĐÃ ĐƯỢC CHỌN:**
{selected_info}

**CHIẾN LƯỢC TRẢ LỜI:**
{strategy}

**HƯỚNG DẪN TRẢ LỜI:**
1. **Xưng hô**: "Tôi là EKS Master Agent"
2. **Ưu tiên kịch bản**: Nếu có thông tin từ kịch bản Q&A, sử dụng y nguyên
3. **Bổ sung sản phẩm**: Thêm chi tiết kỹ thuật từ database sản phẩm nếu cần
4. **Phong cách**: Chuyên nghiệp, thân thiện, dễ hiểu
5. **Cấu trúc**: Rõ ràng, có logic, dễ theo dõi

**LƯU Ý QUAN TRỌNG:**
- Luôn dựa trên thông tin có sẵn
- Thừa nhận nếu không có đủ thông tin
- Đưa ra lời khuyên thực tế và hữu ích
- Nếu trong kịch bản Q&A thật sự có câu hỏi có hàm ý và câu trả lời liên quan đến query của khách hàng, hãy sử dụng nguyên câu trả lời đó, đừng ưu tién tư vấn quá nhiều.
- Chỉ khi mà kịch bản thật sự không có câu trả lời nào liên quan đến query của khách hàng, hãy sử dụng thông tin từ database sản phẩm để tư vấn.
- Khách hàng ưu tiên câu trả lời ngắn gọn và súc tích, đi vào trọng tâm vấn đề rồi mới diễn giải chi tiết nếu cần thiết.
"""

    def retrieve_all_sources(self, query: str, k_script: int = 3, k_product: int = 5) -> Tuple[List[RetrievedInfo], List[RetrievedInfo]]:
        """Truy vấn thông tin từ cả hai vector database."""
        script_infos = []
        product_infos = []
        
        try:
            # Truy vấn kịch bản Q&A
            script_results = self.script_store.similarity_search_with_score(query, k=k_script)
            for doc, score in script_results:
                script_infos.append(RetrievedInfo(
                    content=doc.page_content,
                    source_type=SourceType.SCRIPT,
                    score=score,
                    metadata={'source': 'qa_script'}
                ))
            
            # Truy vấn database sản phẩm
            product_results = self.product_store.similarity_search_with_score(query, k=k_product)
            for doc, score in product_results:
                product_infos.append(RetrievedInfo(
                    content=doc.page_content,
                    source_type=SourceType.PRODUCT,
                    score=score,
                    metadata={'source': 'product_db'}
                ))
                
        except Exception as e:
            print(f"ERROR: Lỗi khi truy vấn vector stores: {e}")
            
        return script_infos, product_infos

    def evaluate_and_decide(self, query: str, script_infos: List[RetrievedInfo], product_infos: List[RetrievedInfo]) -> MasterDecision:
        """Đánh giá và quyết định nguồn thông tin tốt nhất."""
        
        # Chuẩn bị thông tin cho prompt
        script_content = "\n\n".join([f"Score: {info.score:.3f}\n{info.content}" for info in script_infos[:3]])
        product_content = "\n\n".join([f"Score: {info.score:.3f}\n{info.content}" for info in product_infos[:3]])
        
        if not script_content:
            script_content = "Không tìm thấy thông tin liên quan trong kịch bản Q&A"
        if not product_content:
            product_content = "Không tìm thấy thông tin liên quan trong database sản phẩm"
        
        try:
            prompt = self.evaluation_prompt.format(
                query=query,
                script_info=script_content,
                product_info=product_content
            )
            
            response = self.model.generate_content(prompt)
            decision_text = self._extract_json_from_response(response.text)
            
            if decision_text:
                decision_data = json.loads(decision_text)
                
                # Chọn thông tin dựa trên quyết định
                selected_info = self._select_info_based_on_decision(
                    decision_data.get('primary_source', 'script'),
                    script_infos,
                    product_infos
                )
                
                return MasterDecision(
                    primary_source=SourceType(decision_data.get('primary_source', 'script')),
                    confidence=decision_data.get('confidence', 0.7),
                    reasoning=decision_data.get('reasoning', 'Quyết định tự động'),
                    selected_info=selected_info,
                    response_strategy=decision_data.get('response_strategy', 'Trả lời dựa trên thông tin có sẵn')
                )
                
        except Exception as e:
            print(f"WARNING: Lỗi khi đánh giá, sử dụng fallback logic: {e}")
            
        # Fallback logic
        return self._fallback_decision(query, script_infos, product_infos)

    def _extract_json_from_response(self, text: str) -> Optional[str]:
        """Trích xuất JSON từ response."""
        try:
            # Làm sạch text
            cleaned_text = text.strip()
            if "```json" in cleaned_text:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', cleaned_text, re.DOTALL)
                if json_match:
                    return json_match.group(1)
            
            # Tìm JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text)
            if json_match:
                return json_match.group(0)
                
            return None
        except Exception as e:
            print(f"WARNING: Lỗi extract JSON: {e}")
            return None

    def _select_info_based_on_decision(self, primary_source: str, script_infos: List[RetrievedInfo], product_infos: List[RetrievedInfo]) -> List[RetrievedInfo]:
        """Chọn thông tin dựa trên quyết định của Master Agent."""
        if primary_source == 'script':
            return script_infos[:2] + product_infos[:1]  # Ưu tiên script
        elif primary_source == 'product':
            return product_infos[:3] + script_infos[:1]  # Ưu tiên product
        else:  # hybrid
            return script_infos[:2] + product_infos[:2]  # Cân bằng

    def _fallback_decision(self, query: str, script_infos: List[RetrievedInfo], product_infos: List[RetrievedInfo]) -> MasterDecision:
        """Logic fallback khi không thể đánh giá được."""
        
        # Tính điểm trung bình cho mỗi nguồn
        script_avg_score = sum(info.score for info in script_infos[:2]) / max(len(script_infos[:2]), 1) if script_infos else 1.0
        product_avg_score = sum(info.score for info in product_infos[:2]) / max(len(product_infos[:2]), 1) if product_infos else 1.0
        
        # FAISS sử dụng L2 distance, score thấp = tốt hơn
        if script_infos and script_avg_score < 0.4:  # Script có kết quả tốt
            return MasterDecision(
                primary_source=SourceType.SCRIPT,
                confidence=0.7,
                reasoning="Fallback: Kịch bản Q&A có thông tin phù hợp",
                selected_info=script_infos[:2] + product_infos[:1],
                response_strategy="Ưu tiên câu trả lời từ kịch bản, bổ sung thông tin sản phẩm"
            )
        elif product_infos and product_avg_score < 0.5:  # Product có kết quả khá tốt
            return MasterDecision(
                primary_source=SourceType.PRODUCT,
                confidence=0.6,
                reasoning="Fallback: Database sản phẩm có thông tin liên quan",
                selected_info=product_infos[:3],
                response_strategy="Tập trung vào thông tin chi tiết sản phẩm"
            )
        else:  # Kết hợp cả hai
            return MasterDecision(
                primary_source=SourceType.HYBRID,
                confidence=0.5,
                reasoning="Fallback: Kết hợp thông tin từ cả hai nguồn",
                selected_info=script_infos[:1] + product_infos[:2],
                response_strategy="Kết hợp thông tin từ kịch bản và sản phẩm"
            )

    def generate_response_stream(self, query: str, decision: MasterDecision):
        """Generate streaming response dựa trên quyết định của Master Agent."""
        
        # Chuẩn bị thông tin đã chọn
        selected_content = "\n\n".join([
            f"[{info.source_type.value.upper()}] {info.content}" 
            for info in decision.selected_info
        ])
        
        prompt = self.response_prompt.format(
            query=query,
            selected_info=selected_content,
            strategy=decision.response_strategy
        )
        
        try:
            response_stream = self.model.generate_content(prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            st.error(f"Lỗi khi generate response: {e}")
            yield f"Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn."

# --- Agent Manager được cập nhật ---
class AgentManager:
    def __init__(self):
        self.model = genai.GenerativeModel(GENERATIVE_MODEL)
        embedder = get_embedder()
        self.product_store = load_or_create_product_faiss(embedder)
        self.script_store = load_or_create_script_faiss(embedder)
        
        # Khởi tạo Master Agent thay vì Router
        self.master_agent = MasterAgent(self.model, self.product_store, self.script_store)

    def process_query(self, query: str) -> Dict[str, Any]:
        """Xử lý query với Master Agent."""
        
        # 1. Truy vấn tất cả nguồn thông tin
        script_infos, product_infos = self.master_agent.retrieve_all_sources(query)
        
        # 2. Master Agent đánh giá và quyết định
        decision = self.master_agent.evaluate_and_decide(query, script_infos, product_infos)
        
        return {
            'master_decision': decision,
            'script_infos': script_infos,
            'product_infos': product_infos,
            'query': query
        }

    def get_response_stream(self, query: str, decision: MasterDecision):
        """Lấy streaming response từ Master Agent."""
        return self.master_agent.generate_response_stream(query, decision)

# --- FAISS Loading Functions (giữ nguyên) ---
@st.cache_resource
def load_or_create_product_faiss(_embedder):
    """Tải hoặc tạo FAISS index cho dữ liệu sản phẩm."""
    if os.path.exists(PRODUCT_FAISS_PATH):
        print(f"INFO: Đang tải chỉ mục sản phẩm từ '{PRODUCT_FAISS_PATH}'...")
        return FAISS.load_local(PRODUCT_FAISS_PATH, _embedder, allow_dangerous_deserialization=True)

    st.info("Đang tạo chỉ mục sản phẩm...")
    print("INFO: Bắt đầu tạo chỉ mục FAISS cho sản phẩm...")
    
    try:
        df = pd.read_csv(PRODUCT_CSV_FILE, encoding='utf-8')
        df.columns = [col.strip() for col in df.columns]

        documents = []
        fields_to_chunk = [
            "CÔNG DỤNG", "THÀNH PHẦN", "CHỈ ĐỊNH",
            "CHỐNG CHỈ ĐỊNH", "CÁCH DÙNG", "BẢO QUẢN", "LƯU Ý KHI SỬ DỤNG"
        ]

        for _, row in df.iterrows():
            product_name = row.get("SẢN PHẨM", "Không rõ tên")
            for field in fields_to_chunk:
                if field in row and pd.notna(row[field]):
                    chunk_content = f"Sản phẩm: {product_name}\nThông tin về '{field}': {row[field]}"
                    documents.append(chunk_content)

        if not documents:
            st.error("Lỗi: Không thể tạo được chunks từ file sản phẩm.")
            st.stop()
        
        print(f"INFO: Đã tạo được {len(documents)} chunks sản phẩm.")
        vectorstore = FAISS.from_texts(texts=documents, embedding=_embedder)
        vectorstore.save_local(PRODUCT_FAISS_PATH)
        print(f"INFO: Lưu chỉ mục sản phẩm thành công.")
        st.success("Tạo chỉ mục sản phẩm thành công!")
        return vectorstore
        
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file '{PRODUCT_CSV_FILE}'.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi tạo chỉ mục sản phẩm: {e}")
        st.stop()

@st.cache_resource
def load_or_create_script_faiss(_embedder):
    """Tải hoặc tạo FAISS index cho kịch bản Q&A."""
    if os.path.exists(SCRIPT_FAISS_PATH):
        print(f"INFO: Đang tải chỉ mục kịch bản từ '{SCRIPT_FAISS_PATH}'...")
        return FAISS.load_local(SCRIPT_FAISS_PATH, _embedder, allow_dangerous_deserialization=True)

    st.info("Đang tạo chỉ mục kịch bản...")
    print("INFO: Bắt đầu tạo chỉ mục FAISS cho kịch bản...")

    try:
        df = pd.read_csv(SCRIPT_CSV_FILE, encoding='utf-8')
        documents = [
            f"Câu hỏi: {row['Câu hỏi']}\nTrả lời: {row['Trả lời']}"
            for _, row in df.iterrows() if pd.notna(row['Câu hỏi']) or pd.notna(row['Trả lời'])
        ]
        
        if not documents:
            st.error("Lỗi: Không tìm thấy dữ liệu hợp lệ trong file kịch bản.")
            st.stop()
        
        print(f"INFO: Đã tạo được {len(documents)} documents kịch bản.")
        vectorstore = FAISS.from_texts(texts=documents, embedding=_embedder)
        vectorstore.save_local(SCRIPT_FAISS_PATH)
        print(f"INFO: Lưu chỉ mục kịch bản thành công.")
        st.success("Tạo chỉ mục kịch bản thành công!")
        return vectorstore
        
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file '{SCRIPT_CSV_FILE}'.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi tạo chỉ mục kịch bản: {e}")
        st.stop()

# --- Cấu hình API ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("Lỗi: Vui lòng thiết lập GEMINI_API_KEY trong Streamlit Secrets.")
    st.stop()

# --- Khởi Tạo Agent Manager ---
@st.cache_resource
def get_agent_manager():
    return AgentManager()

# --- STREAMLIT UI ---
st.set_page_config(page_title="EKS Agentic RAG System", page_icon="🤖")
st.title("🤖 EKS Agentic RAG System")
st.caption("Hệ thống AI Master Agent với RAG thông minh từ nhiều nguồn dữ liệu")

# Sidebar thông tin hệ thống
with st.sidebar:
    st.header("🎯 Agentic RAG System")
    st.write("**🧠 Master Agent**: Đánh giá và quyết định nguồn thông tin")
    st.write("**📚 Script Database**: Kịch bản Q&A có sẵn")
    st.write("**🛍️ Product Database**: Chi tiết sản phẩm EKS")
    
    st.header("🔍 Debug Mode")
    show_decision = st.checkbox("Hiển thị quyết định Master Agent")
    show_sources = st.checkbox("Hiển thị nguồn thông tin")

# Khởi tạo Agent Manager
try:
    agent_manager = get_agent_manager()
    st.success("✅ Agentic RAG system initialized successfully!")
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo agentic RAG system: {e}")
    st.stop()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": """
🎉 **Chào mừng đến với EKS Agentic RAG System!**

**Hệ thống hoạt động như thế nào:**
- 🧠 **Master Agent** truy vấn đồng thời cả database kịch bản và sản phẩm
- 🎯 Tự động đánh giá và chọn nguồn thông tin tốt nhất
- 📋 Ưu tiên câu trả lời từ kịch bản Q&A nếu phù hợp
- 🔬 Bổ sung thông tin chi tiết từ database sản phẩm khi cần

**Bạn có thể hỏi bất kỳ điều gì về:**
- Sản phẩm EKS (thành phần, công dụng, cách dùng...)
- Chính sách, dịch vụ, hướng dẫn mua hàng
- So sánh và tư vấn lựa chọn sản phẩm

Hãy thử hỏi để trải nghiệm sức mạnh của Agentic RAG! 🚀
        """}
    ]

# Hiển thị chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Hỏi EKS Agentic RAG System..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with Agentic RAG system
    with st.chat_message("assistant"):
        with st.spinner("🧠 Master Agent đang phân tích và truy vấn dữ liệu..."):
            # 1. Process query với Master Agent
            processing_result = agent_manager.process_query(prompt)
            decision = processing_result['master_decision']
            script_infos = processing_result['script_infos']
            product_infos = processing_result['product_infos']
            
            # 2. Show decision info if debug enabled
            if show_decision:
                with st.expander("🧠 Master Agent Decision"):
                    st.json({
                        "primary_source": decision.primary_source.value,
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                        "response_strategy": decision.response_strategy,
                        "selected_info_count": len(decision.selected_info)
                    })
            
            # 3. Show sources if debug enabled
            if show_sources:
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("📚 Script Sources"):
                        for i, info in enumerate(script_infos[:3]):
                            st.write(f"**Score:** {info.score:.3f}")
                            st.write(info.content[:200] + "..." if len(info.content) > 200 else info.content)
                            st.divider()
                
                with col2:
                    with st.expander("🛍️ Product Sources"):
                        for i, info in enumerate(product_infos[:3]):
                            st.write(f"**Score:** {info.score:.3f}")
                            st.write(info.content[:200] + "..." if len(info.content) > 200 else info.content)
                            st.divider()
            
            # 4. Display Master Agent response strategy
            strategy_emoji = {
                SourceType.SCRIPT: "📚",
                SourceType.PRODUCT: "🛍️", 
                SourceType.HYBRID: "🔄"
            }
            
            emoji = strategy_emoji.get(decision.primary_source, "🤖")
            st.write(f"{emoji} **Master Agent** (Strategy: {decision.primary_source.value}) đang trả lời...")
            
            # 5. Stream response from Master Agent
            response_generator = agent_manager.get_response_stream(prompt, decision)
            full_response = st.write_stream(response_generator)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})