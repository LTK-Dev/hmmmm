import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

# Import các thư viện cần thiết cho FAISS và LangChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- Data Classes và Enums ---
class AgentType(Enum):
    ROUTER = "router"
    PRODUCT_SPECIALIST = "product_specialist"
    GENERAL_CONSULTANT = "general_consultant"

@dataclass
class AgentResponse:
    content: str
    confidence: float
    agent_type: AgentType
    metadata: Dict[str, Any] = None

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

# --- Base Agent Class ---
class BaseAgent:
    def __init__(self, agent_type: AgentType, model: genai.GenerativeModel):
        self.agent_type = agent_type
        self.model = model
        self.name = agent_type.value
    
    def process(self, request: TaskRequest) -> AgentResponse:
        """Base method để xử lý request - sẽ được override bởi các agent con"""
        raise NotImplementedError("Subclasses must implement process method")
    
    def _generate_stream_response(self, prompt: str):
        """Helper method để generate streaming response"""
        try:
            response_stream = self.model.generate_content(prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            st.error(f"Lỗi khi generate response từ {self.name}: {e}")
            yield f"Xin lỗi, {self.name} gặp lỗi khi xử lý yêu cầu."

# --- Router Agent ---
class RouterAgent(BaseAgent):
    def __init__(self, model: genai.GenerativeModel):
        super().__init__(AgentType.ROUTER, model)
        self.routing_prompt = """
        Bạn là Router Agent của hệ thống EKS AI Assistant. Nhiệm vụ của bạn là phân tích câu hỏi và quyết định agent nào sẽ xử lý.

        **Các Agent có sẵn:**
        1. **product_specialist**: Chuyên gia sản phẩm - xử lý câu hỏi về:
           - Thông tin chi tiết sản phẩm (công dụng, thành phần, cách dùng)
           - Chỉ định, chống chỉ định của sản phẩm
           - Bảo quản, lưu ý khi sử dụng
           - Tác dụng phụ, liều lượng

        2. **general_consultant**: Tư vấn viên chung - xử lý câu hỏi về:
           - So sánh nhiều sản phẩm
           - Thông tin về công ty, thương hiệu
           - Chính sách, dịch vụ khách hàng
           - Địa chỉ mua hàng, liên hệ
           - Câu hỏi tổng quát về ngành mỹ phẩm

        **Câu hỏi cần phân tích:**
        "{query}"

        **Yêu cầu:**
        Trả lời CHÍNH XÁC theo format JSON sau:
        {{
            "agent": "product_specialist" hoặc "general_consultant",
            "confidence": số từ 0.0 đến 1.0,
            "reasoning": "lý do ngắn gọn cho quyết định"
        }}

        CHỈ trả lời JSON, không thêm text nào khác.
        """
    
    def process(self, request: TaskRequest) -> Dict[str, Any]:
        """Phân tích và route câu hỏi đến agent phù hợp"""
        try:
            prompt = self.routing_prompt.format(query=request.query)
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            routing_decision = json.loads(response.text.strip())
            
            return {
                'target_agent': routing_decision.get('agent', 'general_consultant'),
                'confidence': routing_decision.get('confidence', 0.5),
                'reasoning': routing_decision.get('reasoning', 'Phân tích tự động')
            }
            
        except json.JSONDecodeError:
            # Fallback nếu JSON parsing thất bại
            st.warning("Router Agent trả về format không đúng, sử dụng fallback logic")
            return self._fallback_routing(request.query)
        except Exception as e:
            st.error(f"Lỗi Router Agent: {e}")
            return self._fallback_routing(request.query)
    
    def _fallback_routing(self, query: str) -> Dict[str, Any]:
        """Fallback routing logic nếu AI routing thất bại"""
        query_lower = query.lower()
        
        product_keywords = ['sản phẩm', 'công dụng', 'thành phần', 'cách dùng', 'chỉ định', 'tác dụng']
        general_keywords = ['so sánh', 'công ty', 'mua ở đâu', 'giá', 'chính sách']
        
        product_score = sum(1 for kw in product_keywords if kw in query_lower)
        general_score = sum(1 for kw in general_keywords if kw in query_lower)
        
        if product_score > general_score:
            return {'target_agent': 'product_specialist', 'confidence': 0.7, 'reasoning': 'Fallback: Phát hiện từ khóa sản phẩm'}
        else:
            return {'target_agent': 'general_consultant', 'confidence': 0.7, 'reasoning': 'Fallback: Câu hỏi tổng quát'}

# --- Product Specialist Agent ---
class ProductSpecialistAgent(BaseAgent):
    def __init__(self, model: genai.GenerativeModel, vector_store: FAISS):
        super().__init__(AgentType.PRODUCT_SPECIALIST, model)
        self.vector_store = vector_store
        self.specialist_prompt = """
        Bạn là Product Specialist Agent của EKS - chuyên gia sâu về các sản phẩm mỹ phẩm. 
        Bạn có quyền truy cập vào cơ sở dữ liệu chi tiết về tất cả sản phẩm EKS.

        **Thông tin sản phẩm từ database:**
        ---
        {context}
        ---

        **Câu hỏi từ khách hàng:**
        {query}

        **Chuyên môn của bạn:**
        - Phân tích thành phần chi tiết
        - Giải thích cơ chế hoạt động
        - Hướng dẫn sử dụng chính xác
        - Cảnh báo chống chỉ định
        - Tư vấn bảo quản và lưu ý

        **Cách trả lời:**
        1. **Xưng hô**: "Tôi là chuyên gia sản phẩm EKS"
        2. **Phong cách**: Chuyên nghiệp, chi tiết, dựa trên khoa học
        3. **Cấu trúc**: Sử dụng bullet points và headers để dễ đọc
        4. **Trích dẫn**: Luôn nêu rõ tên sản phẩm cụ thể
        5. **Độ tin cậy**: Chỉ trả lời dựa trên data có sẵn, thừa nhận nếu không có thông tin

        **Lưu ý quan trọng**: Nếu không tìm thấy thông tin chính xác trong database, 
        hãy nói rõ "Tôi không tìm thấy thông tin chi tiết về vấn đề này trong cơ sở dữ liệu sản phẩm EKS."
        """
    
    def process(self, request: TaskRequest) -> AgentResponse:
        # Retrieve relevant product information
        context = self._get_product_context(request.query)
        
        return AgentResponse(
            content="", # Sẽ được fill bởi streaming
            confidence=0.9,
            agent_type=self.agent_type,
            metadata={'context_size': len(context), 'source': 'product_database'}
        )
    
    def process_stream(self, request: TaskRequest):
        """Streaming version của process method"""
        context = self._get_product_context(request.query)
        prompt = self.specialist_prompt.format(context="\n\n".join(context), query=request.query)
        
        yield from self._generate_stream_response(prompt)
    
    def _get_product_context(self, query: str, k: int = 5) -> List[str]:
        """Lấy context từ product vector store"""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            st.error(f"Lỗi khi truy xuất thông tin sản phẩm: {e}")
            return []

# --- General Consultant Agent ---
class GeneralConsultantAgent(BaseAgent):
    def __init__(self, model: genai.GenerativeModel, vector_store: FAISS):
        super().__init__(AgentType.GENERAL_CONSULTANT, model)
        self.vector_store = vector_store
        self.consultant_prompt = """
        Bạn là General Consultant Agent của EKS - tư vấn viên chuyên về thông tin tổng quát, 
        so sánh sản phẩm và dịch vụ khách hàng.

        **Thông tin tham khảo từ kịch bản Q&A:**
        ---
        {context}
        ---

        **Câu hỏi từ khách hàng:**
        {query}

        **Chuyên môn của bạn:**
        - So sánh và đánh giá sản phẩm
        - Thông tin về thương hiệu EKS
        - Chính sách và dịch vụ
        - Hướng dẫn mua hàng và liên hệ
        - Tư vấn lựa chọn phù hợp

        **Cách trả lời:**
        1. **Xưng hô**: "Tôi là tư vấn viên EKS"
        2. **Phong cách**: Thân thiện, dễ hiểu, tập trung vào giải pháp
        3. **Ưu tiên**: Nếu có sẵn câu trả lời trong kịch bản, sử dụng y nguyên
        4. **Linh hoạt**: Có thể tham khảo kiến thức chung nếu cần
        5. **Hành động**: Đưa ra lời khuyên cụ thể và bước tiếp theo

        **Đặc biệt**: Nếu tìm thấy câu hỏi tương tự trong kịch bản, 
        hãy sử dụng chính xác câu trả lời đó.
        """
    
    def process(self, request: TaskRequest) -> AgentResponse:
        context = self._get_general_context(request.query)
        
        return AgentResponse(
            content="", # Sẽ được fill bởi streaming
            confidence=0.8,
            agent_type=self.agent_type,
            metadata={'context_size': len(context), 'source': 'qa_script'}
        )
    
    def process_stream(self, request: TaskRequest):
        """Streaming version của process method"""
        context = self._get_general_context(request.query)
        prompt = self.consultant_prompt.format(context="\n\n".join(context), query=request.query)
        
        yield from self._generate_stream_response(prompt)
    
    def _get_general_context(self, query: str, k: int = 3) -> List[str]:
        """Lấy context từ general Q&A vector store"""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            st.error(f"Lỗi khi truy xuất thông tin Q&A: {e}")
            return []

# --- Agent Manager ---
class AgentManager:
    def __init__(self):
        self.model = genai.GenerativeModel(GENERATIVE_MODEL)
        self.embedder = self._get_embedder()
        self.product_store = self._load_product_store()
        self.script_store = self._load_script_store()
        
        # Khởi tạo các agents
        self.router = RouterAgent(self.model)
        self.product_specialist = ProductSpecialistAgent(self.model, self.product_store)
        self.general_consultant = GeneralConsultantAgent(self.model, self.script_store)
        
        self.agents = {
            'product_specialist': self.product_specialist,
            'general_consultant': self.general_consultant
        }
    
    def _get_embedder(self):
        """Tải và cache mô hình embedding."""
        return get_embedder()
    
    def _load_product_store(self):
        """Load product FAISS store"""
        return load_or_create_product_faiss(self.embedder)
    
    def _load_script_store(self):
        """Load script FAISS store"""
        return load_or_create_script_faiss(self.embedder)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Xử lý query thông qua agent routing"""
        # 1. Router quyết định agent
        request = TaskRequest(query=query)
        routing_decision = self.router.process(request)
        
        # 2. Lấy target agent
        target_agent_name = routing_decision['target_agent']
        target_agent = self.agents.get(target_agent_name, self.general_consultant)
        
        return {
            'agent': target_agent,
            'routing_info': routing_decision,
            'request': request
        }

# --- FAISS Loading Functions (fixed caching issues) ---
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
st.set_page_config(page_title="EKS Agent-to-Agent System", page_icon="🤖")
st.title("🤖 EKS Agent-to-Agent AI System")
st.caption("Hệ thống AI đa tác nhân với Router Agent và Task Agents chuyên biệt")

# Sidebar thông tin agents
with st.sidebar:
    st.header("🎯 System Agents")
    st.write("**🧭 Router Agent**: Phân tích và điều phối")
    st.write("**👨‍🔬 Product Specialist**: Chuyên gia sản phẩm")
    st.write("**👨‍💼 General Consultant**: Tư vấn viên chung")
    
    st.header("🔍 Debug Mode")
    show_routing = st.checkbox("Hiển thị thông tin routing")

# Khởi tạo Agent Manager
try:
    agent_manager = get_agent_manager()
    st.success("✅ Agent system initialized successfully!")
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo agent system: {e}")
    st.stop()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": """
🎉 **Chào mừng đến với EKS Agent-to-Agent System!**

**Hệ thống có 3 AI Agents:**
- 🧭 **Router Agent**: Phân tích câu hỏi và điều phối
- 👨‍🔬 **Product Specialist**: Chuyên gia chi tiết về sản phẩm
- 👨‍💼 **General Consultant**: Tư vấn viên cho câu hỏi chung

**Bạn có thể hỏi:**
- Chi tiết về sản phẩm (thành phần, công dụng, cách dùng...)
- So sánh sản phẩm, thông tin thương hiệu
- Chính sách, dịch vụ, địa chỉ mua hàng

Hãy đặt câu hỏi để trải nghiệm hệ thống AI thông minh! 😊
        """}
    ]

# Hiển thị chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Hỏi EKS Agent System..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with Agent-to-Agent system
    with st.chat_message("assistant"):
        with st.spinner("🧭 Router Agent đang phân tích..."):
            # 1. Get routing decision
            processing_result = agent_manager.process_query(prompt)
            agent = processing_result['agent']
            routing_info = processing_result['routing_info']
            request = processing_result['request']
            
            # 2. Show routing info if debug enabled
            if show_routing:
                with st.expander("🔍 Agent Routing Information"):
                    st.json({
                        "selected_agent": routing_info['target_agent'],
                        "confidence": routing_info['confidence'],
                        "reasoning": routing_info['reasoning']
                    })
            
            # 3. Display which agent is responding
            agent_emoji = "👨‍🔬" if agent.agent_type == AgentType.PRODUCT_SPECIALIST else "👨‍💼"
            agent_name = "Product Specialist" if agent.agent_type == AgentType.PRODUCT_SPECIALIST else "General Consultant"
            
            st.write(f"{agent_emoji} **{agent_name} Agent** đang xử lý...")
            
            # 4. Stream response from selected agent
            response_generator = agent.process_stream(request)
            full_response = st.write_stream(response_generator)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})