import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from requests.exceptions import RequestException, Timeout

# Загрузка переменных окружения
if os.path.exists(".env"):
    load_dotenv(verbose=True)

# Загрузка API-ключей
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    USER_AGENT = st.secrets["USER_AGENT"]
    LANGSMITH_TRACING = st.secrets["LANGSMITH_TRACING"] 
    LANGSMITH_ENDPOINT = st.secrets["LANGSMITH_ENDPOINT"]
    LANGSMITH_API_KEY = st.secrets["LANGSMITH_API_KEY"]
    LANGSMITH_PROJECT = st.secrets["LANGSMITH_PROJECT"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    USER_AGENT = os.getenv("USER_AGENT")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING") 
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Проверка API-ключей
if not all([GROQ_API_KEY, USER_AGENT, LANGSMITH_TRACING, LANGSMITH_ENDPOINT, LANGSMITH_API_KEY, LANGSMITH_PROJECT, OPENAI_API_KEY]):
    st.error("Ошибка: Не все переменные окружения заданы.")
    st.stop()

# Инициализация LLM
try:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.6, api_key=GROQ_API_KEY)
    print("[DEBUG] LLM успешно инициализирован")
except Exception as e:
    st.error(f"Ошибка инициализации LLM: {e}")
    st.stop()

# Инициализация эмбеддингов
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
print("[DEBUG] Модель эмбеддингов загружена")

# Список страниц для анализа (вручную перечислены)
urls = [
    "https://status.law/about",
    "https://status.law/services",
    "https://status.law/team",
    "https://status.law/blog",
    "https://status.law/faq"
]

# Загрузка данных
@st.cache_data
def load_data(urls):
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents.extend(loader.load(timeout=10))
            print(f"[DEBUG] Загружен контент с {url}")
        except (RequestException, Timeout) as e:
            print(f"[ERROR] Ошибка загрузки страницы {url}: {e}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"[DEBUG] Разбито на {len(chunks)} фрагментов")
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings_model)
    retriever = vector_store.as_retriever()
    print("[DEBUG] Векторное хранилище создано")
    return retriever

# Загрузка данных и создание ретривера
if "retriever" not in st.session_state:
    st.session_state.retriever = load_data(urls)
retriever = st.session_state.retriever

# Промпт для бота
template = """
You are a helpful legal assistant that answers questions based on information from status.law.
Answer accurately and concisely.
Question: {question}
Only use the provided context to answer the question.
Context: {context}
"""
prompt = PromptTemplate.from_template(template)

# Инициализация цепочки обработки запроса
if "chain" not in st.session_state:
    st.session_state.chain = (
        RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]}) 
        | prompt
        | llm
        | StrOutputParser()
    )
chain = st.session_state.chain

# Интерфейс Streamlit
st.set_page_config(page_title="Legal Chatbot", page_icon="🤖")
st.title("🤖 Legal Chatbot")
st.write("Этот бот отвечает на юридические вопросы, используя информацию с сайта status.law.")

# Поле для ввода вопроса
user_input = st.text_input("Введите ваш вопрос:")
if st.button("Отправить") and user_input:
    retrieved_docs = retriever.get_relevant_documents(user_input)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    response = chain.invoke({"question": user_input, "context": context_text})
    
    # Сохранение истории сообщений
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    st.session_state.message_history.append({"question": user_input, "answer": response})
    
    # Вывод ответа
    st.write(response)

# Вывод истории сообщений
if "message_history" in st.session_state:
    st.write("### История сообщений")
    for msg in st.session_state.message_history:
        st.write(f"**User:** {msg['question']}")
        st.write(f"**Bot:** {msg['answer']}")
