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
from requests.exceptions import RequestException

# Загружаем переменные окружения (если работаем локально)
# load_dotenv(verbose=True)
# Определяем, работает ли код локально (например, если `.env` существует)
is_local = os.path.exists(".env")

# Загружаем переменные окружения, если работаем локально
if is_local:
    load_dotenv(verbose=True)

# Загружаем API-ключи через Streamlit secrets (для облачного запуска)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    USER_AGENT = st.secrets["USER_AGENT"]
    LANGSMITH_TRACING = st.secrets["LANGSMITH_TRACING"] 
    LANGSMITH_ENDPOINT = st.secrets["LANGSMITH_ENDPOINT"]
    LANGSMITH_API_KEY = st.secrets["LANGSMITH_API_KEY"]
    LANGSMITH_PROJECT = st.secrets["LANGSMITH_PROJECT"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    # Если secrets.toml не найден, используем переменные окружения
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    USER_AGENT = os.getenv("USER_AGENT")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING") 
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Проверяем, заданы ли API-ключи
if not GROQ_API_KEY:
    st.error("Ошибка: GROQ_API_KEY не задана в переменных окружения.")
    st.stop()
if not USER_AGENT:
    st.error("Ошибка: USER_AGENT не задана в переменных окружения.")
    st.stop()
if not LANGSMITH_TRACING:
    st.error("Ошибка: LANGSMITH_TRACING не задана в переменных окружения.")
    st.stop()
if not LANGSMITH_ENDPOINT:
    st.error("Ошибка: LANGSMITH_ENDPOINT не задана в переменных окружения.")
    st.stop()
if not LANGSMITH_API_KEY:
    st.error("Ошибка: LANGSMITH_API_KEY не задана в переменных окружения.")
    st.stop()
if not LANGSMITH_PROJECT:
    st.error("Ошибка: LANGSMITH_PROJECT не задана в переменных окружения.")
    st.stop()
if not OPENAI_API_KEY :
    st.error("Ошибка: OPENAI_API_KEY  не задана в переменных окружения.")
    st.stop()

# Настройка LLM
try:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.6,
        api_key=GROQ_API_KEY
    )
    print("[DEBUG] LLM успешно инициализирован")
except Exception as e:
    print(f"[ERROR] Ошибка инициализации LLM: {e}")

# Настройка эмбеддингов
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
print("[DEBUG] Модель эмбеддингов загружена")

# Функция загрузки только англоязычного контента
def load_english_pages(urls):
    english_docs = []
    for url in urls:
        if not any(lang in url for lang in ["/ru", "/ar", "/es", "/ch"]):  
            try:
                loader = WebBaseLoader(url)
                documents = loader.load()
                if documents:
                    english_docs.extend(documents)
                    print(f"[DEBUG] Загружен контент с {url}")
            except RequestException as e:
                print(f"[ERROR] Ошибка загрузки страницы {url}: {e}")
    return english_docs

# Пример URL, где английские страницы без префиксов
urls = ["https://status.law/about", "https://status.law/", "https://status.law/contact"]
documents = load_english_pages(urls)

# Разбиваем на фрагменты
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"[DEBUG] Разбито на {len(chunks)} фрагментов")

# Создание векторного хранилища
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings_model)
retriever = vector_store.as_retriever()
print("[DEBUG] Векторное хранилище создано")

# Промпт для бота
template = """
You are a helpful legal assistant that answers questions based on information from status.law.
Answer accurately and concisely.
Question: {question}
Only use the provided context to answer the question.
Context: {context}
"""
prompt = PromptTemplate.from_template(template)

# История сообщений
def format_history(message_history):
    formatted = ""
    for msg in message_history:
        formatted += f"User: {msg['question']}\nBot: {msg['answer']}\n\n"
    return formatted

message_history = []

# Интерфейс Streamlit
st.set_page_config(page_title="Legal Chatbot", page_icon="🤖")
st.title("🤖 Legal Chatbot")
st.write("Этот бот отвечает на юридические вопросы, используя информацию с сайта status.law.")

# Поле для ввода вопроса
user_input = st.text_input("Введите ваш вопрос:")
if st.button("Отправить"):
    if user_input:
        # Получаем релевантные документы из ретривера
        retrieved_docs = retriever.get_relevant_documents(user_input)
        
        # Формируем текстовый контекст из документов
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Создание цепочки обработки запроса
        chain = (
            RunnableLambda(lambda x: {"context": context_text, "question": x["question"]}) 
            | prompt
            | llm
            | StrOutputParser()
        )

        # Запуск цепочки
        response = chain.invoke({"question": user_input, "context": context_text})

        # Добавляем в историю сообщений
        message_history.append({"question": user_input, "answer": response})

        # Выводим ответ
        st.write(response)
