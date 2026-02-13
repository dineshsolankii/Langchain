# Document → Split → Embed → Store → Retrieve → LLM answers

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1️⃣ Load LLM (OpenRouter)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)

# 2️⃣ Load document
loader = TextLoader("knowledge.txt")
documents = loader.load()

# 3️⃣ Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)
docs = text_splitter.split_documents(documents)

# 4️⃣ Create embeddings (OpenRouter format)
embeddings = OpenAIEmbeddings(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/text-embedding-3-small",
)

# 5️⃣ Store in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# 6️⃣ Create retriever
retriever = vectorstore.as_retriever()

# 7️⃣ Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question only using the context below.

Context:
{context}

Question:
{question}
""")

# 8️⃣ RAG Chain
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    return response.content

# 9️⃣ Chat loop
print("Type 'exit' to stop.\n")

while True:
    user_input = input("Ask a question: ")
    
    if user_input.lower() == "exit":
        break
    
    answer = rag_chain(user_input)
    print("\nAI:", answer)
