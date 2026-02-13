# Level 2 — Memory, making AI remember past conversations too.
# ConversationBufferMemory
# Chat history
# Stateful chains

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv() # to load the env

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini"
)

# Prompt template with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create chain with message history
chain = prompt | llm 

# store the chat history
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap chain with memory
chat_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("Type 'exit' to stop.\n")

session_id = "user_1"

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        break

    response = chat_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    print("AI:", response.content)