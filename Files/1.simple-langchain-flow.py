# Level 1 - Input → Prompt → LLM → Output

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.callbacks import get_openai_callback

load_dotenv() # to load the env

# to configure and call the model
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini"
)

# creating a prompt template
prompt = ChatPromptTemplate.from_template(
    "{query}"
)

# output parser
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

user_prompt = input("Enter your query: ")

# invoking the chain with token usage tracking
with get_openai_callback() as cb:
    result = chain.invoke({"query": user_prompt})
    print(result)
    print("\n--- Token Usage ---")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost:.6f}")