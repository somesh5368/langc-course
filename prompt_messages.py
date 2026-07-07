from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

load_dotenv()

# #chatprompttemplate
# prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}")


# #format and inspect
# messages = prompt.format_messages(adjective="funny", topic="chickens")

# print(messages)
 
model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0,
)
 
# --- Translation via ChatPromptTemplate ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}",
        ),
        ("human", "translate the following text: {text}"),
    ]
)
 
translate_messages = prompt.format_messages(
    input_language="English",
    output_language="French",
    text="I love programming.",
)
 
response = model.invoke(translate_messages)
print(f"Translation: {response.content}")
 
# --- Message types demo (just showing the shapes, not sent to the model) ---
message_types_demo = [
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there! How can I assist you today?"),
    SystemMessage(content="This is a system message"),
    ToolMessage(content="Tool executed successfully", tool_call_id="call_123"),
    ChatMessage(content="This is a general chat message.", role="custom_role"),
]
for msg in message_types_demo:
    print(f"{type(msg).__name__}: {msg.content}")
 
# --- Few-shot example ---
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]
 
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
 
fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
 
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Give the opposite of each word."),
        fewshot_prompt,
        ("human", "{input}"),
    ]
)
 
response = model.invoke(final_prompt.format_messages(input="happy"))
print(f"Opposite of 'happy': {response.content}")


# Reusable components
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}.")
])
user_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}")
])



#Combine
full_prompt = system_prompt + user_prompt

fin = full_prompt.format_messages(role="helpful assistant", question="What is AI")
print(fin)