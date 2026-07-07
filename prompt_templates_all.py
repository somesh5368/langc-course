"""
Prompt Templates and Messages in LangChain V.1
"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def demo_basic_templates():
    """Basic ChatPromptTemplate usage"""

    #Simple template
    simple = ChatPromptTemplate.from_template("Translate '{text}' to {language}")

    messages = simple.format_messages(text = "Hello, world!", language="French")
    print("Simple templates:")
    print(f"{messages}")

    # Multi-message template
    multi = ChatPromptTemplate.from_messages(
        [
            ("system","You are a translator. Be concise"),
            ("human", "Translate '{text}' to {language}"),

        ]
    )

    messages = multi.format_messages(text= "Good morning", language="Japanese")
    print("\nMulti-message template:")
    for msg in messages:
        print(f" {type(msg).__name__}:{msg.content}")

def demo_message_types():
    """Working with different message types."""

    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    #Build conversion manually
    messages = [
        SystemMessage(content="You are a math tutor. Be brief."),
        HumanMessage(content="What's 5 * 5?"),
        AIMessage(content="25"),
        HumanMessage(content="And if I add 10?")
    ]

    response = model.invoke(messages)
    print(f"Conversion result: {response.content}")


    def demo_messages_placeholder():
        """Use MessagesPlaceholder for dynamic conversation history."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "question"),
            ]
        )


    history = [
        HumanMessage(content = "My name is Paulo"),
        AIMessage(content="Nice to meet you, Paulo"),
    ]

    messages = prompt.format_messages(history=history, question ="What's my name?")

    print("With history placeholder:")
    for msg in messages:
        print(f" {types(msg).__name__}: {msg.content[:50]}...")

    # Execute
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = model.invoke(messages)