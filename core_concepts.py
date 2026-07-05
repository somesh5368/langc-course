""
""


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables."""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question": "What is LangChain?"})
    print(f"Response: {result}")

    return chain

def demo_batch_exectution():

    """Demonstrate batch execution for multiple inputs."""
    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name"},
        {"text": "where is the nearest Restaurant?"},
    ]
    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")

def demo_streaming():
    """Demonstrate streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7,)
    parser = StrOutputParser()

    chain = prompt | model | parser

    #Streaming - run with streaming enabled
    print("Streaming output:")
    for chunk in chain.stream({"topic":"nature"}):
        print(chunk, end="", flush=True)
    print() # for newline after streaming

def demo_schema_inspection():
    """Demonstrate input/output schema inspection."""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspect input and output schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")


"""

    # Excercise: Create a chain that:
    # 1. Takes a product name and target audience
    # 2. Generates a marketing tagline.
    # 3. Return just the tagline as a string.
    
    Test with: product"AI Course", audience="developer"
"""
def generate_tagline():
    """Demonstrate input/output schema inspection."""
    prompt = ChatPromptTemplate.from_template("Generate tagline  on the basis of : '{product}' and target '{audience}'")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

      # Execute the chain with an input
    result = chain.invoke({"product": "AI COURSE", "audience":"developers"})
    print(f"Marketing Tagline: {result}")

   

def new_way():
    model = init_chat_model("gpt-40-mini", temperature=0.7, max_tokens=1500)




if __name__ == "__main__":
    #demo_basic_chain()
   # demo_batch_exectution()
  # demo_streaming()
  #demo_schema_inspection()
  generate_tagline()



