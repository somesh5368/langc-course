from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

from langchain_core import __version__ as core_version
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {version('langgraph')}")


def main():
     # test Gemini
    llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    response_gemini = llm_gemini.invoke("Say 'setup complete!' in one word")
    print(f"Response from Gemini: {response_gemini}")

    # test Groq
    llm_groq = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response_groq = llm_groq.invoke("Say 'setup complete!' in one word")
    print(f"Response from Groq: {response_groq}")

    print("setup complete!")


if __name__ == "__main__":
    main()