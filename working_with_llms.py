"""
Multiple providers, configuration, streaming, and cost optimization
"""

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

load_dotenv()


def demo_init_chat_model():
    chat_model = init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        temperature=0.7,
        streaming=True,
        max_retries=3,
    )

    response = chat_model.invoke("What is the capital of france? Answer in one word")
    print(f"response: {response.content}")

    # Easy to switch model providers
    if os.getenv("GROQ_API_KEY"):
        groq = init_chat_model(
            model="llama-3.3-70b-versatile",
            model_provider="groq",
            temperature=0.7,
            streaming=True,
            max_retries=3,
        )
        response = groq.invoke("What is the capital of france? Answer in one word")
        print(f"response from groq: {response.content}")
    else:
        print("Skipping Groq: GROQ_API_KEY not set")


def demo_model_comparison():
    prompt = "Explain recursion in one sentence."
    models = {
        "gemini-2.5-flash": init_chat_model(
            model="gemini-2.5-flash",
            model_provider="google_genai",
            temperature=0.7,
            streaming=False,
        ),
        "gemini-2.5-flash-lite": init_chat_model(
            model="gemini-2.5-flash-lite",
            model_provider="google_genai",
            temperature=0.7,
            streaming=False,
        ),
    }

    if os.getenv("GROQ_API_KEY"):
        models["llama-3.3-70b-versatile"] = init_chat_model(
            model="llama-3.3-70b-versatile",
            model_provider="groq",
            temperature=0.7,
            streaming=False,
        )

    print(f"Prompt: {prompt}\n")
    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")


if __name__ == "__main__":
    demo_init_chat_model()
    demo_model_comparison()