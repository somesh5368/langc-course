"""
Multiple providers, configuration, streaming, and cost optimization
"""

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

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


def demo_message():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Using message objects (more control over roles)
    messages = [
        SystemMessage(content="You are a pirate. Always answer like a pirate."),
        HumanMessage(content="What's the weather like today?"),
    ]
    print("Using message objects:")
    print(f"Message:{messages[0]} |  {messages[1]}")

    response = model.invoke(messages)
    print(f"Response from Pirate: {response.content}")

    #Multi-turn conversation using message objects
    messages.append(response) # add model's response to the conversation
    messages.append(HumanMessage(content="What about tomorrow?"))

    print("\nMulti-turn conversation:")
    response = model.invoke(messages)
    print(f"Follow-up response from the Pirate: {response.content}")
    

    import os
from langchain.chat_models import init_chat_model


#Excercise multi model

def get_multi_model_responses(question: str, model_names: list[str]) -> dict:
    """
    Exercise: Create a function that:
    1. Takes a question and list of model names
    2. Gets a response from all models
    3. Returns a dict of {model_name: response}
    """

    # Map friendly model names -> (provider model id, provider)
    model_registry = {
        "gemini-2.5-flash": ("gemini-2.5-flash", "google_genai"),
        "gemini-2.5-flash-lite": ("gemini-2.5-flash-lite", "google_genai"),
        "llama-3.3-70b-versatile": ("llama-3.3-70b-versatile", "groq"),
    }

    results = {}

    for name in model_names:
        if name not in model_registry:
            results[name] = f"Error: unknown model '{name}'"
            continue

        model_id, provider = model_registry[name]

        # Skip Groq models if no API key is set
        if provider == "groq" and not os.getenv("GROQ_API_KEY"):
            results[name] = "Error: GROQ_API_KEY not set"
            continue

        try:
            model = init_chat_model(
                model=model_id,
                model_provider=provider,
                temperature=0.7,
                streaming=False,
            )
            response = model.invoke(question)
            results[name] = response.content
        except Exception as e:
            results[name] = f"Error: {e}"

    return results


def exercise_multi_model():
    prompt = "What is AI."
    models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "llama-3.3-70b-versatile"]

    responses = get_multi_model_responses(prompt, models)

    print(f"Prompt: {prompt}\n")
    for model_name, response in responses.items():
        print(f"Response from {model_name}: {response}\n")

    return responses



if __name__ == "__main__":
        #demo_init_chat_model()
        #demo_model_comparison()
        #demo_message()
        exercise_multi_model()

