from langchain_core.messages import HumanMessage # high level framework that allows us to build AI applications
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent # complex framework that allows us to build AI agents
from dotenv import load_dotenv # allows us to load env variable files from within our Python script (our api key)
import time

load_dotenv()
import openai

@tool
# define any tool, useful for website-specific questions
def calculator(a:float,b:float) -> str:
    """Useful for performing basic arithmetic calculations with numbers"""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a+b}"

def main():
    model = ChatOpenAI(temperature=0) # temperature refers to randomness

    tools = [calculator]
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I am your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me!")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "quit":
            break

        print("\nAssistant: ", end = "")
        try:
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}
            ):
                if "agent" in chunk and "messages" in chunk["agent"]:
                    for message in chunk["agent"]["messages"]: # types in each word individually instead of all at once
                        print(message.content, end="")
            print()
        except Exception as e:
            # Handle OpenAI quota/rate limit errors gracefully.
            try:
                # The OpenAI package exposes RateLimitError in different places depending on version.
                RateLimitClass = None
                if hasattr(openai, "RateLimitError"):
                    RateLimitClass = openai.RateLimitError
                elif hasattr(openai, "error") and hasattr(openai.error, "RateLimitError"):
                    RateLimitClass = openai.error.RateLimitError

                if RateLimitClass is not None and isinstance(e, RateLimitClass):
                    print("\nAssistant: The OpenAI API returned a rate limit/quota error. Check your billing/plan or try again later.")
                else:
                    print(f"\nAssistant: Error: {e}")
            except Exception:
                print(f"\nAssistant: Error: {e}")

if __name__ == "__main__":
    main()            