import logging

from llm_service.llm_client import LLMClient
from llm_service.reasoning import build_prompt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def run_test():
    llm = LLMClient(
        model="ollama/llama3:8b",
        api_base="http://localhost:11434",
    )

    test_cases = [
        "I hate you",
        "You are so stupid",
        "Have a nice day!",
        "I will destroy you",
        "Let's work together on this project",
        "I guess you're not the smartest person",
        "Wow, you're actually useless",
        "That was... not very clever of you",
    ]

    for text in test_cases:
        print("\n" + "=" * 50)
        print(f"TEXT: {text}")

        prompt = build_prompt(
            text=text,
            examples = [
                "You are an idiot",
                "Go kill yourself",
                "Wow, you're useless",
                "That was not very smart of you",
                "I guess you're not very capable",
            ],
        )

        result = llm.complete(prompt)

        print("RESULT:", result)


if __name__ == "__main__":
    run_test()