from dotenv import load_dotenv
from groq import Groq
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
)

from guardrails import Guard

load_dotenv()

client = Groq()


def groq_wrapper(*, messages, **kwargs) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
    )
    return response.choices[0].message.content


@register_validator(name="simple_topic_check", data_type="string")
def simple_topic_check(value, metadata):
    financial_keywords = ["stock", "apple", "investment", "ticker", "finance", "market"]

    if any(keyword in value.lower() for keyword in financial_keywords):
        return PassResult()
    else:
        return FailResult(error_message="Query is not about financial topics")


guard = Guard().use(simple_topic_check, on_fail="exception")

queries = [
    "How is Apple stock doing?",
    "What's the weather today?",
]

for query in queries:
    print(f"\nQuery: {query}")
    try:
        guard.validate(query)

        result = groq_wrapper(messages=[{"role": "user", "content": query}])
        print(result)
    except Exception as e:
        print(f"BLOCKED: {e}")
