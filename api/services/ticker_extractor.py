import instructor
from config.company_mappings import COMPANY_TICKER_MAPPINGS, TICKER_EXTRACTION_PROMPT
from config.settings import settings
from groq import Groq
from pydantic import BaseModel, Field


class TickerResult(BaseModel):
    ticker: str = Field(
        pattern="^[A-Z]{1,5}$",
        description="Stock ticker symbol (1-5 uppercase letters)",
    )


class TickerExtractor:
    def __init__(self):
        client = Groq(api_key=settings.groq_api_key)
        self.client = instructor.from_groq(client, mode=instructor.Mode.JSON)
        self.mappings = COMPANY_TICKER_MAPPINGS

    def extract_ticker(self, query: str) -> str | None:
        query_lower = query.lower()
        for company_name, ticker in self.mappings.items():
            if company_name in query_lower:
                return ticker

        return self._extract_with_llm(query)

    def _extract_with_llm(self, query: str) -> str | None:
        prompt = TICKER_EXTRACTION_PROMPT.format(query=query)

        result = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_model=TickerResult,
        )
        return result.ticker
