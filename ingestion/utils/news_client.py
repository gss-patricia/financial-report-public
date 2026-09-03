from typing import Dict, List

import trafilatura
import yfinance as yf


class NewsClient:
    def fetch_news(self, ticker: str, max_stories: int = 10) -> List[Dict[str, any]]:
        data = yf.Ticker(ticker)
        news = data.news

        news_data = []

        for item in news[:max_stories]:
            content = item.get("content", {})
            content_type = content.get("contentType")

            if content_type != "STORY":
                continue

            canonical_url = content.get("canonicalUrl") or {}
            title = content.get("title")
            date = content.get("pubDate")
            url = canonical_url.get("url")

            if not url or "finance.yahoo.com" not in url:
                continue

            # One article failing to download must not take the rest of the
            # ingestion down: networks drop, Yahoo blocks, pages disappear.
            try:
                downloaded = trafilatura.fetch_url(url)
            except Exception as exc:
                print(f"failed to download {url}: {exc}")
                continue

            if not downloaded:
                print(f"no content at {url}")
                continue

            text_content = trafilatura.extract(downloaded)

            if text_content:
                metadata = {
                    "ticker": ticker,
                    "title": title,
                    "url": url,
                    "date": date,
                    "source": "yahoo_finance",
                }
                news_data.append({"text": text_content, "metadata": metadata})

        return news_data
