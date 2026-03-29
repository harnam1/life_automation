import feedparser
from newspaper import Article
import anthropic
import time

CLIENT = anthropic.Anthropic()
MODEL = "claude-opus-4-6"

RSS_FEEDS = [
    "https://globallnghub.com/feed",
    "https://www.lngworldnews.com/feed/",
    "https://www.lngindustry.com/feed/"
]


def fetch_rss_articles():
    articles = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:5]:
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "summary": entry.get("summary", "")
            })
    return articles


def fetch_full_article(link):
    try:
        article = Article(link)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch {link}: {e}")
        return None


def generate_questions(title: str, text: str) -> str:
    prompt = f"""You are an LNG market analyst. Based on this article, generate exactly 3 comprehension questions:
- 1 main analytical question (requires understanding the broader market implication)
- 2 rapid-fire factual questions (specific numbers, names, dates, or decisions mentioned)

Article title: {title}

Article text:
{text[:2000]}

Format your response as:
MAIN Q: <question>

RAPID 1: <question>
RAPID 2: <question>"""

    response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def main():
    print("Fetching LNG articles...\n")
    rss_articles = fetch_rss_articles()
    structured_articles = []

    for art in rss_articles:
        print(f"Fetching: {art['title']}")
        full_text = fetch_full_article(art['link'])
        if full_text:
            structured_articles.append({
                "title": art["title"],
                "link": art["link"],
                "text": full_text[:2000]
            })
        else:
            print("  Could not extract full text")
        time.sleep(1)

    print(f"\nFetched {len(structured_articles)} articles. Generating questions...\n")
    print("=" * 70)

    for i, art in enumerate(structured_articles, 1):
        print(f"\n[{i}/{len(structured_articles)}] {art['title']}")
        print(f"  {art['link']}\n")
        questions = generate_questions(art["title"], art["text"])
        print(questions)
        print("-" * 70)

        # CLI drill: prompt user for answers
        input("\nPress Enter to continue to next article...")
        print()


if __name__ == "__main__":
    main()
