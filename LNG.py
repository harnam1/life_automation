"""
LNG Daily Market Trainer — V1
Fetches LNG news, generates topical questions, grades answers,
tracks performance in SQLite, adapts topic weighting, and
auto-commits the updated player model to git.
"""

import feedparser
from newspaper import Article
import anthropic
import sqlite3
import json
import time
import os
import subprocess
from datetime import datetime, date

# —— Config ————————————————————————————————————————————————————————
CLIENT = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
DB_PATH = os.path.join(os.path.dirname(__file__), "lng_trainer.db")
PLAYER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "player_model.json")

TOPIC_TAGS = [
    "jkm_ttf_spreads",
    "henry_hub",
    "freight_shipping",
    "europe_demand_storage_weather",
    "asia_demand_jkm_drivers",
    "supply_outages_maintenance",
    "relative_value_route_economics",
    "headline_interpretation",
]

RSS_FEEDS = [
    "https://globallnghub.com/feed",
    "https://www.lngworldnews.com/feed/",
    "https://www.lngindustry.com/feed/",
]


# —— Database ——————————————————————————————————————————————————————
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            article_title   TEXT,
            article_link    TEXT,
            topic_tags      TEXT,       -- JSON list
            main_question   TEXT,
            rapid_1         TEXT,
            rapid_2         TEXT,
            user_main       TEXT,
            user_r1         TEXT,
            user_r2         TEXT,
            confidence      INTEGER,
            score_total     REAL,
            main_score      REAL,
            rapid_score     REAL,
            missed_concepts TEXT,       -- JSON list
            model_answer    TEXT,
            feedback        TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS player_model (
            id              INTEGER PRIMARY KEY CHECK (id = 1),
            total_sessions  INTEGER DEFAULT 0,
            streak_days     INTEGER DEFAULT 0,
            last_session    TEXT,
            avg_score_7d    REAL DEFAULT 0,
            avg_confidence  REAL DEFAULT 0,
            confidence_bias TEXT DEFAULT 'unknown',
            weak_topics     TEXT DEFAULT '{}',   -- JSON {tag: weight}
            strong_topics   TEXT DEFAULT '{}',
            common_errors   TEXT DEFAULT '[]',   -- JSON list
            updated_at      TEXT DEFAULT (datetime('now'))
        );

        INSERT OR IGNORE INTO player_model (id) VALUES (1);
    """)
    conn.commit()
    return conn


# —— Feed ingestion ————————————————————————————————————————————————
def fetch_rss_articles():
    articles = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "summary": entry.get("summary", ""),
                })
        except Exception as e:
            print(f"  Feed error ({feed_url}): {e}")
    return articles


def fetch_full_article(link):
    try:
        article = Article(link)
        article.download()
        article.parse()
        text = article.text.strip()
        return text if len(text) > 100 else None
    except Exception as e:
        print(f"  Extraction failed ({link}): {e}")
        return None


# —— Topic tagging ————————————————————————————————————————————————
def tag_article(title: str, text: str) -> list[str]:
    prompt = f"""You are an LNG market analyst. Tag this article with 1-3 of these exact topic tags:
{json.dumps(TOPIC_TAGS)}

Article title: {title}
Article excerpt: {text[:800]}

Reply with ONLY a JSON list of matching tags, e.g. ["freight_shipping", "asia_demand_jkm_drivers"]"""

    resp = CLIENT.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return json.loads(resp.content[0].text.strip())
    except (json.JSONDecodeError, IndexError):
        return ["headline_interpretation"]


# —— Question generation ——————————————————————————————————————————
def generate_questions(title: str, text: str, weak_topics: dict) -> dict:
    weak_hint = ""
    if weak_topics:
        top_weak = sorted(weak_topics, key=weak_topics.get)[:2]
        weak_hint = f"\nTry to angle questions toward these weak areas if relevant: {', '.join(top_weak)}"

    prompt = f"""You are an LNG market trainer for a junior analyst moving toward front office.
Based on this article, generate exactly 3 questions.{weak_hint}

Rules:
- MAIN Q: analytical — requires understanding the broader market implication, the "why" behind the move, or a commercial judgment call.
- RAPID 1: factual but topical — a specific number, name, decision, or market reference from the article.
- RAPID 2: interpretive — connects the article to a broader LNG theme (spreads, freight, demand, supply balance).

Article title: {title}
Article text: {text[:2000]}

Reply in EXACTLY this format (no extra text):
MAIN Q: <question>
RAPID 1: <question>
RAPID 2: <question>"""

    resp = CLIENT.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    lines = raw.split("\n")
    parsed = {"main": "", "r1": "", "r2": ""}
    for line in lines:
        line = line.strip()
        if line.startswith("MAIN Q:"):
            parsed["main"] = line[7:].strip()
        elif line.startswith("RAPID 1:"):
            parsed["r1"] = line[8:].strip()
        elif line.startswith("RAPID 2:"):
            parsed["r2"] = line[8:].strip()
    return parsed


# —— Grading ——————————————————————————————————————————————————————
def grade_answers(article_title: str, article_text: str, questions: dict,
                  answers: dict, confidence: int) -> dict:
    prompt = f"""You are grading an LNG trading trainee's daily drill answers.
Be honest and calibrated — partial credit for directionally correct reasoning,
penalise unsupported certainty, reward commercial framing.

Article title: {article_title}
Article text: {article_text[:2000]}

Questions and answers:
MAIN Q: {questions['main']}
ANSWER: {answers['main']}

RAPID 1: {questions['r1']}
ANSWER: {answers['r1']}

RAPID 2: {questions['r2']}
ANSWER: {answers['r2']}

Confidence self-rating: {confidence}/5

Reply in EXACTLY this JSON format (no markdown, no backticks):
{{
    "main_score": <0-7 float>,
    "rapid_score": <0-3 float, 1.5 per rapid>,
    "score_total": <0-10 float>,
    "missed_concepts": ["concept1", "concept2"],
    "model_answer": "<what a good answer would have been, 2-3 sentences>",
    "one_win": "<one thing the trainee got right>",
    "one_miss": "<one thing the trainee missed or got wrong>",
    "confidence_calibration": "<overconfident | calibrated | underconfident>"
}}"""

    resp = CLIENT.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "main_score": 0, "rapid_score": 0, "score_total": 0,
            "missed_concepts": [], "model_answer": "Grading failed.",
            "one_win": "—", "one_miss": "—",
            "confidence_calibration": "unknown",
        }


# —— Player model ————————————————————————————————————————————————
def load_player_model(conn) -> dict:
    row = conn.execute("SELECT * FROM player_model WHERE id = 1").fetchone()
    return {
        "total_sessions": row["total_sessions"],
        "streak_days": row["streak_days"],
        "last_session": row["last_session"],
        "avg_score_7d": row["avg_score_7d"],
        "avg_confidence": row["avg_confidence"],
        "confidence_bias": row["confidence_bias"],
        "weak_topics": json.loads(row["weak_topics"]),
        "strong_topics": json.loads(row["strong_topics"]),
        "common_errors": json.loads(row["common_errors"]),
    }


def update_player_model(conn, grade: dict, tags: list[str], confidence: int):
    pm = load_player_model(conn)
    today = date.today().isoformat()
    yesterday = date.fromordinal(date.today().toordinal() - 1).isoformat()

    # Streak
    if pm["last_session"] == today:
        pass  # already drilled today
    elif pm["last_session"] == yesterday:
        pm["streak_days"] += 1
    else:
        pm["streak_days"] = 1

    pm["total_sessions"] += 1
    pm["last_session"] = today

    # 7-day average score
    rows = conn.execute(
        "SELECT score_total FROM sessions WHERE date >= date('now', '-7 days')"
    ).fetchall()
    scores = [r["score_total"] for r in rows] + [grade["score_total"]]
    pm["avg_score_7d"] = round(sum(scores) / len(scores), 2)

    # Confidence tracking
    conf_rows = conn.execute(
        "SELECT confidence FROM sessions WHERE date >= date('now', '-7 days') AND confidence IS NOT NULL"
    ).fetchall()
    confs = [r["confidence"] for r in conf_rows] + [confidence]
    pm["avg_confidence"] = round(sum(confs) / len(confs), 2)
    pm["confidence_bias"] = grade.get("confidence_calibration", "unknown")

    # Topic weights — decay toward 0.5 baseline, shift based on score
    score_ratio = grade["score_total"] / 10.0
    for tag in TOPIC_TAGS:
        current = pm["weak_topics"].get(tag, 0.5)
        if tag in tags:
            new = current + (score_ratio - 0.5) * 0.3
        else:
            new = current + (0.5 - current) * 0.05
        pm["weak_topics"][tag] = round(max(0.0, min(1.0, new)), 3)

    pm["strong_topics"] = {k: v for k, v in pm["weak_topics"].items() if v >= 0.65}

    # Common errors (keep last 20)
    errors = pm["common_errors"]
    errors.extend(grade.get("missed_concepts", []))
    pm["common_errors"] = errors[-20:]

    conn.execute("""
        UPDATE player_model SET
            total_sessions = ?, streak_days = ?, last_session = ?,
            avg_score_7d = ?, avg_confidence = ?, confidence_bias = ?,
            weak_topics = ?, strong_topics = ?, common_errors = ?,
            updated_at = datetime('now')
        WHERE id = 1
    """, (
        pm["total_sessions"], pm["streak_days"], pm["last_session"],
        pm["avg_score_7d"], pm["avg_confidence"], pm["confidence_bias"],
        json.dumps(pm["weak_topics"]), json.dumps(pm["strong_topics"]),
        json.dumps(pm["common_errors"]),
    ))
    conn.commit()
    return pm


def export_player_model(conn):
    pm = load_player_model(conn)
    pm["exported_at"] = datetime.now().isoformat()
    with open(PLAYER_MODEL_PATH, "w") as f:
        json.dump(pm, f, indent=2)
    return pm


# —— Git auto-commit ——————————————————————————————————————————————
def git_commit_player_model():
    repo_dir = os.path.dirname(__file__)
    try:
        subprocess.run(["git", "add", "player_model.json", "lng_trainer.db"],
                       cwd=repo_dir, check=True, capture_output=True)
        msg = f"trainer: update player model ({date.today().isoformat()})"
        subprocess.run(["git", "commit", "-m", msg],
                       cwd=repo_dir, check=True, capture_output=True)
        print(f"\n  Git: committed ({msg})")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        if "nothing to commit" in stderr:
            print("\n  Git: nothing new to commit")
        else:
            print(f"\n  Git: commit failed — {stderr}")


# —— Feedback display ————————————————————————————————————————————
def show_feedback(grade: dict, pm: dict, prev_score: float | None):
    delta = ""
    if prev_score is not None:
        diff = grade["score_total"] - prev_score
        delta = f" ({'+' if diff >= 0 else ''}{diff:.1f})"

    weak_list = sorted(
        [(k, v) for k, v in pm["weak_topics"].items() if v < 0.5],
        key=lambda x: x[1]
    )[:3]

    print("\n" + "=" * 60)
    print("  FEEDBACK")
    print("=" * 60)
    print(f"  Score:      {grade['score_total']:.1f} / 10{delta}")
    print(f"  Streak:     {pm['streak_days']} day{'s' if pm['streak_days'] != 1 else ''}")
    print(f"  7d avg:     {pm['avg_score_7d']:.1f}")
    print(f"  Confidence: {grade.get('confidence_calibration', '—')}")
    print(f"\n  ✓ Win:  {grade['one_win']}")
    print(f"  ✗ Miss: {grade['one_miss']}")
    print(f"\n  Model answer:")
    print(f"    {grade['model_answer']}")
    if weak_list:
        print(f"\n  Weak spots to revisit:")
        for tag, weight in weak_list:
            print(f"    • {tag.replace('_', ' ')} ({weight:.2f})")
    print("=" * 60)


# —— Article selection ————————————————————————————————————————————
def select_best_article(articles: list[dict], weak_topics: dict) -> dict | None:
    if not articles:
        return None

    scored = []
    for art in articles:
        tags = tag_article(art["title"], art["text"])
        art["tags"] = tags
        relevance = sum(1.0 - weak_topics.get(t, 0.5) for t in tags)
        scored.append((relevance, art))
        time.sleep(0.3)

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# —— Main session ————————————————————————————————————————————————
def run_session():
    print("\n" + "=" * 60)
    print("  LNG DAILY TRAINER — V1")
    print(f"  {date.today().strftime('%A %d %B %Y')}")
    print("=" * 60)

    conn = init_db()
    pm = load_player_model(conn)

    prev = conn.execute(
        "SELECT score_total FROM sessions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    prev_score = prev["score_total"] if prev else None

    # Fetch articles
    print("\nFetching LNG articles...")
    rss_articles = fetch_rss_articles()
    enriched = []
    for art in rss_articles:
        text = fetch_full_article(art["link"])
        if text:
            art["text"] = text[:2000]
            enriched.append(art)
        time.sleep(0.5)

    print(f"  {len(enriched)} articles extracted.")

    if not enriched:
        print("  No articles available. Try again later.")
        conn.close()
        return

    # Select article weighted toward weak topics
    print("  Selecting best article for your weak spots...")
    chosen = select_best_article(enriched, pm["weak_topics"])
    if not chosen:
        chosen = enriched[0]
        chosen["tags"] = ["headline_interpretation"]

    tags = chosen.get("tags", [])
    print(f"\n  Article: {chosen['title']}")
    print(f"  Tags:    {', '.join(tags)}")
    print(f"  Link:    {chosen['link']}")

    # Generate questions
    print("\n  Generating questions...\n")
    questions = generate_questions(chosen["title"], chosen["text"], pm["weak_topics"])

    print("-" * 60)
    print(f"  MAIN Q:  {questions['main']}")
    print(f"  RAPID 1: {questions['r1']}")
    print(f"  RAPID 2: {questions['r2']}")
    print("-" * 60)

    # Collect answers
    print("\nYour answers (type your response, press Enter):\n")
    ans_main = input("  MAIN:    ").strip()
    ans_r1 = input("  RAPID 1: ").strip()
    ans_r2 = input("  RAPID 2: ").strip()

    while True:
        try:
            confidence = int(input("  CONFIDENCE (1-5): ").strip())
            if 1 <= confidence <= 5:
                break
        except ValueError:
            pass
        print("  Enter a number 1-5.")

    answers = {"main": ans_main, "r1": ans_r1, "r2": ans_r2}

    # Grade
    print("\n  Grading...")
    grade = grade_answers(chosen["title"], chosen["text"], questions, answers, confidence)

    # Store session
    conn.execute("""
        INSERT INTO sessions (
            date, article_title, article_link, topic_tags,
            main_question, rapid_1, rapid_2,
            user_main, user_r1, user_r2, confidence,
            score_total, main_score, rapid_score,
            missed_concepts, model_answer, feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        date.today().isoformat(),
        chosen["title"], chosen["link"], json.dumps(tags),
        questions["main"], questions["r1"], questions["r2"],
        ans_main, ans_r1, ans_r2, confidence,
        grade["score_total"], grade["main_score"], grade["rapid_score"],
        json.dumps(grade.get("missed_concepts", [])),
        grade.get("model_answer", ""),
        json.dumps({"win": grade["one_win"], "miss": grade["one_miss"]}),
    ))
    conn.commit()

    # Update player model
    pm = update_player_model(conn, grade, tags, confidence)

    # Show feedback
    show_feedback(grade, pm, prev_score)

    # Export and git commit
    export_player_model(conn)
    git_commit_player_model()

    conn.close()
    print("\nDone. See you tomorrow.\n")


if __name__ == "__main__":
    run_session()
