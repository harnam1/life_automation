"""
London Weekend Curator — v3
Three-pass pipeline: Discovery → Verify+Rewrite → Link Validation → Send
Runs every Saturday evening via GitHub Actions.

Changes from v2:
- Dropped SPORTS category entirely
- Restructured categories: FOOD, CULTURE, NEW_OPENINGS, WEIRD, FREEBIES
- Overhauled source list — banned mainstream aggregators, prioritised niche
- Tighter prompts to reduce token waste
- Verify+rewrite prompt personalised and sharpened
- Token usage logging added
"""

import os
import json
import re
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus

import anthropic
import requests


CLIENT = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

# Track token usage across all API calls
USAGE_LOG: list[dict] = []


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def get_next_weekend_dates() -> tuple[str, str, str]:
    """Returns (friday, saturday, sunday) date strings for the upcoming weekend."""
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    if today.weekday() == 5:
        days_until_friday = 6

    next_friday = today + timedelta(days=days_until_friday)
    next_saturday = next_friday + timedelta(days=1)
    next_sunday = next_friday + timedelta(days=2)

    return (
        next_friday.strftime("%A %d %B %Y"),
        next_saturday.strftime("%A %d %B %Y"),
        next_sunday.strftime("%A %d %B %Y"),
    )


# ---------------------------------------------------------------------------
# Claude API helpers
# ---------------------------------------------------------------------------

def _extract_text(response) -> str:
    return "\n".join(block.text for block in response.content if block.type == "text")


def _log_usage(label: str, response):
    """Log token usage from an API response."""
    usage = response.usage
    web_searches = getattr(usage, "server_tool_use", None)
    search_count = 0
    if web_searches and hasattr(web_searches, "web_search_requests"):
        search_count = web_searches.web_search_requests

    entry = {
        "step": label,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "web_searches": search_count,
    }
    USAGE_LOG.append(entry)

    total_in = entry["input_tokens"] + entry["cache_read"] + entry["cache_write"]
    print(
        f"  [{label}] in={total_in:,} out={entry['output_tokens']:,} "
        f"searches={search_count}"
    )


def _create_message(label: str = "unknown", **kwargs):
    """Wrapper with retry, rate-limit handling, and usage logging."""
    for attempt in range(3):
        try:
            response = CLIENT.messages.create(**kwargs)
            _log_usage(label, response)
            return response
        except anthropic.RateLimitError:
            if attempt == 2:
                raise
            wait = 65 * (attempt + 1)
            print(f"Rate limit hit — waiting {wait}s (attempt {attempt + 1}/3)...")
            time.sleep(wait)


def _parse_json_response(raw: str) -> list[dict]:
    """Extract and parse a JSON array from Claude's response."""
    def extract_json_array(text: str) -> list[dict]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        start = text.find("[")
        if start == -1:
            raise ValueError("No JSON array found in response")
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
        raise ValueError("Unterminated JSON array in response")

    try:
        return extract_json_array(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    print("JSON parse failed — waiting 65s before retry...")
    time.sleep(65)

    raw_truncated = raw[:8000] if len(raw) > 8000 else raw
    fix_response = _create_message(
        label="json_fix",
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": (
                    "Fix this into a valid JSON array. Return ONLY the array, nothing else.\n\n"
                    f"{raw_truncated}"
                ),
            },
        ],
    )
    fixed = _extract_text(fix_response)
    return extract_json_array(fixed)


# ---------------------------------------------------------------------------
# Pass 1: Discovery
# ---------------------------------------------------------------------------

DISCOVERY_PROMPT = """Find 15 things to do in London on {fri}, {sat}, and {sun}.

You are writing for a 23-year-old Canary Wharf resident who has lived in London
for years. He does NOT want: tourist attractions, "best of London" listicle
staples, overpriced "experience" packages, anything designed for Instagram
influencers, or generic chain restaurant recommendations. He wants things a
well-connected local would know about.

Categories — find roughly 3-4 per category:

FOOD: Pop-up kitchens, supper clubs, new restaurant soft launches, street food
      collabs, tasting menus under £50, market stalls worth queuing for, wine
      bars doing interesting by-the-glass lists, brewery taproom events. NOT
      established restaurants just being open as normal.

CULTURE: Gallery openings/private views, new exhibitions launching that week,
         fringe theatre, comedy clubs (not West End), independent cinema
         screenings, book launches, gigs at small venues (<500 cap), DJ nights
         at interesting spaces, spoken word, poetry slams, talks/panels.

NEW_OPENINGS: Anything that opened in the last 4 weeks OR is launching this
              weekend. Bars, restaurants, shops, co-working spaces, studios,
              markets. The newer the better.

WEIRD: One-off events, niche meetups, unusual workshops, late-night museum
       events, immersive stuff that isn't mainstream, supper clubs in strange
       locations, guerrilla cinema, foraging walks, competitive events
       (chess, ping pong tournaments, pub quizzes with a twist). Things you'd
       screenshot and send to a group chat.

FREEBIES: Free exhibitions, open studios, free gigs, free comedy nights,
          gallery openings with free drinks, community events, outdoor
          screenings, free workshops, brand launch parties that are actually
          open to public. London residents shouldn't have to pay for everything.

{extra_instruction}

SOURCE RULES — this is critical:

MUST USE these sources (search them directly):
- Eater London (eater.com/london) — new openings, pop-ups, restaurant news
- Hot Dinners (hot-dinners.com) — London restaurant openings and pop-ups
- Infatuation London — honest restaurant reviews
- Resident Advisor (ra.co) — music, club nights, DJ events
- Dice.fm — gigs, comedy, cultural events with actual dates
- Dazed, i-D, Another Magazine — culture picks
- Londonist (londonist.com) — weird London, offbeat events
- Design My Night — bar openings, event listings with dates
- Skiddle — gigs and club nights
- Venue sites directly: Barbican, Southbank, ICA, Serpentine, Whitechapel
  Gallery, Corsica Studios, Village Underground, Omeara, EartH

NEVER USE these sources:
- Time Out London (timeout.com) — generic, obvious, SEO-optimised
- TripAdvisor — tourist-oriented
- Yelp — irrelevant for events
- Viator / GetYourGuide — tourist experiences
- Generic "top 10 things to do" blog posts
- London Theatre Direct or similar for West End shows

If you find yourself recommending the Tower of London, a West End musical,
afternoon tea, or a Thames river cruise, you have failed the assignment.

Return ONLY a JSON array, no markdown, no preamble:
[{{"name":"...","category":"FOOD|CULTURE|NEW_OPENINGS|WEIRD|FREEBIES","dates":["..."],"location":"...","description":"...","url":"...","price":"..."}}]
"""


def discover_events(fri: str, sat: str, sun: str, extra_instruction: str = "") -> list[dict]:
    prompt = DISCOVERY_PROMPT.format(
        fri=fri, sat=sat, sun=sun,
        extra_instruction=extra_instruction,
    )
    response = _create_message(
        label="discovery",
        model=MODEL,
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 10}],
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _extract_text(response)
    return _parse_json_response(raw)


# ---------------------------------------------------------------------------
# Pass 2: Verify + Rewrite
# ---------------------------------------------------------------------------

VERIFY_PROMPT = """You have {n} candidate London events for {fri} to {sun}.

{candidates_json}

For EACH event do three things:

1. DATE CHECK — search to confirm it's genuinely on {fri}, {sat}, or {sun}.
   - CONFIRMED: found direct evidence (event page, ticket link, venue calendar)
   - ONGOING: permanent/long-running thing currently open (exhibition, restaurant)
   - FAILED: already happened, wrong weekend, cancelled, no evidence it exists

2. REWRITE — for non-FAILED events, completely rewrite the description:
   - You're texting a mate, not writing listings copy
   - Be specific: "their smash burgers are up there with Buns From Home and
     they pour natural wine" NOT "a new burger restaurant with a great wine list"
   - Include practical tips: "book ahead, it's tiny" / "walk-ins only, go before 6"
   - Mention travel from Canary Wharf where useful: "10 mins on the Jubilee line"
   - 2 sentences max. No fluff.
   - BANNED WORDS: hidden gem, vibrant, bustling, iconic, unmissable, curated,
     artisanal, bespoke, eclectic, up-and-coming, trendy, must-visit, foodie

3. LINK UPGRADE — if the URL points to timeout.com, a generic eventbrite browse
   page, tripadvisor, or any aggregator homepage, replace it with:
   - The venue's own event page or booking link
   - A direct Dice.fm / FIXR / Eventbrite EVENT page (not browse)
   - The venue/event Instagram post
   - An Eater London or Hot Dinners article about it

Return JSON array with ALL events (including FAILED):
[{{"name":"...","category":"...","status":"CONFIRMED|ONGOING|FAILED","dates":["..."],"location":"...","description":"...","url":"...","price":"...","verification_note":"..."}}]

ONLY the JSON array. No markdown, no preamble.
"""


def verify_and_rewrite(candidates: list[dict], fri: str, sat: str, sun: str) -> list[dict]:
    candidates_json = json.dumps(candidates, indent=2)
    prompt = VERIFY_PROMPT.format(
        n=len(candidates),
        fri=fri, sat=sat, sun=sun,
        candidates_json=candidates_json,
    )
    response = _create_message(
        label="verify_rewrite",
        model=MODEL,
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 15}],
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _extract_text(response)
    return _parse_json_response(raw)


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def backfill(confirmed: list[dict], fri: str, sat: str, sun: str) -> list[dict]:
    needed = 12 - len(confirmed)
    if needed <= 0:
        return confirmed

    all_cats = ["FOOD", "CULTURE", "NEW_OPENINGS", "WEIRD", "FREEBIES"]
    category_counts: dict[str, int] = {}
    for e in confirmed:
        cat = e.get("category", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    gaps = []
    for cat in all_cats:
        have = category_counts.get(cat, 0)
        if have < 2:
            gaps.append(f"{2 - have} more {cat}")

    gap_desc = ", ".join(gaps) if gaps else f"{needed} events across any category"
    existing_names = ", ".join(e["name"] for e in confirmed)
    extra = (
        f"I need {gap_desc} for {fri} to {sun}. "
        f"Use date-specific queries like 'london events {fri}'. "
        f"Do NOT repeat: {existing_names}"
    )

    print(f"  Backfill: requesting {needed} events ({gap_desc})")
    new_candidates = discover_events(fri, sat, sun, extra_instruction=extra)
    time.sleep(65)
    new_verified = verify_and_rewrite(new_candidates, fri, sat, sun)
    new_confirmed = [e for e in new_verified if e.get("status") in ("CONFIRMED", "ONGOING")]

    merged = confirmed + new_confirmed
    if len(merged) < 10:
        print(f"  Warning: only {len(merged)} events after backfill — shipping anyway")
    return merged


# ---------------------------------------------------------------------------
# Pass 3: Link validation
# ---------------------------------------------------------------------------

def _google_fallback(event_name: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(event_name + ' london')}"


def validate_links(events: list[dict]) -> list[dict]:
    """HTTP HEAD check every URL; replace broken links with Google fallback."""

    def check_url(event: dict) -> dict:
        url = event.get("url", "").strip()
        if not url:
            event["url"] = _google_fallback(event["name"])
            event["link_status"] = "NO_URL"
            return event

        headers = {"User-Agent": "Mozilla/5.0 (compatible; WeekendCurator/1.0)"}
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True, headers=headers)
            if resp.status_code < 400:
                event["link_status"] = "OK"
                return event
            resp = requests.get(url, timeout=10, allow_redirects=True, headers=headers, stream=True)
            if resp.status_code < 400:
                event["link_status"] = "OK"
            else:
                event["url"] = _google_fallback(event["name"])
                event["link_status"] = f"REPLACED ({resp.status_code})"
        except requests.RequestException:
            event["url"] = _google_fallback(event["name"])
            event["link_status"] = "REPLACED (timeout/error)"

        return event

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(check_url, e): e for e in events}
        results = [future.result() for future in as_completed(futures)]

    ok = sum(1 for e in results if e.get("link_status") == "OK")
    replaced = sum(1 for e in results if "REPLACED" in e.get("link_status", ""))
    print(f"LINKS: {ok} valid, {replaced} replaced with Google fallback")

    return results


# ---------------------------------------------------------------------------
# Format email
# ---------------------------------------------------------------------------

FORMAT_PROMPT = """Format these London weekend events as HTML email content.

Weekend: {fri} to {sun}

{events_json}

Rules:
- Dark header (#1a1a1a bg, white text) with weekend dates
- Group by category with headers: 🍴 FOOD, 🎭 CULTURE, 🆕 NEW OPENINGS, 🔮 WEIRD, 🆓 FREEBIES
- Each event: name (bold), location + date + price on one line, description,
  clickable "→ More info" link (#2563eb)
- Inline CSS only, max-width 640px, Gmail-safe
- No <html>/<head>/<body> tags
- ONLY return HTML, nothing else
"""


def format_email(events: list[dict], fri: str, sun: str) -> str:
    if not events:
        inner_html = f"""
        <div style="background:#1a1a1a;color:#fff;padding:32px;text-align:center;border-radius:8px;">
            <h1 style="margin:0;">Your London Weekend</h1>
            <p style="color:#aaa;margin:8px 0 0;">{fri} &mdash; {sun}</p>
        </div>
        <p style="margin-top:32px;">Slim pickings this weekend — couldn't verify enough.
        Check back next Saturday.</p>
        """
    else:
        events_json = json.dumps(events, indent=2)
        prompt = FORMAT_PROMPT.format(fri=fri, sun=sun, events_json=events_json)
        response = _create_message(
            label="format_email",
            model=MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        inner_html = _extract_text(response).strip()
        if inner_html.startswith("```"):
            inner_html = inner_html[inner_html.index("\n") + 1:]
        if inner_html.endswith("```"):
            inner_html = inner_html[: inner_html.rfind("```")]

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.6;color:#1a1a1a;max-width:640px;margin:0 auto;padding:20px;background:#f8f8f8;">
    {inner_html.strip()}
    <hr style="border:none;border-top:1px solid #e5e5e5;margin:32px 0 16px;">
    <p style="font-size:12px;color:#999;text-align:center;">
        Auto-generated by Weekend Curator · Powered by Claude
    </p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Email sending
# ---------------------------------------------------------------------------

def send_email(html_content: str, fri: str, sun: str):
    sender = os.environ["GMAIL_ADDRESS"]
    password = os.environ["GMAIL_APP_PASSWORD"]
    recipient_str = os.environ.get("RECIPIENT_EMAILS", "").strip()
    recipients = [r.strip() for r in recipient_str.split(",")] if recipient_str else [sender]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Your London Weekend — {fri} to {sun}"
    msg["From"] = f"Weekend Curator <{sender}>"
    msg["To"] = ", ".join(recipients)

    plain = f"Your London weekend picks for {fri} to {sun}. View in HTML for the full experience."
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())

    print(f"Email sent to {', '.join(recipients)}")


# ---------------------------------------------------------------------------
# Usage summary
# ---------------------------------------------------------------------------

def print_usage_summary():
    """Print a breakdown of token usage and estimated cost."""
    print("\n" + "=" * 60)
    print("TOKEN USAGE SUMMARY")
    print("=" * 60)

    total_in = 0
    total_out = 0
    total_searches = 0

    for entry in USAGE_LOG:
        step_in = entry["input_tokens"] + entry["cache_read"] + entry["cache_write"]
        total_in += step_in
        total_out += entry["output_tokens"]
        total_searches += entry["web_searches"]
        print(
            f"  {entry['step']:20s}  "
            f"in={step_in:>7,}  out={entry['output_tokens']:>6,}  "
            f"searches={entry['web_searches']}"
        )

    # Haiku 4.5 pricing: $1/MTok input, $5/MTok output, $0.01/search
    cost_in = (total_in / 1_000_000) * 1.0
    cost_out = (total_out / 1_000_000) * 5.0
    cost_search = total_searches * 0.01
    cost_total = cost_in + cost_out + cost_search

    print("-" * 60)
    print(f"  {'TOTAL':20s}  in={total_in:>7,}  out={total_out:>6,}  searches={total_searches}")
    print()
    print(f"  Input tokens:   ${cost_in:.4f}")
    print(f"  Output tokens:  ${cost_out:.4f}")
    print(f"  Web searches:   ${cost_search:.2f}")
    print(f"  TOTAL COST:     ${cost_total:.4f}")
    print(f"  Monthly (4 runs): ~${cost_total * 4:.2f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    fri, sat, sun = get_next_weekend_dates()
    print(f"Curating weekend: {fri} — {sun}")

    # Pass 1: Discovery
    print("\nPass 1: Discovering events...")
    candidates = discover_events(fri, sat, sun)
    print(f"Found {len(candidates)} candidates")

    print("Waiting 65s between passes...")
    time.sleep(65)

    # Pass 2: Verify + Rewrite
    print("\nPass 2: Verifying dates and rewriting...")
    verified = verify_and_rewrite(candidates, fri, sat, sun)
    confirmed = [e for e in verified if e.get("status") in ("CONFIRMED", "ONGOING")]
    failed = [e for e in verified if e.get("status") == "FAILED"]
    print(f"Verified: {len(confirmed)} passed, {len(failed)} dropped")
    for f in failed:
        print(f"  DROPPED: {f['name']} — {f.get('verification_note', 'no reason')}")

    # Backfill if needed
    if len(confirmed) < 12:
        print(f"\nBackfilling: need {12 - len(confirmed)} more events...")
        time.sleep(65)
        confirmed = backfill(confirmed, fri, sat, sun)

    # Edge case
    if not confirmed:
        print("All events failed — sending apology email")
        html = format_email([], fri, sun)
        send_email(html, fri, sun)
        print_usage_summary()
        return

    # Pass 3: Link validation
    print("\nPass 3: Validating links...")
    validated = validate_links(confirmed)

    # Format and send
    print("\nWaiting 65s before formatting...")
    time.sleep(65)
    print("Formatting email...")
    html = format_email(validated, fri, sun)
    send_email(html, fri, sun)

    print(f"\nDone! Sent {len(validated)} events")
    print(f"  CONFIRMED: {sum(1 for e in validated if e.get('status') == 'CONFIRMED')}")
    print(f"  ONGOING:   {sum(1 for e in validated if e.get('status') == 'ONGOING')}")
    link_ok = sum(1 for e in validated if e.get("link_status") == "OK")
    print(f"  Links OK:  {link_ok}/{len(validated)}")

    print_usage_summary()


if __name__ == "__main__":
    main()
