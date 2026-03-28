"""
London Weekend Curator — v2
Three-pass pipeline: Discovery → Verify+Rewrite → Link Validation → Send
Runs every Saturday evening via GitHub Actions.
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


def _create_message(**kwargs):
    """Wrapper around CLIENT.messages.create with automatic 429 retry (up to 3 attempts)."""
    for attempt in range(3):
        try:
            return CLIENT.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt == 2:
                raise
            wait = 65 * (attempt + 1)
            print(f"Rate limit hit — waiting {wait}s before retry (attempt {attempt + 1}/3)...")
            time.sleep(wait)


def _parse_json_response(raw: str, original_prompt: str = "") -> list[dict]:  # noqa: ARG001
    """Extract and parse a JSON array from Claude's response.

    Handles: raw JSON, markdown-fenced JSON, preamble text before the array.
    Retries once (after a rate-limit cooldown) if extraction fails.
    """
    def extract_json_array(text: str) -> list[dict]:
        """Find the first [...] array in text, regardless of surrounding content."""
        # Try direct parse first (clean JSON with no wrapper)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip a single markdown code fence if present (```json ... ``` or ``` ... ```)
        fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        # Extract the first top-level JSON array using bracket matching
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

    # Wait for the rate-limit window to reset before retrying
    print("JSON parse failed — waiting 65s for rate limit window to reset before retry...")
    time.sleep(65)

    # Truncate raw to ~8000 chars to keep input tokens well within limits
    raw_truncated = raw[:8000] if len(raw) > 8000 else raw

    # Retry: ask Claude to fix its own JSON (minimal prompt to avoid token limits)
    fix_response = _create_message(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": (
                    "The following text should be a JSON array but is not valid JSON. "
                    "Fix it and return ONLY the JSON array — no markdown, no preamble, no backticks.\n\n"
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

def discover_events(fri: str, sat: str, sun: str, extra_instruction: str = "") -> list[dict]:
    """Call Claude with web search to find candidate events. Returns raw JSON list."""
    prompt = f"""You are a London weekend curator for someone who lives in Canary Wharf.

Find exactly 15 things to do in London for the weekend of {fri} to {sun}.

The selections MUST cover these categories (aim for roughly 3 each):

- SPORTS: Live sports events, pick-up games, interesting fitness events, spectator sport
- FOOD: New restaurant openings, pop-ups, food markets, supper clubs, notable dining
- CULTURE: Exhibitions, theatre, gigs, comedy, film, talks, literary events
- NEW_OPENINGS: Bars, venues, shops, spaces that recently opened or are launching that weekend
- WEIRD: Unusual, quirky, surprising, niche, or one-off events — the stranger the better

{extra_instruction}

Prioritise niche and specialist sources over generic listings aggregators.
Preferred sources by category:

SPORTS: London Sport, TimeOut Sport, club/venue sites directly, parkrun pages,
        British Tennis, England Athletics, London Marathon Events, community
        league sites, MeetUp groups

FOOD: Eater London, Hot Dinners, Infatuation London, London Eater Instagram
      accounts, individual restaurant Instagram/sites, Feast It, KERB,
      Street Feast, Maltby Street Market site, Borough Market calendar

CULTURE: Artsy, Frieze, Barbican/Southbank/BFI/ICA calendars directly,
         Resident Advisor (music), Dice.fm, Songkick, Dazed, Another Magazine,
         London Review of Books events, Serpentine site, White Cube/Gagosian
         sites, Curzon/Prince Charles Cinema listings

NEW_OPENINGS: Eater London openings tracker, Hot Dinners, Costar/proptech
              press releases, individual venue Instagram announcements,
              Evening Standard Going Out, Wallpaper* city guide

WEIRD: Atlas Obscura London, Londonist, Niche London (newsletter), Reddit
       r/london, Obscura Magazine, Museum of the Mind events, Viktor Wynd
       Museum, London Fortean Society, unusual meetup groups

Avoid relying heavily on: Time Out "best of" listicles, generic Eventbrite
browse pages, TripAdvisor, Yelp, or SEO-optimised "top 10" blog posts.
These produce generic, obvious recommendations.

Return ONLY a JSON array in this exact format. No markdown, no preamble, no backticks:

[
  {{
    "name": "Event Name",
    "category": "SPORTS",
    "dates": ["Friday 4 April 2025"],
    "location": "Peckham",
    "description": "Why it's worth going — 1-2 sentences",
    "url": "https://...",
    "price": "£15"
  }}
]
"""

    response = _create_message(
        model=MODEL,
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 10}],
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _extract_text(response)
    return _parse_json_response(raw, prompt)


# ---------------------------------------------------------------------------
# Pass 2: Verify + Rewrite
# ---------------------------------------------------------------------------

def verify_and_rewrite(candidates: list[dict], fri: str, sat: str, sun: str) -> list[dict]:
    """Second Claude call: fact-check dates, rewrite copy, improve links."""
    candidates_json = json.dumps(candidates, indent=2)
    n = len(candidates)

    prompt = f"""You are a fact-checker and rewriter for a London weekend newsletter aimed at
a 23-year-old guy living in Canary Wharf. He's into powerlifting, tennis,
squash, Bhangra, good food (not fine dining wank), contemporary art, live
music, comedy, and anything genuinely strange or one-off. He's not interested
in generic tourist stuff, overpriced "experiences", or anything that feels
like it was designed for Instagram influencers. He goes out with his
girlfriend and with mates — so both date-worthy and group-friendly picks
are good.

You have {n} candidate events for the weekend of {fri} to {sun}.

{candidates_json}

For EACH event, do the following:

## Step 1: Date Verification

Search for the event to confirm it is genuinely happening on one of the
target dates.

Classify as:
- CONFIRMED: Direct evidence (event page, ticket listing, venue calendar)
  confirming the event falls on {fri}, {sat}, or {sun}
- ONGOING: Permanent or long-running attraction (exhibition, restaurant,
  bar) that is currently open — no specific date needed
- FAILED: Event already happened, is on a different weekend, has been
  cancelled, venue has closed, recurring series has ended, or you cannot
  find any evidence it exists

If FAILED, briefly note why and move on. Do not attempt to salvage it.

## Step 2: Rewrite the Description

For events that pass verification, rewrite the description completely.
Rules:
- Write like a mate recommending something over a pint, not a listings
  magazine. First person observations, casual language, occasional swearing
  is fine.
- Be specific about WHY it's good — don't just describe what it is.
  "Their carbonara is filthy good and they do half-price negronis before 7"
  beats "An Italian restaurant offering classic dishes."
- If you know something non-obvious (e.g. "get there early, it's first
  come first served" or "the support act is actually better than the
  headliner"), include it.
- Keep it to 2-3 sentences max.
- Never use: "hidden gem", "vibrant", "bustling", "iconic", "unmissable",
  "curated", "artisanal", "bespoke". These words are banned.
- Reference the person's proximity to Canary Wharf where useful (e.g.
  "15 mins on the Jubilee line" or "walkable from yours").

## Step 3: Find the Best Link

If the original URL is a generic aggregator page (e.g. timeout.com/london,
eventbrite.co.uk browse page, tripadvisor listing), replace it with a more
direct source:
- The venue's own website or event page
- A direct ticket purchase link (Dice, FIXR, Eventbrite event page)
- The venue/event's Instagram post announcing it
- A specific article from a niche source (Eater London, Hot Dinners, RA)

The ideal link is one where the reader lands and can immediately see what
the event is + how to go/book. Not a homepage. Not a search results page.

## Output

Return a JSON array of ALL events (including FAILED ones):

[
  {{
    "name": "Event Name",
    "category": "SPORTS",
    "status": "CONFIRMED",
    "dates": ["Saturday 5 April 2025"],
    "location": "Peckham",
    "description": "Rewritten description in the voice described above",
    "url": "https://direct-link.com/event",
    "price": "£15",
    "verification_note": "Found on venue calendar, tickets on Dice"
  }}
]

Include FAILED events with status "FAILED" and a brief verification_note,
but leave other fields as-is.

Return ONLY the JSON array. No markdown, no preamble, no backticks.
"""

    response = _create_message(
        model=MODEL,
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 15}],
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _extract_text(response)
    return _parse_json_response(raw, prompt)


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def backfill(confirmed: list[dict], fri: str, sat: str, sun: str) -> list[dict]:
    """Top up to 12 events if verification dropped too many."""
    needed = 12 - len(confirmed)
    if needed <= 0:
        return confirmed

    all_cats = ["SPORTS", "FOOD", "CULTURE", "NEW_OPENINGS", "WEIRD"]
    category_counts: dict[str, int] = {}
    for e in confirmed:
        cat = e.get("category", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    gaps = []
    for cat in all_cats:
        have = category_counts.get(cat, 0)
        if have < 3:
            gaps.append(f"{3 - have} more {cat}")

    gap_desc = ", ".join(gaps) if gaps else f"{needed} events across any category"
    existing_names = ", ".join(e["name"] for e in confirmed)
    extra = (
        f"I need {gap_desc} for the weekend of {fri} to {sun}. "
        f"Use date-specific search queries like 'london {fri}' to find real events. "
        f"Do NOT suggest events already in this list: {existing_names}"
    )

    print(f"  Backfill: requesting {needed} events ({gap_desc})")
    new_candidates = discover_events(fri, sat, sun, extra_instruction=extra)
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
    """HTTP HEAD check every URL; replace broken links with Google search fallback."""

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
            # HEAD rejected — try GET without downloading body
            resp = requests.get(url, timeout=10, allow_redirects=True, headers=headers, stream=True)
            if resp.status_code < 400:
                event["link_status"] = "OK"
            else:
                event["url"] = _google_fallback(event["name"])
                event["link_status"] = f"REPLACED (was {resp.status_code})"
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
# Format email (Claude formats the verified JSON into HTML)
# ---------------------------------------------------------------------------

def format_email(events: list[dict], fri: str, sun: str) -> str:
    """Ask Claude to format the verified events as a styled HTML email body."""
    if not events:
        inner_html = f"""
        <div style="background:#1a1a1a;color:#fff;padding:32px;text-align:center;border-radius:8px;">
            <h1 style="margin:0;">Your London Weekend</h1>
            <p style="color:#aaa;margin:8px 0 0;">{fri} &mdash; {sun}</p>
        </div>
        <p style="margin-top:32px;">Slim pickings this weekend — couldn't verify enough events in time.
        Check back next Saturday.</p>
        """
    else:
        events_json = json.dumps(events, indent=2)
        prompt = f"""Format the following verified London weekend events as a clean HTML email body.

Weekend dates: {fri} to {sun}

Events:
{events_json}

Requirements:
- Dark header banner (#1a1a1a background, white text) with the weekend dates
- Group events by category with emoji headers: SPORTS, FOOD, CULTURE, NEW_OPENINGS, WEIRD
- For each event show: name (bold), location, date(s), price, description, and a
  clickable "Book / Info →" anchor using the url field (color: #2563eb)
- Tone: punchy, no waffle
- Simple inline styles, max-width 640px, renders in Gmail
- Do NOT include <html>, <head>, or <body> tags — just the inner content
- Return ONLY the HTML. No markdown, no preamble, no backticks.
"""
        response = _create_message(
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
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1a1a1a; max-width: 640px; margin: 0 auto; padding: 20px; background: #f8f8f8;">
    {inner_html.strip()}
    <hr style="border: none; border-top: 1px solid #e5e5e5; margin: 32px 0 16px;">
    <p style="font-size: 12px; color: #999; text-align: center;">
        Auto-generated by Weekend Curator · Powered by Claude
    </p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Email sending
# ---------------------------------------------------------------------------

def send_email(html_content: str, fri: str, sun: str):
    """Send the curated email via Gmail SMTP."""
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
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    fri, sat, sun = get_next_weekend_dates()
    print(f"Curating weekend: {fri} — {sun}")

    # Pass 1: Discovery
    print("Pass 1: Discovering events...")
    candidates = discover_events(fri, sat, sun)
    print(f"Found {len(candidates)} candidates")

    # Wait for rate-limit window to reset between heavy web-search passes
    print("Waiting 65s between passes to avoid rate limits...")
    time.sleep(65)

    # Pass 2: Verify + Rewrite
    print("Pass 2: Verifying dates and rewriting...")
    verified = verify_and_rewrite(candidates, fri, sat, sun)
    confirmed = [e for e in verified if e.get("status") in ("CONFIRMED", "ONGOING")]
    failed = [e for e in verified if e.get("status") == "FAILED"]
    print(f"Verified: {len(confirmed)} passed, {len(failed)} dropped")
    for f in failed:
        print(f"  DROPPED: {f['name']} — {f.get('verification_note', 'no reason')}")

    # Backfill if needed
    if len(confirmed) < 12:
        print(f"Backfilling: need {12 - len(confirmed)} more events...")
        time.sleep(65)
        confirmed = backfill(confirmed, fri, sat, sun)

    # Edge case: everything failed
    if not confirmed:
        print("All events failed verification — sending apology email")
        html = format_email([], fri, sun)
        send_email(html, fri, sun)
        return

    # Pass 3: Link validation
    print("Pass 3: Validating links...")
    validated = validate_links(confirmed)

    # Format and send (wait for rate-limit window before the formatting call)
    print("Waiting 65s before formatting to avoid rate limits...")
    time.sleep(65)
    print("Formatting email...")
    html = format_email(validated, fri, sun)
    send_email(html, fri, sun)

    print(f"\nDone! Sent {len(validated)} events")
    print(f"  CONFIRMED: {sum(1 for e in validated if e.get('status') == 'CONFIRMED')}")
    print(f"  ONGOING:   {sum(1 for e in validated if e.get('status') == 'ONGOING')}")
    link_ok = sum(1 for e in validated if e.get("link_status") == "OK")
    print(f"  Links OK:  {link_ok}/{len(validated)}")


if __name__ == "__main__":
    main()
