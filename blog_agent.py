#!/usr/bin/env python3
"""
BRI Blog Agent
Generates weekly blog posts for Blue Ridge Intelligence using the Anthropic API,
commits them to the GitHub Pages repo, and optionally posts to LinkedIn.

Usage:
    python blog_agent.py                    # Run normally (publish)
    python blog_agent.py --dry-run          # Generate only, no publish
    python blog_agent.py --draft            # Commit as draft, skip LinkedIn
"""

import os
import sys
import json
import base64
import argparse
import smtplib
import datetime
import re
import anthropic
import requests
from email.mime.text import MIMEText
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — set via environment variables or edit defaults below
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
GITHUB_TOKEN        = os.environ.get("GH_PAT", "")
GITHUB_REPO         = os.environ.get("GH_REPO", "carltonjr03/github.io")
GITHUB_BRANCH       = os.environ.get("GH_BRANCH", "main")
POSTS_DIR           = os.environ.get("POSTS_DIR", "_posts")       # Jekyll posts folder in your repo
DRAFTS_DIR          = os.environ.get("DRAFTS_DIR", "_drafts")     # Jekyll drafts folder

LINKEDIN_ACCESS_TOKEN = os.environ.get("LINKEDIN_ACCESS_TOKEN", "")
LINKEDIN_AUTHOR_URN   = os.environ.get("LINKEDIN_AUTHOR_URN", "")  # e.g. "urn:li:person:XXXXXXXX"

NOTIFY_EMAIL        = os.environ.get("NOTIFY_EMAIL", "")          # Your email for run notifications
SENDGRID_API_KEY    = os.environ.get("SENDGRID_API_KEY", "")      # Optional — for email alerts

LOG_FILE            = "posts_log/run_log.json"
SYSTEM_PROMPT_FILE  = "prompts/system_prompt.txt"

CONTENT_THEMES = [
    "AI for operations",
    "Data analytics for decisions",
    "WNC business spotlight / use case",
    "Demystifying AI",
    "Getting started",
]

MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_FILE, "r") as f:
        return f.read()


def load_run_log() -> list:
    if not Path(LOG_FILE).exists():
        return []
    with open(LOG_FILE, "r") as f:
        return json.load(f)


def save_run_log(log: list):
    Path("posts_log").mkdir(exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def get_week_number() -> int:
    """Returns ISO week number, used to rotate content themes."""
    return datetime.date.today().isocalendar()[1]


def get_theme_for_week() -> str:
    week = get_week_number()
    return CONTENT_THEMES[(week - 1) % len(CONTENT_THEMES)]


def get_recent_titles(log: list, n: int = 8) -> list[str]:
    return [entry.get("title", "") for entry in log[-n:]]


def build_user_prompt(theme: str, recent_titles: list[str]) -> str:
    today = datetime.date.today().strftime("%B %d, %Y")
    week_num = get_week_number()
    recent_str = "\n".join(f"- {t}" for t in recent_titles) if recent_titles else "None yet."
    return f"""Today is {today}. This is week {week_num} of the year.

Assigned content theme for this week: {theme}

Recent post titles (avoid repeating these topics):
{recent_str}

Please write the weekly BRI blog post following all instructions in the system prompt.
Use web search if you need current statistics, recent AI news relevant to small businesses,
or local WNC context. Return only the JSON object — no other text."""


def validate_post(post: dict) -> list[str]:
    """Returns a list of validation errors. Empty list = pass."""
    errors = []
    long_form = post.get("long_form_markdown", "")
    linkedin = post.get("linkedin_post", "")
    word_count = len(long_form.split())

    if word_count < 800:
        errors.append(f"Long-form post too short: {word_count} words (minimum 800)")
    if word_count > 1500:
        errors.append(f"Long-form post too long: {word_count} words (maximum 1500)")
    if "<!-- BRI_POST_END -->" not in long_form:
        errors.append("Missing <!-- BRI_POST_END --> marker")
    if len(linkedin.split()) < 100:
        errors.append(f"LinkedIn post too short: {len(linkedin.split())} words")
    if not post.get("slug"):
        errors.append("Missing slug")
    if not post.get("meta_description"):
        errors.append("Missing meta_description")
    banned = ["leverage", "utilize", "game-changer", "unlock", "revolutionize", "seamlessly"]
    for word in banned:
        if word.lower() in long_form.lower():
            errors.append(f"Banned word found: '{word}'")

    return errors


def build_jekyll_post(post: dict, is_draft: bool = False) -> tuple[str, str]:
    """Returns (file_path_in_repo, file_content)."""
    today = datetime.date.today()
    date_str = today.strftime("%Y-%m-%d")
    slug = post["slug"]
    tags_yaml = "\n  - ".join(post.get("tags", []))

    front_matter = f"""---
layout: post
title: "{post['title']}"
date: {date_str}
description: "{post['meta_description']}"
tags:
  - {tags_yaml}
published: {"false" if is_draft else "true"}
---

"""
    content = front_matter + post["long_form_markdown"]

    if is_draft:
        file_path = f"{DRAFTS_DIR}/{slug}.md"
    else:
        file_path = f"{POSTS_DIR}/{date_str}-{slug}.md"

    return file_path, content


def github_commit_file(file_path: str, content: str, commit_message: str) -> dict:
    """Creates or updates a file in the GitHub repo."""
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Check if file exists (need SHA for updates)
    sha = None
    r = requests.get(api_url, headers=headers)
    if r.status_code == 200:
        sha = r.json().get("sha")

    payload = {
        "message": commit_message,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(api_url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()


def post_to_linkedin(post: dict) -> dict:
    """Posts the LinkedIn version via the LinkedIn UGC Posts API."""
    if not LINKEDIN_ACCESS_TOKEN or not LINKEDIN_AUTHOR_URN:
        raise ValueError("LINKEDIN_ACCESS_TOKEN and LINKEDIN_AUTHOR_URN must be set")

    today = datetime.date.today().strftime("%Y-%m-%d")
    slug = post["slug"]
    post_url = f"https://blueridgeintelligence.com/{POSTS_DIR}/{today}-{slug}"

    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    payload = {
        "author": LINKEDIN_AUTHOR_URN,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": post["linkedin_post"]
                },
                "shareMediaCategory": "ARTICLE",
                "media": [
                    {
                        "status": "READY",
                        "originalUrl": post_url,
                        "title": {"text": post["title"]},
                        "description": {"text": post.get("meta_description", "")},
                    }
                ],
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }

    r = requests.post(
        "https://api.linkedin.com/v2/ugcPosts",
        headers=headers,
        json=payload,
    )
    r.raise_for_status()
    return r.json()


def send_notification(subject: str, body: str):
    """Sends an email notification via SendGrid or falls back to stdout."""
    print(f"\n{'='*60}")
    print(f"NOTIFICATION: {subject}")
    print(body)
    print('='*60)

    if not SENDGRID_API_KEY or not NOTIFY_EMAIL:
        return

    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "personalizations": [{"to": [{"email": NOTIFY_EMAIL}]}],
        "from": {"email": "noreply@blueridgeintelligence.com", "name": "BRI Blog Agent"},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
    try:
        r = requests.post("https://api.sendgrid.com/v3/mail/send", headers=headers, json=payload)
        r.raise_for_status()
    except Exception as e:
        print(f"Email notification failed: {e}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(dry_run: bool = False, draft: bool = False):
    print("BRI Blog Agent starting...")
    run_log = load_run_log()
    system_prompt = load_system_prompt()
    theme = get_theme_for_week()
    recent_titles = get_recent_titles(run_log)
    user_prompt = build_user_prompt(theme, recent_titles)

    print(f"  Week theme: {theme}")
    print(f"  Model: {MODEL}")

    # --- Call Claude ---
    print("  Calling Anthropic API...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system_prompt,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract the text content block (last one if there are tool uses)
    raw_text = ""
    for block in response.content:
        if block.type == "text":
            raw_text = block.text

# Extract JSON — find the first { and last } to handle any preamble or fences
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not json_match:
        print(f"ERROR: No JSON object found in Claude output.\nRaw output:\n{raw_text[:500]}")
        send_notification(
            "BRI Blog Agent FAILED — no JSON found",
            f"Raw output:\n{raw_text[:800]}",
        )
        sys.exit(1)
    raw_text = json_match.group(0)

    try:
        post = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"ERROR: Claude returned invalid JSON.\n{e}\nRaw output:\n{raw_text[:500]}")
        send_notification(
            "BRI Blog Agent FAILED — JSON parse error",
            f"Claude returned invalid JSON.\n\nError: {e}\n\nRaw output:\n{raw_text[:800]}",
        )
        sys.exit(1)

    print(f"  Generated: \"{post.get('title', '(no title)')}\"")
    print(f"  Estimated word count: {post.get('word_count_estimate', '?')}")

    # --- Validate ---
    errors = validate_post(post)
    if errors:
        msg = "Validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        print(f"WARNING: {msg}")
        if not dry_run:
            send_notification("BRI Blog Agent WARNING — validation issues", msg)

    if dry_run:
        print("\n--- DRY RUN: Long-form post preview ---")
        print(post.get("long_form_markdown", "")[:1200])
        print("\n--- LinkedIn post ---")
        print(post.get("linkedin_post", ""))
        print("\nDry run complete. Nothing published.")
        return

    # --- Build Jekyll file ---
    file_path, file_content = build_jekyll_post(post, is_draft=draft)

    # --- Commit to GitHub ---
    print(f"  Committing to GitHub: {file_path}")
    commit_msg = f"blog: {post['title']}"
    github_result = github_commit_file(file_path, file_content, commit_msg)
    post_url = github_result.get("content", {}).get("html_url", "(url unavailable)")
    print(f"  Committed: {post_url}")

    # --- Post to LinkedIn (skip if draft) ---
    linkedin_url = "(skipped — draft mode)"
    if not draft and LINKEDIN_ACCESS_TOKEN:
        print("  Posting to LinkedIn...")
        linkedin_result = post_to_linkedin(post)
        linkedin_url = linkedin_result.get("id", "(posted — no URL returned)")
        print(f"  LinkedIn post ID: {linkedin_url}")
    elif not draft:
        print("  Skipping LinkedIn (LINKEDIN_ACCESS_TOKEN not set)")

    # --- Log the run ---
    log_entry = {
        "date": datetime.date.today().isoformat(),
        "title": post["title"],
        "slug": post["slug"],
        "theme": theme,
        "tags": post.get("tags", []),
        "word_count": post.get("word_count_estimate"),
        "github_path": file_path,
        "linkedin_id": linkedin_url,
        "draft": draft,
        "validation_warnings": errors,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    run_log.append(log_entry)
    save_run_log(run_log)

    # --- Notify ---
    status = "DRAFT committed" if draft else "PUBLISHED"
    send_notification(
        f"BRI Blog Agent — {status}: {post['title']}",
        f"Title: {post['title']}\n"
        f"Theme: {theme}\n"
        f"Words: {post.get('word_count_estimate')}\n"
        f"GitHub: {post_url}\n"
        f"LinkedIn: {linkedin_url}\n"
        f"Tokens used: {response.usage.input_tokens} in / {response.usage.output_tokens} out\n"
        + (f"\nWarnings:\n" + "\n".join(errors) if errors else ""),
    )

    print(f"\nDone. Post '{post['title']}' {status.lower()}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BRI Blog Agent")
    parser.add_argument("--dry-run", action="store_true", help="Generate post but do not publish")
    parser.add_argument("--draft", action="store_true", help="Commit as Jekyll draft, skip LinkedIn")
    args = parser.parse_args()
    run(dry_run=args.dry_run, draft=args.draft)
