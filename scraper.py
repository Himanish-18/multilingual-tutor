"""
scraper.py — Dynamic knowledge base scraper for multilingual-tutor.

Sources:
  1. Wiktionary (idioms, proverbs, phrases per language)
  2. Tatoeba API  (example sentences with translations)

Usage:
  from scraper import scrape_all
  entries = scrape_all()          # returns list of dicts matching data.json schema
  entries = scrape_all(languages=["Japanese", "French"])  # specific languages

Requirements (add to requirements.txt):
  requests
  beautifulsoup4
"""

import time
import logging
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="[Scraper] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language config — maps display name → Wiktionary category slugs + Tatoeba
# language codes.  Extend this dict to add more languages.
# ---------------------------------------------------------------------------
LANGUAGE_CONFIG = {
    "Bengali":          {"wiki_lang": "Bengali",          "tatoeba_code": "ben"},
    "Hindi":            {"wiki_lang": "Hindi",            "tatoeba_code": "hin"},
    "Punjabi":          {"wiki_lang": "Punjabi",          "tatoeba_code": "pan"},
    "Urdu":             {"wiki_lang": "Urdu",             "tatoeba_code": "urd"},
    "Sanskrit":         {"wiki_lang": "Sanskrit",         "tatoeba_code": "san"},
    "Japanese":         {"wiki_lang": "Japanese",         "tatoeba_code": "jpn"},
    "Mandarin Chinese": {"wiki_lang": "Chinese",          "tatoeba_code": "cmn"},
    "Korean":           {"wiki_lang": "Korean",           "tatoeba_code": "kor"},
    "Spanish":          {"wiki_lang": "Spanish",          "tatoeba_code": "spa"},
    "French":           {"wiki_lang": "French",           "tatoeba_code": "fra"},
    "German":           {"wiki_lang": "German",           "tatoeba_code": "deu"},
    "Italian":          {"wiki_lang": "Italian",          "tatoeba_code": "ita"},
    "Portuguese":       {"wiki_lang": "Portuguese",       "tatoeba_code": "por"},
    "Russian":          {"wiki_lang": "Russian",          "tatoeba_code": "rus"},
    "Arabic":           {"wiki_lang": "Arabic",           "tatoeba_code": "ara"},
    "Turkish":          {"wiki_lang": "Turkish",          "tatoeba_code": "tur"},
    "Swahili":          {"wiki_lang": "Swahili",          "tatoeba_code": "swh"},
    "Yoruba":           {"wiki_lang": "Yoruba",           "tatoeba_code": "yor"},
    "Persian":          {"wiki_lang": "Persian",          "tatoeba_code": "pes"},
    "Vietnamese":       {"wiki_lang": "Vietnamese",       "tatoeba_code": "vie"},
    "Thai":             {"wiki_lang": "Thai",             "tatoeba_code": "tha"},
    "Indonesian":       {"wiki_lang": "Indonesian",       "tatoeba_code": "ind"},
}

HEADERS = {"User-Agent": "multilingual-tutor-bot/1.0 (educational project; contact via github)"}
REQUEST_DELAY = 1.2   # seconds between requests — be a polite scraper


# ===========================================================================
# SOURCE 1 — Wiktionary
# ===========================================================================

def _wiktionary_category_url(lang_name: str, category: str) -> str:
    """Build a Wiktionary category URL, e.g. Category:Bengali_idioms."""
    slug = f"{lang_name}_{category}".replace(" ", "_")
    return f"https://en.wiktionary.org/wiki/Category:{slug}"


def _parse_wiktionary_entry(page_title: str, language: str) -> dict | None:
    """
    Fetch a single Wiktionary entry page and extract:
      - the phrase
      - its definition / meaning
      - any usage example found
    Returns a dict matching data.json schema, or None on failure.
    """
    url = f"https://en.wiktionary.org/wiki/{requests.utils.quote(page_title)}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # --- grab first definition ---
    meaning = ""
    ol = soup.find("ol")
    if ol:
        first_li = ol.find("li")
        if first_li:
            # Strip nested <ul> (usage examples inside the def list)
            for nested in first_li.find_all("ul"):
                nested.decompose()
            meaning = first_li.get_text(" ", strip=True)

    if not meaning:
        return None

    # --- grab first usage example if present ---
    context = ""
    example_dl = soup.find("dl")
    if example_dl:
        context = example_dl.get_text(" ", strip=True)[:200]

    return {
        "language": language,
        "category": "Idiom",
        "phrase": page_title,
        "literal_translation": "",          # Wiktionary rarely has this; left blank
        "meaning": meaning[:300],
        "context": context or "See Wiktionary entry for usage examples.",
        "cultural_nuance": f"Sourced from Wiktionary ({language} idioms category).",
    }


def scrape_wiktionary(language: str, wiki_lang: str, max_entries: int = 20) -> list[dict]:
    """
    Scrape up to `max_entries` idiom entries for a language from Wiktionary.
    Iterates through category pages following 'next page' links.
    """
    entries = []
    url = _wiktionary_category_url(wiki_lang, "idioms")
    visited_urls = set()

    while url and len(entries) < max_entries:
        if url in visited_urls:
            break
        visited_urls.add(url)

        log.info(f"Wiktionary [{language}] → {url}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            log.warning(f"Category page fetch failed: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Category member links live in <div class="mw-category">
        cat_div = soup.find("div", class_="mw-category")
        if not cat_div:
            break

        links = cat_div.find_all("a")
        for link in links:
            if len(entries) >= max_entries:
                break
            title = link.get("title", "").strip()
            if not title or ":" in title:   # skip meta / category links
                continue

            entry = _parse_wiktionary_entry(title, language)
            if entry:
                entries.append(entry)
                log.info(f"  ✓ {title}")

            time.sleep(REQUEST_DELAY)

        # Follow "next page" if available
        next_link = soup.find("a", string=lambda t: t and "next page" in t.lower())
        url = ("https://en.wiktionary.org" + next_link["href"]) if next_link else None

    log.info(f"Wiktionary [{language}]: collected {len(entries)} entries.")
    return entries


# ===========================================================================
# SOURCE 2 — Tatoeba API
# Tatoeba is CC-BY 2.0 licensed — free to use with attribution.
# API docs: https://tatoeba.org/en/api#sentences
# ===========================================================================

TATOEBA_API = "https://tatoeba.org/en/api_v0/search"


def scrape_tatoeba(language: str, tatoeba_code: str, max_entries: int = 20) -> list[dict]:
    """
    Fetch example sentences from Tatoeba for a given language code.
    Each sentence becomes a 'Phrase' entry with a real usage example.
    We fetch sentences that also have an English translation available.
    """
    entries = []
    page = 1
    page_size = min(max_entries, 20)

    while len(entries) < max_entries:
        params = {
            "from": tatoeba_code,
            "to": "eng",
            "orphans": "no",
            "unapproved": "no",
            "page": page,
            "limit": page_size,
        }

        log.info(f"Tatoeba [{language}] page {page}")
        try:
            resp = requests.get(TATOEBA_API, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"Tatoeba API error: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            if len(entries) >= max_entries:
                break

            phrase = item.get("text", "").strip()
            if not phrase or len(phrase) > 120:   # skip very long sentences
                continue

            # Try to get the English translation
            translation = ""
            for trans in item.get("translations", []):
                # translations is a list of lists
                if isinstance(trans, list):
                    for t in trans:
                        if isinstance(t, dict) and t.get("lang") == "eng":
                            translation = t.get("text", "")
                            break
                if translation:
                    break

            if not translation:
                continue

            entries.append({
                "language": language,
                "category": "Phrase",
                "phrase": phrase,
                "literal_translation": translation,
                "meaning": f'This phrase means: "{translation}"',
                "context": "Example sentence from Tatoeba corpus.",
                "cultural_nuance": (
                    f"Authentic {language} sentence contributed by native speakers "
                    f"on Tatoeba (CC BY 2.0)."
                ),
            })

        page += 1
        time.sleep(REQUEST_DELAY)

    log.info(f"Tatoeba [{language}]: collected {len(entries)} entries.")
    return entries


# ===========================================================================
# Main public function
# ===========================================================================

def scrape_all(
    languages: list[str] | None = None,
    max_per_source_per_lang: int = 15,
) -> list[dict]:
    """
    Scrape all configured languages (or a subset) from both Wiktionary and
    Tatoeba.  Returns a flat list of entry dicts compatible with data.json.

    Args:
        languages: list of language display names to scrape, e.g. ["French",
                   "Japanese"].  Pass None to scrape all configured languages.
        max_per_source_per_lang: how many entries to fetch per source per
                                  language.  Keep ≤ 20 to stay polite.
    """
    target = languages or list(LANGUAGE_CONFIG.keys())
    all_entries: list[dict] = []

    for lang in target:
        config = LANGUAGE_CONFIG.get(lang)
        if not config:
            log.warning(f"No config found for language '{lang}' — skipping.")
            continue

        log.info(f"━━ Scraping: {lang} ━━")

        wiki_entries = scrape_wiktionary(
            language=lang,
            wiki_lang=config["wiki_lang"],
            max_entries=max_per_source_per_lang,
        )
        all_entries.extend(wiki_entries)

        tatoeba_entries = scrape_tatoeba(
            language=lang,
            tatoeba_code=config["tatoeba_code"],
            max_entries=max_per_source_per_lang,
        )
        all_entries.extend(tatoeba_entries)

    log.info(f"Scraping complete. Total entries: {len(all_entries)}")
    return all_entries


# ---------------------------------------------------------------------------
# Optional: run standalone to test / pre-cache a data.json snapshot
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    entries = scrape_all(max_per_source_per_lang=10)
    out_path = "data_scraped.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(entries)} entries → {out_path}")
