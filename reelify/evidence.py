import urllib.parse
import feedparser
import requests
import re

from duckduckgo_search import DDGS

from .config import NEWSAPI_KEY, NEWSAPI_BASE_URL


_RELEVANCE_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "has",
    "into", "about", "over", "after", "years", "year", "anniversary",
    "party", "trend", "trending", "social", "media", "online", "news",
    "world", "report", "reports"
}

def _clean_tag_to_name(tag):
    """
    Turn a hashtag / handle / messy tag into a plain name string.

    Examples:
      "#TravisHead"   -> "Travis Head"
      "Travis_Head"   -> "Travis Head"
      "   travis head " -> "travis head"
    """
    if not tag:
        return ""
    # strip leading #/@ and whitespace
    tag = re.sub(r'^[#@]+', '', str(tag)).strip()
    # keep only alphabetic chunks and join with spaces
    tokens = re.findall(r"[A-Za-z]+", tag)
    return " ".join(tokens).strip()


def _article_mentions_name(article, full_name_lc):
    """
    True if the article's title or description contains the full name.
    """
    if not full_name_lc:
        return False
    title = (article.get("title") or "").lower()
    desc = (article.get("description") or "").lower()
    return full_name_lc in title or full_name_lc in desc


def _map_location_to_country(location):
    """
    Map your frontend 'location' to NewsAPI country codes.
    Extend this mapping as needed.
    """
    if not location:
        return None
    loc = location.lower().strip()
    mapping = {
        "pakistan": "pk",
        "pk": "pk",
        "united-states": "us",
        "united states": "us",
        "usa": "us",
        "us": "us",
        "america": "us",
    }
    return mapping.get(loc)


def _build_news_queries_for_trend(trend_name, trend_description=None):
    """
    Build multiple candidate queries for NewsAPI from a hashtag/trend name
    + optional description (Gemini one-liner).
    Example: "#58YearsOfPPP" -> ["58YearsOfPPP", "Years Of PPP", "PPP", "Pakistan Peoples Party", ...]
    """
    queries = []

    if not trend_name and not trend_description:
        return queries

    name = (trend_name or "").strip()
    base = name.lstrip("#").strip()  # "#58YearsOfPPP" -> "58YearsOfPPP"

    # 1) Raw forms
    if name:
        queries.append(name)
    if base and base != name:
        queries.append(base)

    # 2) Split camelCase / digits into words, e.g. "58YearsOfPPP" -> ["Years", "Of", "PPP"]
    tokens = re.findall(r"[A-Za-z]+", base)
    if tokens:
        joined = " ".join(tokens)  # "Years Of PPP"
        if joined.lower() != base.lower():
            queries.append(joined)
        # Add last token alone (e.g., "PPP")
        last = tokens[-1]
        if len(last) >= 3:
            queries.append(last)

    # 3) Special-case mappings (you can extend this)
    lower = base.lower()
    if "ppp" in lower:
        queries.append("Pakistan Peoples Party")
        queries.append("Pakistan Peoples Party anniversary")

    # 4) Use description (Gemini one-liner) to mine keywords
    if trend_description:
        text = trend_description
        words = re.findall(r"[A-Za-z]{3,}", text)
        stop = {
            "the", "and", "for", "with", "from", "that", "this",
            "have", "has", "into", "about", "over", "after",
            "years", "year", "anniversary", "party", "trend",
            "trending", "social", "media", "online"
        }
        keywords = [w for w in words if w.lower() not in stop]
        if keywords:
            queries.append(" ".join(keywords[:4]))  # e.g. "Pakistan Peoples Party founding"
        # Also add the short description itself
        queries.append(text[:120])

    # 5) Dedupe while preserving order
    seen = set()
    out = []
    for q in queries:
        q_clean = q.strip()
        if not q_clean:
            continue
        key = q_clean.lower()
        if key not in seen:
            seen.add(key)
            out.append(q_clean)

    # Limit to avoid spamming the API
    return out[:6]


def fetch_google_news_evidence(trend_name, location="Pakistan", max_results=5):
    """
    Fetches news from Google News RSS Feed (Pakistan Edition).
    Best for local coverage (Geo, Dawn, ARY, etc.) without an API key.
    """
    print(f"[GoogleNews] Fetching evidence for: {trend_name}...")

    # Clean the query
    query = trend_name.replace("#", "").strip()

    # URL Encode the query (last 7 days)
    encoded_query = urllib.parse.quote(f"{query} when:7d")

    # Pakistan edition of Google News
    # hl=en-PK -> English, Pakistan locale
    # gl=PK   -> Geo Location: Pakistan
    # ceid=PK:en -> Country Edition: Pakistan (English)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-PK&gl=PK&ceid=PK:en"

    try:
        feed = feedparser.parse(rss_url)
        evidence = []

        for entry in feed.entries[:max_results]:
            evidence.append({
                "title": entry.title,
                "description": entry.title,  # Google News often has the key info in the title
                "content_snippet": entry.title,
                "url": entry.link,
                "source_name": entry.source.title if 'source' in entry else "Google News",
                "published_at": getattr(entry, "published", None),
            })

        print(f"[GoogleNews] Found {len(evidence)} articles.")
        return evidence

    except Exception as e:
        print(f"[GoogleNews] Error: {e}")
        return []


def _build_duckduckgo_queries_for_trend(trend_name, trend_description=None, location=None):
    """
    Build queries for DuckDuckGo news search from trend name + optional description.
    - Always keep full name (e.g. 'Travis Head')
    - Do NOT query on first name only
    - Add light context (cricket / politics / Pakistan, etc.)
    """
    queries = []

    name = (trend_name or "").strip()
    base = name.lstrip("#").strip()

    if not base and not trend_description:
        return queries

    # 1) Basic forms
    if base:
        queries.append(base)           # Travis Head
    if " " in base:
        phrase = base                  # "Travis Head"
        if phrase not in queries:
            queries.append(phrase)
        q_quoted = '"%s"' % phrase     # '"Travis Head"'
        if q_quoted not in queries:
            queries.append(q_quoted)

    # 2) Context from description
    desc = (trend_description or "").lower()
    desc_tokens = set(re.findall(r"[a-z]{3,}", desc))

    looks_cricket = any(w in desc_tokens for w in ["cricket", "batsman", "bowler", "ashes"])
    looks_politics = any(w in desc_tokens for w in ["prime", "minister", "president", "party", "election"])
    looks_military = any(w in desc_tokens for w in ["army", "forces", "defence", "defense", "general"])

    base_for_context = base or name

    if looks_cricket:
        queries.append(base_for_context + " cricket")
    if looks_politics:
        queries.append(base_for_context + " politics")
    if looks_military:
        queries.append(base_for_context + " military")

    # 3) Location-aware hint (esp. Pakistan)
    if location:
        loc_lower = str(location).lower()
        if "pakistan" in loc_lower or loc_lower in ("pk", "pak"):
            queries.append(base_for_context + " Pakistan")
        elif loc_lower in ("us", "usa", "united states", "america"):
            queries.append(base_for_context + " USA")

    # 4) Use short description as a query
    if trend_description:
        short_desc = trend_description.strip()
        if short_desc:
            queries.append(short_desc[:160])

    # 5) Dedupe / limit
    seen = set()
    out = []
    for q in queries:
        q_clean = q.strip()
        if not q_clean:
            continue
        key = q_clean.lower()
        if key not in seen:
            seen.add(key)
            out.append(q_clean)

    return out[:6]


def fetch_duckduckgo_evidence_for_trend(trend_name, location=None, max_results=5, trend_description=None):
    """
    Use DuckDuckGo news search as a fallback evidence source.
    Returns article dicts compatible with GeminiStoryGenerator (title, description, content_snippet, url, source_name, published_at).
    """
    if not trend_name:
        return []

    queries = _build_duckduckgo_queries_for_trend(
        trend_name=trend_name,
        trend_description=trend_description,
        location=location,
    )

    if not queries:
        print("[DuckDuckGo] No queries built for trend '%s'" % trend_name)
        return []

    results = []

    try:
        with DDGS() as ddg:
            for q in queries:
                print("[DuckDuckGo] TRY news q=%s" % q)
                for item in ddg.news(q, max_results=max_results):
                    title = item.get("title") or ""
                    desc = item.get("description") or ""
                    url = item.get("url") or item.get("link") or ""
                    source = item.get("source") or ""
                    published = item.get("date") or item.get("published")

                    if not title and not desc:
                        continue

                    results.append({
                        "title": title,
                        "description": desc,
                        "content_snippet": (desc or "")[:500],
                        "url": url,
                        "source_name": source,
                        "published_at": published,
                    })

                print("[DuckDuckGo] news items found so far for '%s': %d" % (trend_name, len(results)))
                if results:
                    break

    except Exception as e:
        print("[DuckDuckGo] fetch failed for '%s': %s" % (trend_name, e))
        return []

    print("[DuckDuckGo] FINAL evidence count for '%s' (location=%s): %d" %
          (trend_name, location, len(results)))
    for i, ev in enumerate(results[:max_results]):
        print("  [%d] %s (%s)" % (i + 1, ev["title"], ev["source_name"]))

    return results[:max_results]


def fetch_news_evidence_for_trend(trend_name, location=None, max_articles=4, trend_description=None):
    if not NEWSAPI_KEY or NEWSAPI_KEY == "YOUR_NEWSAPI_KEY_HERE":
        print("[NewsAPI] Missing API key; skipping evidence fetch.")
        return []

    queries = _build_news_queries_for_trend(trend_name, trend_description)
    if not queries:
        print("[NewsAPI] No queries built for trend '%s'" % trend_name)
        return []

    country = _map_location_to_country(location)
    base_params = {
        "apiKey": NEWSAPI_KEY,
        "pageSize": max_articles,
    }

    articles = []

    try:
        # 1) Try country-specific top-headlines
        if country:
            for q in queries:
                params_local = dict(base_params)
                params_local["country"] = country
                params_local["q"] = q

                print("\n[NewsAPI] TRY top-headlines country=%s q=%s" % (country, q))
                resp = requests.get(NEWSAPI_BASE_URL + "/top-headlines", params=params_local, timeout=8)

                print("[NewsAPI] URL:", resp.url)
                print("[NewsAPI] STATUS:", resp.status_code)

                # show first 800 chars of body for debugging
                print("[NewsAPI] BODY (truncated):", resp.text[:800])

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ok":
                        arts = data.get("articles") or []
                        print("[NewsAPI] top-headlines articles found:", len(arts))
                        if arts:
                            articles = arts
                            break
                    else:
                        print("[NewsAPI] top-headlines non-ok status:", data)
                else:
                    print("[NewsAPI] top-headlines HTTP error:", resp.status_code)

        # 2) If still nothing, try global 'everything'
        if not articles:
            for q in queries:
                params_global = dict(base_params)
                params_global.update({
                    "q": q,
                    "language": "en",
                    "sortBy": "relevancy",
                })

                print("\n[NewsAPI] TRY everything q=%s" % q)
                resp = requests.get(NEWSAPI_BASE_URL + "/everything", params=params_global, timeout=8)

                print("[NewsAPI] URL:", resp.url)
                print("[NewsAPI] STATUS:", resp.status_code)
                print("[NewsAPI] BODY (truncated):", resp.text[:800])

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ok":
                        arts = data.get("articles") or []
                        print("[NewsAPI] everything articles found:", len(arts))
                        if arts:
                            articles = arts
                            break
                    else:
                        print("[NewsAPI] everything non-ok status:", data)
                else:
                    print("[NewsAPI] everything HTTP error:", resp.status_code)

        # ------------------------------------------------------------------
        # 3) NEW: entity-aware filtering (avoid Travis Head vs Travis Turner mix)
        # ------------------------------------------------------------------
        cleaned_name = _clean_tag_to_name(trend_name)
        if cleaned_name:
            name_tokens = cleaned_name.split()
            if len(name_tokens) >= 2:
                # looks like a person-style full name; require full name in title/description
                full_name_lc = cleaned_name.lower()
                filtered_articles = []
                for art in articles or []:
                    if _article_mentions_name(art, full_name_lc):
                        filtered_articles.append(art)

                # Only override if we actually found some that match full name
                if filtered_articles:
                    print("[NewsAPI] Entity filter kept %d/%d articles for '%s'" %
                          (len(filtered_articles), len(articles or []), cleaned_name))
                    articles = filtered_articles
                else:
                    print("[NewsAPI] Entity filter found no matches for full name '%s'; keeping original set." %
                          cleaned_name)

        # 4) Build evidence list (unchanged logic)
        evidence = []
        for art in (articles or [])[:max_articles]:
            source = art.get("source") or {}
            content = art.get("content") or ""
            evidence.append({
                "title": art.get("title") or "",
                "description": art.get("description") or "",
                "content_snippet": content[:500] if content else "",
                "url": art.get("url") or "",
                "source_name": source.get("name") or "",
                "published_at": art.get("publishedAt"),
            })

        print("\n[NewsAPI] FINAL evidence count for '%s' (location=%s): %d" %
              (trend_name, location, len(evidence)))

        # Also dump titles to see what we got
        for i, ev in enumerate(evidence):
            print("  [%d] %s (%s)" % (i + 1, ev["title"], ev["source_name"]))

        return evidence

    except Exception as e:
        print("[NewsAPI] fetch failed for '%s': %s" % (trend_name, e))
        return []


def _normalize_text_for_relevance(text):
    if not text:
        return ""
    return str(text).lower()


def _extract_tokens_for_relevance(text):
    """
    Simple tokenization for relevance scoring.
    """
    text = _normalize_text_for_relevance(text)
    tokens = re.findall(r"[a-z][a-z\-']+", text)
    return {t for t in tokens if len(t) > 3 and t not in _RELEVANCE_STOPWORDS}


def _score_article_relevance(trend_name, trend_description, location, article):
    """
    Heuristic similarity between (trend_name+description) and article (title+desc+snippet).

    - Overlap of tokens.
    - Extra penalty if context is Pakistan but article never mentions Pakistan.
    """
    if not article:
        return 0.0

    topic_text = u"%s %s" % (
        str(trend_name or ""),
        str(trend_description or ""),
    )
    topic_tokens = _extract_tokens_for_relevance(topic_text)

    if not topic_tokens:
        # If we can't build any useful tokens, don't try to be clever.
        return 0.0

    # Article text: title + description + content_snippet if present
    art_title = article.get("title") or ""
    art_desc = article.get("description") or ""
    art_snip = article.get("content_snippet") or ""

    art_text = u"%s %s %s" % (art_title, art_desc, art_snip)
    art_tokens = _extract_tokens_for_relevance(art_text)

    if not art_tokens:
        return 0.0

    overlap = topic_tokens & art_tokens
    base_score = float(len(overlap)) / float(len(topic_tokens))

    # If the *exact* base tag (without '#') appears as a substring in the article text,
    # give a small bonus.
    base = (trend_name or "").lstrip("#").strip().lower()
    if base and base in _normalize_text_for_relevance(art_text):
        base_score += 0.3

    # Pakistan context penalty: if we are in Pakistan but article never mentions Pakistan
    loc_lower = str(location or "").lower()
    is_pakistani_context = (
        "pakistan" in loc_lower
        or loc_lower in ("pk", "pak", "karachi", "lahore", "islamabad", "rawalpindi")
    )
    if is_pakistani_context:
        art_lower = _normalize_text_for_relevance(art_text)
        if ("pakistan" not in art_lower) and ("pakistani" not in art_lower):
            # Strong down-weight for non-Pakistan articles in a Pakistani trend context
            base_score *= 0.3

    # Clamp to [0, 1]
    if base_score < 0.0:
        base_score = 0.0
    if base_score > 1.0:
        base_score = 1.0

    return base_score


def _filter_evidence_by_relevance(trend_name, trend_description, location,
                                  articles, max_articles):
    """
    Rank articles by relevance and keep only the most relevant ones.

    - Drop obviously off-topic evidence (e.g., Bihar election pieces for a
      Pakistani transport hashtag).
    - Always keep at least 1 article if there is anything at all, but
      prefer those with non-zero scores.
    """
    if not articles:
        return []

    scored = []
    for art in articles:
        score = _score_article_relevance(trend_name, trend_description, location, art)
        scored.append((score, art))

    # Sort best-first
    scored.sort(key=lambda x: x[0], reverse=True)

    # Prefer non-zero scores
    filtered = [art for (s, art) in scored if s > 0.0]

    # If *everything* got 0.0, keep just the top-1 as ultra-weak context
    if not filtered and scored:
        filtered = [scored[0][1]]

    return filtered[:max_articles]


def fetch_evidence_for_trend(trend_name, location=None, max_articles=4, trend_description=None):
    """
    Unified evidence fetcher:

    - If context is Pakistan -> use Google News Pakistan RSS as primary source
      (local outlets, better alignment with Pakistani trends).
    - Else -> use NewsAPI as primary source.
    - In both cases -> if we end up with < 2 articles, fall back to DuckDuckGo.
    - Finally -> dedupe + (optionally) filter by relevance to the trend and context.
    """
    final_evidence = []
    seen_urls = set()

    loc_lower = str(location or "").lower()
    is_pakistani_context = (
        "pakistan" in loc_lower
        or loc_lower in ("pk", "pak", "karachi", "lahore", "islamabad", "rawalpindi")
    )

    # ------------------------------------------------------------------
    # 1. Primary source depending on context
    # ------------------------------------------------------------------
    if is_pakistani_context:
        # PAKISTAN -> Google News PK edition
        print(f"[Evidence] Using Google News Pakistan for '{trend_name}'...")
        google_results = fetch_google_news_evidence(
            trend_name=trend_name,
            location="Pakistan",
            max_results=max_articles,
        ) or []

        for art in google_results:
            url = (art or {}).get("url") or ""
            if url and url not in seen_urls:
                final_evidence.append(art)
                seen_urls.add(url)

    else:
        # GLOBAL -> NewsAPI
        if NEWSAPI_KEY and NEWSAPI_KEY != "YOUR_NEWSAPI_KEY_HERE":
            print(f"[Evidence] Using NewsAPI for '{trend_name}' (location={location})...")
            news_results = fetch_news_evidence_for_trend(
                trend_name=trend_name,
                location=location,
                max_articles=max_articles,
                trend_description=trend_description,
            ) or []

            for art in news_results:
                url = (art or {}).get("url") or ""
                if url and url not in seen_urls:
                    final_evidence.append(art)
                    seen_urls.add(url)
        else:
            print("[Evidence] No valid NewsAPI key; skipping NewsAPI and relying on DuckDuckGo/GoogleNews.")

    # ------------------------------------------------------------------
    # 2. Fallback: DuckDuckGo if we still don't have much
    # ------------------------------------------------------------------
    if len(final_evidence) < 2:
        print(f"[Evidence] Triggering DuckDuckGo fallback (current count: {len(final_evidence)})...")

        # Make sure we don't exceed max_articles, but try to get at least 2 from DDG
        needed = max(2, max_articles - len(final_evidence))

        ddg_results = fetch_duckduckgo_evidence_for_trend(
            trend_name=trend_name,
            location=location,
            max_results=needed,
            trend_description=trend_description,
        ) or []

        for art in ddg_results:
            url = (art or {}).get("url") or ""
            if url and url not in seen_urls:
                final_evidence.append(art)
                seen_urls.add(url)

    print(f"[Evidence] Combined Total BEFORE relevance filter: {len(final_evidence)} articles.")

    # ------------------------------------------------------------------
    # 3. Optional: filter by relevance to avoid obviously off-topic stuff
    # ------------------------------------------------------------------
    try:
        # This assumes you already have _filter_evidence_by_relevance as we discussed earlier.
        final_evidence = _filter_evidence_by_relevance(
            trend_name=trend_name,
            trend_description=trend_description,
            location=location,
            articles=final_evidence,
            max_articles=max_articles,
        )
    except NameError:
        # If you haven't added the relevance filter yet, just skip this step.
        pass

    print(f"[Evidence] Total AFTER relevance filter: {len(final_evidence)} articles.")
    for i, ev in enumerate(final_evidence):
        print(f"  [{i+1}] {ev.get('title')} ({ev.get('source_name')})")

    return final_evidence[:max_articles]
