import re

from .evidence import _clean_tag_to_name

def _split_story_and_scenes(story_text: str):
    """
    Split the full '# STORY ... # SCENES ...' text into (story_part, scenes_part).
    If '# SCENES' is missing, scenes_part will be "".
    """
    if not story_text:
        return "", ""
    lower = story_text.lower()
    idx = lower.find("# scenes")
    if idx == -1:
        return story_text, ""
    return story_text[:idx], story_text[idx:]


def _split_sentences(text: str):
    """
    Very light sentence splitting on . ! ? boundaries.
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _extract_scene_descriptions(scenes_part: str):
    """
    Extract 'Scene X: ...' descriptions from the SCENES block.
    Returns a list of strings (one per scene).
    """
    descs = []
    for line in scenes_part.splitlines():
        m = re.match(r'\s*scene\s*\d+\s*:\s*(.+)', line, flags=re.IGNORECASE)
        if m:
            descs.append(m.group(1).strip())
    return descs


def evaluate_story_quality(story_text: str,
                           evidence_articles: list,
                           tag: str,
                           description: str) -> dict:
    """
    Pure heuristic evaluation of a generated story.

    Returns a dict like:
    {
      "factual_coverage": {"score": 0.8, "notes": "..."},
      "factual_faithfulness": {...},
      "uncertainty_handling": {...},
      "topic_relevance": {...},
      "tone_neutrality": {...},
      "scene_compliance": {...},
      "format_correctness": {...},
      "entity_consistency": {...},
      "overall_score": 0.83
    }
    """
    # ----------------- Build evidence text blob -----------------
    evidence_chunks = []
    for art in evidence_articles or []:
        for key in ("title", "description", "content_snippet", "content"):
            if art.get(key):
                evidence_chunks.append(str(art[key]))
    evidence_text = " ".join(evidence_chunks).lower()
    has_evidence = bool(evidence_text.strip())

    # ----------------- Split story vs scenes -----------------
    story_part, scenes_part = _split_story_and_scenes(story_text or "")
    story_lower = (story_part or "").lower()
    scenes_lower = (scenes_part or "").lower()

    sentences = _split_sentences(story_part)
    scene_descs = _extract_scene_descriptions(scenes_part)

    # ----------------- Metric: format correctness -----------------
    fmt_ok = ("# story" in story_lower) and ("# scenes" in story_lower or "# scenes" in scenes_lower)
    fmt_score = 1.0 if fmt_ok and scene_descs else 0.0
    fmt_notes = "Output contains # STORY, # SCENES and at least one Scene line." if fmt_score == 1.0 \
        else "Missing # STORY / # SCENES headers or Scene lines."

    # ----------------- Metric: factual coverage -----------------
    if not sentences or not has_evidence:
        cov_score = 0.5  # can't really judge, neutral-ish
        cov_notes = "Insufficient sentences or no evidence; coverage set to neutral 0.5."
    else:
        supported = 0
        for sent in sentences:
            tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", sent.lower())
            if any(tok in evidence_text for tok in tokens):
                supported += 1
        cov_score = supported / max(1, len(sentences))
        cov_notes = f"{supported}/{len(sentences)} sentences have token overlap with evidence."

    # ----------------- Metric: factual faithfulness (numbers) -----------------
    nums_story = re.findall(r"\d+", story_part)
    nums_evidence = re.findall(r"\d+", evidence_text)
    nums_evidence_set = set(nums_evidence)

    if not nums_story:
        faith_score = 1.0
        faith_notes = "Story uses no numeric facts; nothing to check."
    elif not has_evidence:
        faith_score = 0.0
        faith_notes = "Story uses numeric facts but there is no evidence; all numbers considered speculative."
    else:
        bad = 0
        for n in nums_story:
            if n not in nums_evidence_set:
                bad += 1
        if bad == 0:
            faith_score = 1.0
            faith_notes = "All numeric details appear in evidence."
        else:
            faith_score = max(0.0, 1.0 - bad / max(1, len(nums_story)))
            faith_notes = f"{bad}/{len(nums_story)} numeric values not found in evidence."

    # ----------------- Metric: uncertainty handling -----------------
    lower_story = story_lower
    has_uncertainty = any(
        phrase in lower_story
        for phrase in [
            "unclear",
            "unconfirmed",
            "not fully known",
            "still emerging",
            "not yet confirmed",
            "details remain unclear",
        ]
    )

    if has_evidence:
        # If we actually have evidence, we don't require explicit uncertainty
        unc_score = 1.0
        unc_notes = "Evidence available; explicit uncertainty not strictly required."
    else:
        if has_uncertainty:
            unc_score = 1.0
            unc_notes = "No evidence but story explicitly flags uncertainty."
        else:
            # no evidence and no uncertainty phrasing
            if nums_story:
                unc_score = 0.0
                unc_notes = "No evidence, numeric specifics and no uncertainty flags; treated as speculative."
            else:
                unc_score = 0.3
                unc_notes = "No evidence and no uncertainty phrasing, but story is non-numeric and high-level."

    # ----------------- Metric: topic relevance -----------------
    # topic tokens from tag + description
    tag_clean = re.sub(r"[#0-9]", " ", str(tag or "")).lower()
    desc_clean = str(description or "").lower()
    topic_tokens = set(
        t for t in re.findall(r"[a-z][a-z\-']+", tag_clean + " " + desc_clean)
        if len(t) > 3
    )

    if not sentences or not topic_tokens:
        rel_score = 0.5
        rel_notes = "Cannot compute topic relevance (no sentences or topic tokens); set to neutral 0.5."
    else:
        off_topic = 0
        for sent in sentences:
            tokens = set(re.findall(r"[a-z][a-z\-']+", sent.lower()))
            if not (tokens & topic_tokens):
                off_topic += 1
        rel_score = 1.0 - off_topic / max(1, len(sentences))
        rel_notes = f"{off_topic}/{len(sentences)} sentences appear off-topic (no overlap with tag/description tokens)."

    # ----------------- Metric: tone neutrality -----------------
    sensational = [
        "shocking", "disastrous", "explosive", "jaw-dropping", "devastating",
        "crazy", "insane", "mind-blowing", "unbelievable", "you won't believe",
        "!!!",
    ]
    hits = 0
    lower_all = (story_text or "").lower()
    for phrase in sensational:
        if phrase in lower_all:
            hits += 1
    tone_score = max(0.0, 1.0 - 0.2 * hits)
    if hits == 0:
        tone_notes = "No obvious clickbait / sensational language detected."
    else:
        tone_notes = f"Detected {hits} sensational phrase(s); tone score reduced."

    # ----------------- Metric: scene compliance -----------------
    if not scene_descs:
        sc_score = 1.0
        sc_notes = "No scene descriptions to validate; treating as compliant."
    else:
        bad = 0
        forbidden_scene_terms = [
            "text", "logo", "logos", "caption", "title", "poster", "sign",
            "subtitle", "headline", "watermark", "banner", "split-screen",
            "split screen"
        ]
        for d in scene_descs:
            words = re.findall(r"[A-Za-z][A-Za-z\-']+", d)
            if len(words) > 18:
                bad += 1
                continue
            lower_d = d.lower()
            if any(ft in lower_d for ft in forbidden_scene_terms):
                bad += 1
        sc_score = 1.0 - bad / max(1, len(scene_descs))
        sc_notes = f"{bad}/{len(scene_descs)} scenes violate length or forbidden-term constraints."

    # ----------------- Metric: entity consistency (NEW) -----------------
    full_name = _clean_tag_to_name(tag)
    if full_name and len(full_name.split()) >= 2:
        full_name_lc = full_name.lower()
        # Check if the full name appears in story text
        in_story = full_name_lc in story_lower
        # Or in any evidence title/description
        in_evidence = False
        for art in evidence_articles or []:
            t = (art.get("title") or "").lower()
            d = (art.get("description") or "").lower()
            if full_name_lc in t or full_name_lc in d:
                in_evidence = True
                break

        if in_story or in_evidence:
            ent_score = 1.0
            ent_notes = "Tag's full name appears in story or evidence."
        else:
            ent_score = 0.0
            ent_notes = "Tag looks like a person-style full name but never appears in story or evidence."
    else:
        ent_score = 0.5
        ent_notes = "Tag not clearly a multi-word name; entity check skipped (neutral 0.5)."

    # ----------------- Overall -----------------
    metrics = {
        "factual_coverage": {
            "score": round(cov_score, 3),
            "notes": cov_notes,
        },
        "factual_faithfulness": {
            "score": round(faith_score, 3),
            "notes": faith_notes,
        },
        "uncertainty_handling": {
            "score": round(unc_score, 3),
            "notes": unc_notes,
        },
        "topic_relevance": {
            "score": round(rel_score, 3),
            "notes": rel_notes,
        },
        "tone_neutrality": {
            "score": round(tone_score, 3),
            "notes": tone_notes,
        },
        "scene_compliance": {
            "score": round(sc_score, 3),
            "notes": sc_notes,
        },
        "format_correctness": {
            "score": round(fmt_score, 3),
            "notes": fmt_notes,
        },
        "entity_consistency": {
            "score": round(ent_score, 3),
            "notes": ent_notes,
        },
    }

    # Average all metric scores
    overall = sum(m["score"] for m in metrics.values()) / float(len(metrics))

    # Optional: if entity consistency is 0, hard-cap overall (wrong entity)
    if ent_score == 0.0:
        overall = min(overall, 0.4)

    metrics["overall_score"] = round(overall, 3)
    return metrics
