import json
import os
import random
import re
import time
import requests

import google.generativeai as genai

# Heavy deps kept optional for Render
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

from .config import GEMINI_API_KEY

# Configure Gemini ONCE at import time (required for Render)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------
# Translator (unchanged)
# ---------------------------------------------------------------------
class FacebookUrduTranslator:
    """
    NOTE: This model is VERY heavy for free hosting. Keep it optional.
    If transformers isn't available or model can't load, it raises a clear error.
    """

    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.tokenizer = None
        self.model = None
        self.model_name = model_name
        self.src_lang = "eng_Latn"
        self.tgt_lang = "urd_Arab"
        self.is_loaded = False

    def load_model(self):
        if self.is_loaded:
            return

        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise RuntimeError(
                "transformers/torch not installed or not supported on this deployment. "
                "Disable Urdu translator for Render demo."
            )

        try:
            print("Loading translation model... (this can be slow / heavy)")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer.src_lang = self.src_lang
            self.lang_code_to_id = {
                "eng_Latn": self.tokenizer.convert_tokens_to_ids("eng_Latn"),
                "urd_Arab": self.tokenizer.convert_tokens_to_ids("urd_Arab"),
            }
            self.is_loaded = True
            print("Translation model loaded successfully!")
        except Exception as e:
            print(f"Error loading translation model: {e}")
            raise

    def translate(self, text: str) -> str:
        if not self.is_loaded:
            self.load_model()

        max_length = 400
        chunks = [text[i: i + max_length] for i in range(0, len(text), max_length)]
        translated_chunks = []

        for chunk in chunks:
            inputs = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            forced_bos_token_id = self.lang_code_to_id[self.tgt_lang]
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )
            translated_chunk = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]
            translated_chunks.append(translated_chunk)

        return " ".join(translated_chunks)


# ---------------------------------------------------------------------
# Shared helpers (stronger, like your “first code”)
# ---------------------------------------------------------------------
_FORBIDDEN_TERMS = {
    "text", "logo", "logos", "caption", "title", "poster", "sign",
    "subtitle", "headline", "watermark", "split-screen", "split screen",
    "collage", "meme", "banner", "mirror", "duplicated", "duplicate",
    "mosaic", "tile", "tiling", "overlaid faces", "morphing faces",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by",
    "from", "at", "as", "this", "that", "these", "those", "it", "its", "is",
    "was", "were", "are", "be", "been", "being", "into", "about", "across",
    "again", "over", "under", "amid", "among", "between", "during", "more",
    "most", "much", "many", "some", "any", "very", "also", "just", "now",
    "new", "latest", "trend", "trending", "hashtag", "social", "media",
    "online", "debate", "discussion",
}

# Diverse non-repeating fallback pool (seeded by topic)
_FALLBACK_SCENES_POOL = [
    "pakistani news anchor at desk, neutral studio lighting, shallow depth of field",
    "close-up of smartphone with blurred social feed, finger scrolling, background bokeh",
    "government press room podium, empty microphones, soft side lighting, plain background",
    "city street b-roll at dusk, traffic lights blurred, single subject in foreground",
    "hands holding printed briefing notes, no readable text, desk lamp glow, shallow focus",
    "courthouse exterior in daylight, people passing by, faces not visible, shallow focus",
    "police barrier tape in foreground, street scene blurred behind, late afternoon light",
    "meeting room table with folders and pens, soft window light, shallow depth of field",
]


def _json_extract(text: str):
    """Try strict JSON, else extract the first {...} block."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _trim_words(s: str, max_words: int = 18) -> str:
    tokens = (s or "").strip().split()
    if len(tokens) <= max_words:
        return " ".join(tokens)
    return " ".join(tokens[:max_words])


def _sanitize_scene(s: str, forbidden_terms=_FORBIDDEN_TERMS) -> str:
    s = (s or "").replace("**", " ").replace("—", "-").replace("–", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = s.strip().lstrip("-•*").strip()
    s = re.sub(r"^scene\s*\d+\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()

    lower = s.lower()
    for t in forbidden_terms:
        if t in lower:
            lower = lower.replace(t, " ")
    lower = " ".join(lower.split())
    lower = re.sub(r"\b(\w+)\s+\1\b", r"\1", lower, flags=re.IGNORECASE)
    lower = re.sub(r"\s+", " ", lower).strip()
    return _trim_words(lower, 18)


def _dedupe_preserve_order(items):
    seen, out = set(), []
    for it in items:
        k = (it or "").strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append((it or "").strip())
    return out


def _format_text_output(paragraphs, scenes) -> str:
    story = "\n\n".join(p.strip() for p in paragraphs if (p or "").strip())
    scene_lines = [f"Scene {i+1}: {scenes[i]}" for i in range(len(scenes))]
    return "# STORY\n" + story + "\n\n# SCENES\n" + "\n".join(scene_lines)


def _extract_keywords(text: str, k: int = 14):
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", (text or "").lower())
    words = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    proper = set(re.findall(r"\b([A-Z][a-zA-Z\-']+)\b", (text or "")))
    proper_l = {p.lower() for p in proper}

    scored = []
    for w, c in freq.items():
        bonus = 2 if w in proper_l else 0
        scored.append((-(c + bonus), w))
    scored.sort()
    return [w for _, w in scored[:k]]


def _story_based_fallback_scenes(story_text: str, num_needed: int) -> list[str]:
    """
    Same idea as your “first code”: keyword-based grounded scenes, not generic repeats.
    """
    kws = _extract_keywords(story_text, k=16)
    kwset = set(kws)

    def has_any(*cands):
        return any(c in kwset for c in cands)

    candidates = []

    if has_any("kabul", "afghan", "afghanistan", "taliban", "border"):
        candidates += [
            "checkpoint barrier on a dusty road at dusk, lone guard silhouette, shallow focus",
            "deserted government hallway, open door, soft window light, quiet atmosphere",
        ]
    if has_any("military", "soldier", "uniform", "army"):
        candidates += [
            "folded military uniform and helmet on a bench, soft light, shallow depth of field",
            "empty briefing room with chairs askew, projector off, subtle dust motes",
        ]
    if has_any("court", "trial", "judge", "case", "hearing"):
        candidates += [
            "wooden gavel beside closed case file on desk, dramatic side light, shallow focus",
            "empty courtroom aisle leading to bench, morning light through blinds",
        ]
    if has_any("rally", "crowd", "protest", "supporters"):
        candidates += [
            "raised hands in a crowd, shallow focus, no signs or text, background blurred",
            "close-up of a face in crowd looking upward, background softly blurred",
        ]
    if has_any("corruption", "aid", "funds", "money"):
        candidates += [
            "hands counting worn banknotes on a desk, selective focus, background blurred",
            "stack of unlabeled folders with paper edges visible, desk lamp glow, shallow depth",
        ]
    if has_any("collapse", "fall", "takeover", "retreat"):
        candidates += [
            "abandoned checkpoint booth with chair pushed back, late afternoon light",
            "discarded boots on a dusty floor near doorway, slanting sunbeams",
        ]
    if has_any("hashtag", "trend", "trending", "twitter", "social", "x"):
        candidates += [
            "thumb flicks through blurred social app feed on phone, selective focus on fingertip",
        ]

    candidates += [
        "close-up of thoughtful eyes in profile, soft window light, plain background",
        "hand points at a blurred unlabeled regional map on table, shallow depth of field",
    ]

    clean = [_sanitize_scene(c) for c in candidates]
    clean = _dedupe_preserve_order(clean)
    return clean[:max(0, num_needed)]


# ---------------------------------------------------------------------
# LlamaStoryGenerator (prompt upgraded to match “first code” strength)
# ---------------------------------------------------------------------
class LlamaStoryGenerator:
    """
    Ollama-local generator (WON'T run on Render Free).
    Kept for local GPU only.
    """

    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"
        self._forbidden_terms = set(_FORBIDDEN_TERMS)
        self._stop = set(_STOPWORDS)
        self.model_label = f"LLaMA via Ollama ({self.model_name})"

    def check_ollama_status(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                return self.model_name in model_names
        except Exception:
            pass
        return False

    def _call_ollama(self, prompt: str, max_tokens: int = 900, temperature: float = 0.65) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
        }
        resp = requests.post(self.api_url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()

    @staticmethod
    def _json_example(num_scenes: int) -> str:
        return json.dumps(
            {
                "story_paragraphs": [
                    "Short, conversational paragraph summarizing the topic.",
                    "Second paragraph with key context and what it means now.",
                ],
                "scenes": [
                    f"Visual-only description for scene 1 (≤18 words).",
                    f"Visual-only description for scene 2 (≤18 words).",
                    f"Visual-only description for scene 3 (≤18 words).",
                    f"Visual-only description for scene {num_scenes} (≤18 words).",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    def build_prompt(self, tag, description, story_length="0.5-1 minute", num_scenes=4, evidence_texts=None):
        example = self._json_example(num_scenes)
        forbidden = ", ".join(sorted(self._forbidden_terms))

        if evidence_texts:
            joined_evidence = "\n\n---\n\n".join(evidence_texts[:3])
            evidence_block = f"""EVIDENCE (verbatim excerpts from recent news articles about the topic):

{joined_evidence}
"""
            evidence_rules = """
You MUST treat the EVIDENCE as ground truth for factual details.

RULES FOR FACTUAL CONTENT:
- Only state facts that appear in the EVIDENCE or are trivial background knowledge.
- If numbers/dates/causes are not clearly in the EVIDENCE, say they are unclear or still emerging.
- Do NOT guess or speculate. Do NOT invent quotes or specific incidents that are not mentioned.
- If the EVIDENCE conflicts, mention that there are conflicting reports.
"""
        else:
            evidence_block = (
                "EVIDENCE: (No external articles available; base the summary only on "
                "the TOPIC and CONTEXT, avoid precise numbers/dates.)"
            )
            evidence_rules = """
No external evidence is available. You MUST:
- Keep the explanation high-level and avoid specific numbers, death/injury counts, or strict timelines.
- Do NOT pretend to know concrete details; say they are unclear instead of guessing.
"""

        # Strong grounding + anti-generic rules (same style as your “first code”)
        return f"""You are a professional Pakistani news scriptwriter and storyboarder.

TOPIC: {tag}
CONTEXT: {description}

{evidence_block}

{evidence_rules}

Return ONLY valid JSON (UTF-8, no markdown, no backticks). Match exactly this schema:
{{
  "story_paragraphs": string[2..3],   // 2–3 short paragraphs, total ≈ {story_length}
  "scenes": string[{num_scenes}]      // EXACTLY {num_scenes} items
}}

HARD RULES FOR STORY:
- Neutral, factual tone, like a news explainer.
- Do not dramatize or add fictional dialogue.
- Explicitly flag uncertain points as "unclear" or "unconfirmed" if the evidence is weak.
- Paragraph structure:
  * Paragraph 1: what the topic is + what is confirmed.
  * Paragraph 2: why it is trending / why it matters now.
  * Paragraph 3 (optional): what to watch next (only if supported or clearly marked as unclear).

HARD RULES FOR SCENES (must all be enforced):
- Ground every scene in the factual situation; do NOT invent unrelated objects or metaphors.
- ≤ 18 words, visual-only (no narration/camera jargon).
- Vertical-friendly composition; medium/tight framing; subject centered.
- Shallow depth of field; simple backgrounds.
- No readable text/logos/signs/posters/captions/banners. Avoid crowds full of small signs.
- No collages, tiling, overlays, morphing faces, or symbolic icon mashups.
- Vary subject, angle, lighting, and depth; keep concrete and news-relevant.
- Forbidden terms anywhere: {forbidden}
- Do NOT include 'Scene 1:' etc. inside the strings.

Example JSON (format only, not content):
{example}

Now produce the JSON for the given TOPIC, CONTEXT and EVIDENCE with your original content ONLY.
"""

    def generate_story(
        self,
        tag,
        description,
        story_length="0.5-1 minute",
        num_scenes=4,
        evidence_articles=None,
        max_tokens=900,
        temperature=0.65,
    ):
        evidence_texts = None
        if evidence_articles:
            evidence_texts = []
            for art in evidence_articles[:4]:
                parts = [art.get("title", ""), art.get("description", ""), art.get("content_snippet", "")]
                snippet = " ".join([p for p in parts if p]).strip()
                if snippet:
                    evidence_texts.append(snippet[:800])

        # IMPORTANT: DO NOT starve prompt; keep short context even with evidence
        desc_for_prompt = _clean_ws(description)[:220]
        prompt = self.build_prompt(
            tag=tag,
            description=desc_for_prompt,
            story_length=story_length,
            num_scenes=num_scenes,
            evidence_texts=evidence_texts,
        )

        try:
            if not self.check_ollama_status():
                raise RuntimeError("Ollama is not running or model not available.")

            raw = self._call_ollama(prompt, max_tokens=max_tokens, temperature=temperature)
            data = _json_extract(raw)

            # One repair pass if needed
            if not isinstance(data, dict) or "story_paragraphs" not in data or "scenes" not in data:
                repair_prompt = f"""You failed to return valid JSON in the required schema.

REPAIR THIS to strict JSON only (no markdown, no commentary). Keep content, just fix format:

---- BEGIN MODEL OUTPUT ----
{raw}
---- END MODEL OUTPUT ----

Return JSON with keys:
- "story_paragraphs": string[2..3]
- "scenes": string[{num_scenes}]"""
                repair_raw = self._call_ollama(repair_prompt, max_tokens=450, temperature=0.2)
                data = _json_extract(repair_raw)

            if not isinstance(data, dict):
                raise ValueError("Model did not return valid JSON.")

            paragraphs = data.get("story_paragraphs") or []
            scenes = data.get("scenes") or []

            if not isinstance(paragraphs, list):
                paragraphs = [str(paragraphs)]
            paragraphs = [_clean_ws(str(p)) for p in paragraphs if _clean_ws(str(p))]
            paragraphs = paragraphs[:3]
            if len(paragraphs) < 2:
                paragraphs = [f"{tag}: {description}", "More details are still emerging; some points remain unclear."]

            if not isinstance(scenes, list):
                scenes = [str(scenes)]
            scenes = [_sanitize_scene(str(s), self._forbidden_terms) for s in scenes if _clean_ws(str(s))]
            scenes = [s for s in scenes if s and len(s.split()) >= 4]
            scenes = _dedupe_preserve_order(scenes)[:num_scenes]

            if len(scenes) < num_scenes:
                story_text = f"{tag}\n{description}\n" + "\n".join(paragraphs)
                scenes.extend(_story_based_fallback_scenes(story_text, num_scenes - len(scenes)))
                scenes = _dedupe_preserve_order(scenes)[:num_scenes]

            # final fill (seeded; non-repeating)
            rng = random.Random(hash(tag) & 0xFFFFFFFF)
            while len(scenes) < num_scenes:
                scenes.append(rng.choice(_FALLBACK_SCENES_POOL))
            scenes = scenes[:num_scenes]

            return _format_text_output(paragraphs, scenes)

        except Exception as e:
            print(f"[LLaMA] generation error: {e}")
            paragraphs = [f"{tag}: {description}", "Developments continue to unfold; details remain unclear."]
            story_text = f"{tag}\n{description}\n" + "\n".join(paragraphs)
            scenes = _story_based_fallback_scenes(story_text, num_scenes)
            rng = random.Random(hash(tag) & 0xFFFFFFFF)
            while len(scenes) < num_scenes:
                scenes.append(rng.choice(_FALLBACK_SCENES_POOL))
            return _format_text_output(paragraphs, scenes[:num_scenes])


# ---------------------------------------------------------------------
# GeminiStoryGenerator (prompt upgraded to match “first code” strength)
# ---------------------------------------------------------------------
class GeminiStoryGenerator:
    """
    Gemini generator (Render-friendly).
    IMPORTANT: Requires GEMINI_API_KEY env var + genai.configure(api_key=...)
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing. Set it in Render Environment variables.")
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self._forbidden_terms = set(_FORBIDDEN_TERMS)
        self._stop = set(_STOPWORDS)
        self.model_label = f"Gemini ({self.model_name})"

    @staticmethod
    def _json_example(num_scenes: int) -> str:
        return json.dumps(
            {
                "story_paragraphs": [
                    "Short, conversational paragraph summarizing the topic.",
                    "Second paragraph with key context and what it means now.",
                ],
                "scenes": [
                    f"Visual-only description for scene 1 (≤18 words).",
                    f"Visual-only description for scene 2 (≤18 words).",
                    f"Visual-only description for scene 3 (≤18 words).",
                    f"Visual-only description for scene {num_scenes} (≤18 words).",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    def build_prompt(self, tag, description, story_length="0.5-1 minute", num_scenes=4, evidence_texts=None):
        example = self._json_example(num_scenes)
        forbidden = ", ".join(sorted(self._forbidden_terms))

        if evidence_texts:
            joined_evidence = "\n\n---\n\n".join(evidence_texts[:3])
            evidence_block = f"""EVIDENCE (verbatim excerpts from recent news articles about the topic):

{joined_evidence}
"""
            evidence_rules = """
You MUST treat the EVIDENCE as ground truth for factual details.

RULES FOR FACTUAL CONTENT:
- Only state facts that appear in the EVIDENCE or are trivial background knowledge.
- If numbers/dates/causes are not clearly in the EVIDENCE, say they are unclear or still emerging.
- Do NOT guess or speculate. Do NOT invent quotes or specific incidents that are not mentioned.
- If the EVIDENCE conflicts, mention that there are conflicting reports.
"""
        else:
            evidence_block = (
                "EVIDENCE: (No external articles available; base the summary only on "
                "the TOPIC and CONTEXT, avoid precise numbers/dates.)"
            )
            evidence_rules = """
No external evidence is available. You MUST:
- Keep the explanation high-level and avoid specific numbers, death/injury counts, or strict timelines.
- Do NOT pretend to know concrete details; say they are unclear instead of guessing.
"""

        return f"""You are a professional Pakistani news scriptwriter and storyboarder.

TOPIC: {tag}
CONTEXT: {description}

{evidence_block}

{evidence_rules}

Return ONLY valid JSON (UTF-8, no markdown, no backticks). Match exactly this schema:
{{
  "story_paragraphs": string[2..3],   // 2–3 short paragraphs, total ≈ {story_length}
  "scenes": string[{num_scenes}]      // EXACTLY {num_scenes} items
}}

HARD RULES FOR STORY:
- Neutral, factual tone, like a news explainer.
- Do not dramatize or add fictional dialogue.
- Explicitly flag uncertain points as "unclear" or "unconfirmed" if the evidence is weak.
- Paragraph structure:
  * Paragraph 1: what the topic is + what is confirmed (use EVIDENCE if provided).
  * Paragraph 2: why it is trending / why it matters now.
  * Paragraph 3 (optional): what to watch next (only if supported, otherwise mark as unclear).

HARD RULES FOR SCENES (must all be enforced):
- Ground every scene in the factual situation; do NOT invent unrelated objects or metaphors.
- ≤ 18 words, visual-only (no narration/camera jargon).
- Vertical-friendly composition; medium/tight framing; subject centered.
- Shallow depth of field; simple backgrounds.
- No readable text/logos/signs/posters/captions/banners. Avoid crowds full of small signs.
- No collages, tiling, overlays, morphing faces, or symbolic icon mashups.
- Vary subject, angle, lighting, and depth; keep concrete and news-relevant.
- Forbidden terms anywhere: {forbidden}
- Do NOT include 'Scene 1:' etc. inside the strings.

Example JSON (format only, not content):
{example}

Now produce the JSON for the given TOPIC, CONTEXT and EVIDENCE with your original content ONLY.
"""

    def generate_story(self, tag, description, story_length="0.5-1 minute", num_scenes=4, evidence_articles=None):
        evidence_texts = None
        if evidence_articles:
            evidence_texts = []
            for art in evidence_articles[:4]:
                parts = []
                if art.get("title"):
                    parts.append(art["title"])
                if art.get("description"):
                    parts.append(art["description"])
                if art.get("content_snippet"):
                    parts.append(art["content_snippet"])
                snippet = " ".join(parts).strip()
                if snippet:
                    evidence_texts.append(snippet[:800])

        # IMPORTANT: DO NOT starve prompt; keep short context even with evidence
        desc_for_prompt = _clean_ws(description)[:220]

        prompt = self.build_prompt(
            tag=tag,
            description=desc_for_prompt,
            story_length=story_length,
            num_scenes=num_scenes,
            evidence_texts=evidence_texts,
        )

        try:
            resp = self.model.generate_content(prompt)
            raw = (resp.text or "").strip()
            data = _json_extract(raw)

            # One repair pass if schema is wrong
            if not isinstance(data, dict) or "story_paragraphs" not in data or "scenes" not in data:
                repair_prompt = f"""You failed to return valid JSON in the required schema.

REPAIR THIS to strict JSON only (no markdown, no commentary). Keep content, just fix format:

---- BEGIN MODEL OUTPUT ----
{raw}
---- END MODEL OUTPUT ----

Return JSON with keys:
- "story_paragraphs": string[2..3]
- "scenes": string[{num_scenes}]"""
                repair = self.model.generate_content(repair_prompt)
                data = _json_extract((repair.text or "").strip())

            if not isinstance(data, dict):
                raise ValueError("Model did not return valid JSON.")

            paragraphs = data.get("story_paragraphs") or []
            scenes = data.get("scenes") or []

            if not isinstance(paragraphs, list):
                paragraphs = [str(paragraphs)]
            paragraphs = [_clean_ws(str(p)) for p in paragraphs if _clean_ws(str(p))]
            paragraphs = paragraphs[:3]
            if len(paragraphs) < 2:
                paragraphs = [f"{tag}: {description}", "More details are still emerging; some points remain unclear."]

            if not isinstance(scenes, list):
                scenes = [str(scenes)]
            scenes = [_sanitize_scene(str(s), self._forbidden_terms) for s in scenes if _clean_ws(str(s))]
            scenes = [s for s in scenes if s and len(s.split()) >= 4]
            scenes = _dedupe_preserve_order(scenes)[:num_scenes]

            # If still short, add grounded fallback based on story keywords (like first code)
            if len(scenes) < num_scenes:
                story_text = f"{tag}\n{description}\n" + "\n".join(paragraphs)
                scenes.extend(_story_based_fallback_scenes(story_text, num_scenes - len(scenes)))
                scenes = _dedupe_preserve_order(scenes)[:num_scenes]

            # Final fill: seeded, non-repeating pool (better than same line)
            rng = random.Random(hash(tag) & 0xFFFFFFFF)
            while len(scenes) < num_scenes:
                scenes.append(rng.choice(_FALLBACK_SCENES_POOL))
            scenes = scenes[:num_scenes]

            return _format_text_output(paragraphs, scenes)

        except Exception as e:
            print(f"[Gemini] generation error: {e}")
            paragraphs = [
                f"{tag}: {description}",
                "Developments continue to unfold; implications remain significant and some details are unclear.",
            ]
            story_text = f"{tag}\n{description}\n" + "\n".join(paragraphs)
            scenes = _story_based_fallback_scenes(story_text, num_scenes)
            rng = random.Random(hash(tag) & 0xFFFFFFFF)
            while len(scenes) < num_scenes:
                scenes.append(rng.choice(_FALLBACK_SCENES_POOL))
            return _format_text_output(paragraphs, scenes[:num_scenes])


# ---------------------------------------------------------------------
# One-liner (unchanged, minor safety)
# ---------------------------------------------------------------------
def get_one_liner_for_trend(trend_name: str) -> str:
    if not GEMINI_API_KEY:
        return "Trending topic related to current events and social discussions."

    prompt = (
        f"Write ONE neutral one-liner (max 25 words) explaining why '{trend_name}' is trending. "
        f"If unsure, say 'people are discussing it online' without inventing facts."
    )
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        time.sleep(0.2)
        return (response.text or "").strip()
    except Exception as e:
        print(f"[Gemini] one-liner error: {e}")
        return "Trending topic related to current events and social discussions."
