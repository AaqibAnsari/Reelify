import json
import os
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
        chunks = [text[i : i + max_length] for i in range(0, len(text), max_length)]
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


class LlamaStoryGenerator:
    """
    Ollama-local generator (WON'T run on Render Free).
    Keep it for local GPU only.
    """

    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"

        self._forbidden_terms = {
            "text", "logo", "logos", "caption", "title", "poster", "sign",
            "subtitle", "headline", "watermark", "split-screen", "split screen",
            "collage", "meme", "banner", "mirror", "duplicated", "duplicate",
            "mosaic", "tile", "tiling", "overlaid faces", "morphing faces"
        }

        self._stop = {
            "the","a","an","and","or","to","of","in","on","for","with","by","from","at","as",
            "this","that","these","those","it","its","is","was","were","are","be","been","being",
            "into","about","across","again","over","under","amid","among","between","during",
            "more","most","much","many","some","any","very","also","just","now","new","latest",
            "trend","trending","hashtag","social","media","online","debate","discussion"
        }

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

    def _call_ollama(self, prompt: str, max_tokens: int = 700, temperature: float = 0.7) -> str:
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
                    "Visual-only description for scene 1 (≤18 words).",
                    "Visual-only description for scene 2 (≤18 words).",
                    "Visual-only description for scene 3 (≤18 words).",
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
- Only state facts that appear in the EVIDENCE or are trivial background knowledge.
- If details are unclear, say so. Do NOT guess/speculate.
"""
        else:
            evidence_block = "EVIDENCE: (None available.)"
            evidence_rules = """
No external evidence is available:
- Keep it high-level; avoid precise numbers/dates.
- Do NOT guess.
"""

        return f"""You are a professional Pakistani news scriptwriter and storyboarder.

TOPIC: {tag}
CONTEXT: {description}

{evidence_block}

{evidence_rules}

Return ONLY valid JSON (UTF-8, no markdown, no backticks). Schema:
{{
  "story_paragraphs": string[2..3],
  "scenes": string[{num_scenes}]
}}

Scene rules:
- ≤ 18 words; visual-only
- No readable text/logos/signs
- Forbidden terms anywhere: {forbidden}

Example (format only):
{example}
"""

    @staticmethod
    def _trim_words(s: str, max_words: int = 18) -> str:
        tokens = s.strip().split()
        return " ".join(tokens[:max_words])

    def _sanitize_scene(self, s: str) -> str:
        s = s.replace("**", " ").strip()
        s = re.sub(r"^scene\s*\d+\s*:\s*", "", s, flags=re.IGNORECASE).strip()
        lower = s.lower()
        for t in self._forbidden_terms:
            lower = lower.replace(t, " ")
        lower = re.sub(r"\s+", " ", lower).strip()
        return self._trim_words(lower, 18)

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen, out = set(), []
        for it in items:
            k = it.lower().strip()
            if k and k not in seen:
                seen.add(k)
                out.append(it.strip())
        return out

    @staticmethod
    def _format_text_output(paragraphs: list[str], scenes: list[str]) -> str:
        story = "\n\n".join(p.strip() for p in paragraphs if p.strip())
        scene_lines = [f"Scene {i+1}: {scenes[i]}" for i in range(len(scenes))]
        return "# STORY\n" + story + "\n\n# SCENES\n" + "\n".join(scene_lines)

    def generate_story(self, tag, description, story_length="0.5-1 minute", num_scenes=4, evidence_articles=None, max_tokens=700, temperature=0.7):
        evidence_texts = None
        if evidence_articles:
            evidence_texts = []
            for art in evidence_articles[:4]:
                parts = [art.get("title",""), art.get("description",""), art.get("content_snippet","")]
                snippet = " ".join([p for p in parts if p]).strip()
                if snippet:
                    evidence_texts.append(snippet[:800])

        prompt = self.build_prompt(tag, "" if evidence_texts else description, story_length, num_scenes, evidence_texts)

        def _parse_json(text: str):
            try:
                return json.loads(text)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", text)
                return json.loads(m.group(0)) if m else None

        try:
            if not self.check_ollama_status():
                raise RuntimeError("Ollama is not running or model not available.")

            raw = self._call_ollama(prompt, max_tokens=max_tokens, temperature=temperature)
            data = _parse_json(raw)
            if not isinstance(data, dict):
                raise ValueError("Model did not return valid JSON.")

            paragraphs = data.get("story_paragraphs") or []
            scenes = data.get("scenes") or []

            if not isinstance(paragraphs, list):
                paragraphs = [str(paragraphs)]
            paragraphs = [re.sub(r"\s+", " ", str(p)).strip() for p in paragraphs if str(p).strip()]
            if len(paragraphs) < 2:
                paragraphs = [f"{tag}: {description}", "More details are still emerging."]

            if not isinstance(scenes, list):
                scenes = [str(scenes)]
            scenes = [self._sanitize_scene(str(s)) for s in scenes if str(s).strip()]
            scenes = [s for s in scenes if len(s.split()) >= 4]
            scenes = self._dedupe_preserve_order(scenes)[:num_scenes]

            while len(scenes) < num_scenes:
                scenes.append("close-up of hands on a desk, soft window light, background blurred")

            return self._format_text_output(paragraphs[:3], scenes[:num_scenes])
        except Exception as e:
            print(f"[LLaMA] generation error: {e}")
            paragraphs = [f"{tag}: {description}", "Developments continue to unfold; details remain unclear."]
            scenes = ["close-up of hands on a desk, soft window light, background blurred"] * num_scenes
            return self._format_text_output(paragraphs, scenes)


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

        self._forbidden_terms = {
            "text", "logo", "logos", "caption", "title", "poster", "sign",
            "subtitle", "headline", "watermark", "split-screen", "split screen",
            "collage", "meme", "banner", "mirror", "duplicated", "duplicate",
            "mosaic", "tile", "tiling", "overlaid faces", "morphing faces"
        }

        self._stop = {
            "the","a","an","and","or","to","of","in","on","for","with","by","from","at","as",
            "this","that","these","those","it","its","is","was","were","are","be","been","being",
            "into","about","across","again","over","under","amid","among","between","during",
            "more","most","much","many","some","any","very","also","just","now","new","latest",
            "trend","trending","hashtag","social","media","online","debate","discussion"
        }

        self.model_label = f"Gemini ({self.model_name})"

    @staticmethod
    def _json_example(num_scenes: int) -> str:
        return json.dumps(
            {
                "story_paragraphs": [
                    "Short, factual paragraph summarizing the topic.",
                    "Second paragraph with context and why it matters now."
                ],
                "scenes": [
                    "Visual-only description for scene 1 (≤18 words).",
                    "Visual-only description for scene 2 (≤18 words).",
                    "Visual-only description for scene 3 (≤18 words).",
                    f"Visual-only description for scene {num_scenes} (≤18 words)."
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
- Only state facts that appear in the EVIDENCE or are trivial background knowledge.
- If numbers/dates/causes are unclear, say they are unclear.
- Do NOT guess or speculate. Do NOT invent quotes/incidents.
"""
        else:
            evidence_block = "EVIDENCE: (No external articles available.)"
            evidence_rules = """
No external evidence is available:
- Keep it high-level and avoid precise numbers/dates.
- Do NOT pretend to know concrete details; say they are unclear.
"""

        return f"""You are a professional Pakistani news scriptwriter and storyboarder.

TOPIC: {tag}
CONTEXT: {description}

{evidence_block}

{evidence_rules}

Return ONLY valid JSON (UTF-8, no markdown, no backticks). Match exactly:
{{
  "story_paragraphs": string[2..3],
  "scenes": string[{num_scenes}]
}}

HARD RULES FOR SCENES:
- ≤ 18 words, visual-only (no narration/camera jargon)
- No readable text/logos/signs/posters/captions/banners
- Forbidden terms anywhere: {forbidden}
- Do NOT include 'Scene 1:' etc. inside the strings.

Example JSON (format only):
{example}
"""

    @staticmethod
    def _trim_words(s: str, max_words: int = 18) -> str:
        tokens = s.strip().split()
        return " ".join(tokens[:max_words])

    def _sanitize_scene(self, s: str) -> str:
        s = s.replace("**", " ").strip()
        s = re.sub(r"^scene\s*\d+\s*:\s*", "", s, flags=re.IGNORECASE).strip()
        lower = s.lower()
        for t in self._forbidden_terms:
            lower = lower.replace(t, " ")
        lower = re.sub(r"\s+", " ", lower).strip()
        lower = re.sub(r"\b(\w+)\s+\1\b", r"\1", lower, flags=re.IGNORECASE)
        return self._trim_words(lower, 18)

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen, out = set(), []
        for it in items:
            k = it.lower().strip()
            if k and k not in seen:
                seen.add(k)
                out.append(it.strip())
        return out

    @staticmethod
    def _format_text_output(paragraphs: list[str], scenes: list[str]) -> str:
        story = "\n\n".join(p.strip() for p in paragraphs if p.strip())
        scene_lines = [f"Scene {i+1}: {scenes[i]}" for i in range(len(scenes))]
        return "# STORY\n" + story + "\n\n# SCENES\n" + "\n".join(scene_lines)

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

        prompt = self.build_prompt(
            tag=tag,
            description="" if evidence_texts else description,
            story_length=story_length,
            num_scenes=num_scenes,
            evidence_texts=evidence_texts,
        )

        def _parse_json(text: str):
            try:
                return json.loads(text)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", text)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        return None
                return None

        try:
            resp = self.model.generate_content(prompt)
            raw = (resp.text or "").strip()
            data = _parse_json(raw)

            if not isinstance(data, dict) or "story_paragraphs" not in data or "scenes" not in data:
                # Repair pass
                repair_prompt = f"""Fix this to strict JSON only (no markdown, no commentary):

---- BEGIN ----
{raw}
---- END ----

Return JSON with keys:
- "story_paragraphs": string[2..3]
- "scenes": string[{num_scenes}]"""
                repair = self.model.generate_content(repair_prompt)
                data = _parse_json((repair.text or "").strip())

            if not isinstance(data, dict):
                raise ValueError("Model did not return valid JSON.")

            paragraphs = data.get("story_paragraphs") or []
            scenes = data.get("scenes") or []

            if not isinstance(paragraphs, list):
                paragraphs = [str(paragraphs)]
            paragraphs = [re.sub(r"\s+", " ", str(p)).strip() for p in paragraphs if str(p).strip()]
            if len(paragraphs) < 2:
                paragraphs = [f"{tag}: {description}", "More details are still emerging."]

            if not isinstance(scenes, list):
                scenes = [str(scenes)]
            scenes = [self._sanitize_scene(str(s)) for s in scenes if str(s).strip()]
            scenes = [s for s in scenes if s and len(s.split()) >= 4]
            scenes = self._dedupe_preserve_order(scenes)[:num_scenes]

            while len(scenes) < num_scenes:
                scenes.append("close-up of hands on a desk, soft window light, background blurred")

            return self._format_text_output(paragraphs[:3], scenes[:num_scenes])

        except Exception as e:
            print(f"[Gemini] generation error: {e}")
            paragraphs = [
                f"{tag}: {description}",
                "Developments continue to unfold; implications remain significant and some details are unclear.",
            ]
            scenes = ["close-up of hands on a desk, soft window light, background blurred"] * num_scenes
            return self._format_text_output(paragraphs, scenes)


def get_one_liner_for_trend(trend_name: str) -> str:
    if not GEMINI_API_KEY:
        return "Trending topic related to current events and social discussions."

    prompt = (
        f"Give a short one-liner explanation (max 30 words) of why "
        f"'{trend_name}' is trending on Twitter/X. Provide context for a user "
        f"who doesn't know about this trend."
    )
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        time.sleep(0.2)
        return (response.text or "").strip()
    except Exception as e:
        print(f"[Gemini] one-liner error: {e}")
        return "Trending topic related to current events and social discussions."
