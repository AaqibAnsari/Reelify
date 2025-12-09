import base64
import gc
import os
import re
import shutil
import tempfile
import threading
import time
from io import BytesIO

import torch
from PIL import Image
from flask import session
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline, StableVideoDiffusionPipeline, StableDiffusionPipeline
from diffusers.utils import export_to_video
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip

try:
    from gradio_client import Client, handle_file
except Exception:
    Client = None
    handle_file = None

from .config import IMG_DIR, VID_DIR


def _device_and_dtype():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if use_cuda else torch.float32
    return device, dtype

_cuda_lock = threading.Lock()

def exclusive_cuda():
    """Context manager to serialize CUDA operations and prevent concurrent VRAM usage"""
    return _cuda_lock


def free_cuda():
    """Force-free CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()


class SDXLImageGenerator:
    """
    SD 1.5 text-to-image, VRAM-friendly.
    Anti-duplicate tuned for social-reel scenes: solo subject, tight framing, no readable text.
    """
    def __init__(self, prefer_turbo: bool = False):
        self.prefer_turbo = prefer_turbo
        self.pipe = None
        self.model_id = "runwayml/stable-diffusion-v1-5"

        # Compact, CLIP-safe negatives (keep short!)
        self.NEGATIVE_BASE = (
            "text, logo, watermark, caption, sign, poster, subtitles, "
            "blurry, noisy, low quality, jpeg artifacts, oversharpened, "
            "bad anatomy, bad hands, extra fingers, deformed, melted face"
        )
        # Extra anti-duplicate blockers
        self.NEGATIVE_DEDUPE = (
            "duplicate, duplicates, clone, twins, double exposure, multi-face, "
            "extra head, multiple main subjects, collage, tiled, mosaic, grid, pattern, checkerboard, kaleidoscope, split screen, mirror"
        )

        # Composition helpers
        self.DEDUPE_SUFFIX = "solo subject, shallow depth of field, no duplicates, no mirror, no collage"
        self.BASE_PREFIX = "cinematic reels b-roll, vertical friendly, medium/tight shot, centered subject, realistic, high detail, no text"

        self._CLIP_MAX_TOKENS = 74  # leave a little slack (<77 hard cap)

        # Common nouns that often get duplicated – force them to be singular
        self._SINGULAR_HINTS = [
            "speaker", "person", "man", "woman", "face", "portrait", "presenter", "anchor",
            "gavel", "judge", "microphone", "mic", "camera", "podium", "flag", "phone",
            "capitol", "white house", "building", "statue"
        ]

    # ---------- utils ----------
    @staticmethod
    def _clean(s: str) -> str:
        return (s or "").replace("**", "").replace('"', "").replace("'", "").strip()

    @staticmethod
    def _words(s: str) -> list:
        import re
        return [w for w in re.split(r"\s+", (s or "").strip()) if w]

    def _clip_shorten(self, s: str, limit: int = None) -> str:
        if not s:
            return s
        limit = limit or self._CLIP_MAX_TOKENS
        w = self._words(s)
        if len(w) <= limit:
            return s.strip()
        return " ".join(w[:limit]).strip()

    def _enforce_single_subject(self, desc: str) -> str:
        d = desc.lower()
        # If crowd terms appear, force single foreground + bokeh background
        if any(k in d for k in ["crowd", "rally", "audience", "supporters", "press", "cameras"]):
            if "single" not in d and "solo" not in d and "one " not in d:
                desc += ", single foreground subject"
            if "bokeh" not in d and "blur" not in d:
                desc += ", background blurred"
        # Force singular for common duplicate-prone nouns
        if not any(k in d for k in ["single", "solo", "one "]):
            for noun in self._SINGULAR_HINTS:
                if noun in d:
                    desc += ", single " + noun
                    break
        return desc

    def _compose_prompt(self, scene_desc: str, base_prefix: str = None) -> str:
        base = base_prefix or self.BASE_PREFIX
        desc = self._enforce_single_subject(self._clean(scene_desc))
        prompt = f"{base}, {desc}, {self.DEDUPE_SUFFIX}"
        return self._clip_shorten(prompt)

    def _compose_negative(self, extra_negative: str = None) -> str:
        neg = ", ".join([self.NEGATIVE_BASE, self.NEGATIVE_DEDUPE, self._clean(extra_negative or "")])
        return self._clip_shorten(neg)

    # ---------- lifecycle ----------
    def _load(self):
        if self.pipe is not None:
            return
        device, dtype = _device_and_dtype()
        try:
            print(f"[SD15] Loading {self.model_id} with memory optimizations...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None
            ).to(device)
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("[SD15] xFormers memory efficient attention enabled.")
            except Exception:
                print("[SD15] xFormers not available; continuing without it.")
            print("[SD15] Model loaded successfully.")
        except Exception as e:
            print(f"[SD15] Load error: {e}")
            self.pipe = None

    def unload(self):
        if self.pipe is not None:
            print("[SD15] Unloading model from GPU...")
            try:
                del self.pipe
            finally:
                self.pipe = None
                free_cuda()
            print("[SD15] Model unloaded successfully.")

    def available(self) -> bool:
        self._load()
        return self.pipe is not None

    # ---------- single ----------
    def generate(self, prompt: str, negative_prompt: str = None,
                 width: int = 768, height: int = 768,
                 steps: int = 30, guidance_scale: float = 6.8,
                 seed: int = None) -> str:
        """Single image with anti-duplicate suffix & CLIP-safe prompts."""
        with exclusive_cuda():
            self._load()
            if not self.pipe:
                raise RuntimeError("Text-to-Image pipeline not available.")

            gen = torch.Generator(device=str(_device_and_dtype()[0]))
            if seed is not None:
                gen.manual_seed(seed)

            prompt_full = self._compose_prompt(prompt)
            neg_full = self._compose_negative(negative_prompt)

            image = self.pipe(
                prompt=prompt_full,
                negative_prompt=neg_full,
                width=width, height=height,
                num_inference_steps=max(20, steps),
                guidance_scale=max(5.5, guidance_scale),
                generator=gen,
            ).images[0]

            ts = int(time.time() * 1000)
            filename = f"t2i_{ts}.png"
            out_path = os.path.join(IMG_DIR, filename)
            image.save(out_path)
            rel = f"/static/outputs/images/{filename}"
            session['last_generated_image'] = rel
            return rel

    # ---------- batch ----------
    def generate_multiple_scenes(self, story: str, count: int = 4,
                                 width: int = 1024, height: int = 1024,
                                 steps: int = 35, guidance_scale: float = 7.2,
                                 seed: int = None, base_prompt_prefix: str = None) -> list:
        """Batch images with the same anti-duplicate treatment."""
        with exclusive_cuda():
            self._load()
            if not self.pipe:
                raise RuntimeError("Text-to-Image pipeline not available.")

            scene_prompts = self._extract_scene_prompts(story, count, base_prompt_prefix)
            neg_full = self._compose_negative()
            gen = torch.Generator(device=str(_device_and_dtype()[0]))
            if seed is not None:
                gen.manual_seed(seed)

            results = []
            for i, scene_desc in enumerate(scene_prompts):
                full = self._compose_prompt(scene_desc, base_prompt_prefix)
                print(f"[SD15] Generating scene {i+1}/{count}: {full[:120]}...")
                scene_gen = gen if seed else torch.Generator(device=str(_device_and_dtype()[0])).manual_seed(1000*(i+1)+17)

                img = self.pipe(
                    prompt=full,
                    negative_prompt=neg_full,
                    width=width, height=height,
                    num_inference_steps=max(20, steps),
                    guidance_scale=max(5.5, guidance_scale),
                    generator=scene_gen,
                ).images[0]

                ts = int(time.time()*1000)
                filename = f"scene_{i+1}_{ts}.png"
                img.save(os.path.join(IMG_DIR, filename))
                results.append(f"/static/outputs/images/{filename}")

            session['last_generated_images'] = results
            session['last_generated_image'] = results[0] if results else None
            return results

    # ---------- scene extraction (unchanged semantics; trimmed outputs) ----------
    def _extract_scene_prompts(self, story: str, count: int, base_prompt_prefix: str = None) -> list:
        import re, google.generativeai as genai
        lines = re.findall(r"Scene\s*\d+\s*:\s*(.+)", story, flags=re.IGNORECASE)
        scenes = [self._clean(x) for x in lines if self._clean(x)]

        if len(scenes) < count:
            try:
                model = genai.GenerativeModel("gemini-2.0-flash")
                g_prompt = f"""
Create exactly {count} short visual-only scene lines (<= 16 words).
- Vertical-friendly, centered subject, medium/tight shot
- NO readable text/logos/signs
- Single clear foreground subject; if crowd, keep crowd blurred in background
- Don't include prompts that include closeups of humans/animals.

Story:
{story}

Return only:
Scene 1: <desc>
...
Scene {count}: <desc>
"""
                resp = model.generate_content(g_prompt)
                text = (resp.text or "").strip()
                more = re.findall(r"Scene\s*\d+\s*:\s*(.+)", text, flags=re.IGNORECASE)
                scenes.extend([self._clean(x) for x in more if self._clean(x)])
            except Exception as e:
                print(f"[SD15] Scene extraction via Gemini failed: {e}")

        while len(scenes) < count:
            scenes.append("single person, medium shot, neutral background, natural light")
        return scenes[:count]


class SVDImageToVideo:
    """
    Image-to-Video (low-VRAM tuned).
    Uses the base SVD img2vid checkpoint (lower VRAM than '-xt') plus aggressive offloading.

    Tips for even lower VRAM:
      - Decrease target_w/target_h (e.g., 640x360)
      - Reduce num_frames (e.g., 10–12)
      - Keep dtype=float16 on CUDA, and always offload when possible
    """
    def __init__(self):
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return
        device, dtype = _device_and_dtype()
        try:
            # Lower-VRAM checkpoint (base, not "-xt")
            model_id = "stabilityai/stable-video-diffusion-img2vid"

            from diffusers import StableVideoDiffusionPipeline
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant="fp16" if (dtype == torch.float16) else None,
            )

            # Aggressive memory savers — these keep VRAM low
            try:
                # Offload to CPU as layers are not in use (best VRAM saver)
                self.pipe.enable_model_cpu_offload()
            except Exception:
                # Fallback: sequential offload (still good)
                try:
                    self.pipe.enable_sequential_cpu_offload()
                except Exception:
                    pass

            # (Some pipelines expose VAE helpers; if available, enable)
            if hasattr(self.pipe, "enable_vae_slicing"):
                try:
                    self.pipe.enable_vae_slicing()
                except Exception:
                    pass
            if hasattr(self.pipe, "enable_vae_tiling"):
                try:
                    self.pipe.enable_vae_tiling()
                except Exception:
                    pass

            # Move to device (offload will still keep layers mostly on CPU)
            self.pipe = self.pipe.to(device)
        except Exception as e:
            print(f"[SVD] Load error: {e}")
            self.pipe = None

    def available(self) -> bool:
        self._load()
        return self.pipe is not None

    def unload(self):
        """Unload the model from GPU memory"""
        if self.pipe is not None:
            print("[SVD] Unloading model from GPU...")
            # Move to CPU and delete
            try:
                self.pipe = self.pipe.to("cpu")
            except Exception:
                pass
            del self.pipe
            self.pipe = None
            free_cuda()
            print("[SVD] Model unloaded successfully")

    def _load_image(self, src: str) -> Image.Image:
        if src.startswith("data:image"):
            header, b64 = src.split(",", 1)
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            return img
        if src.startswith("/static/"):
            path = os.path.join(os.path.dirname(__file__), src.lstrip("/"))
            return Image.open(path).convert("RGB")
        # otherwise assume file path
        return Image.open(src).convert("RGB")

    def _fit_for_svd(self, image: Image.Image, target_w=768, target_h=432) -> Image.Image:
        """
        Fit image into a smaller 16:9 canvas (lower VRAM than 1024x576).
        Keep aspect; letterbox/pad if needed.
        """
        # Resize while preserving aspect to fit inside target
        src_w, src_h = image.size
        target_aspect = target_w / target_h
        src_aspect = src_w / src_h

        if src_aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / src_aspect)
        else:
            new_h = target_h
            new_w = int(target_h * src_aspect)

        resized = image.resize((max(8, new_w // 8 * 8), max(8, new_h // 8 * 8)), Image.LANCZOS)

        # Paste centered on black canvas (ensures exact dims divisible by 8)
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        x = (target_w - resized.width) // 2
        y = (target_h - resized.height) // 2
        canvas.paste(resized, (x, y))
        return canvas

    def generate(self, image_source: str, fps: int = 7,
                 motion_bucket_id: int = 127, noise_aug_strength: float = 0.02,
                 num_frames: int = 12) -> str:
        """
        Defaults tuned for lower VRAM:
          - target 768x432 instead of 1024x576
          - 12 frames instead of 14
        """
        with exclusive_cuda():
            self._load()
            if not self.pipe:
                raise RuntimeError("Image-to-Video pipeline not available.")

            # Load and downscale for lower VRAM
            image = self._load_image(image_source)
            image = self._fit_for_svd(image, target_w=768, target_h=432)  # smaller than 1024x576

            # Inference (disable grads)
            torch.set_grad_enabled(False)

            result = self.pipe(
                image=image,
                decode_chunk_size=8,        # keep decode memory small
                fps=int(fps),
                motion_bucket_id=int(motion_bucket_id),
                noise_aug_strength=float(noise_aug_strength),
                num_frames=int(num_frames),
            )

            frames = result.frames[0]  # list of PIL images
            ts = int(time.time()*1000)
            filename = f"i2v_{ts}.mp4"
            out_path = os.path.join(VID_DIR, filename)
            export_to_video(frames, out_path, fps=fps)
            rel = f"/static/outputs/videos/{filename}"
            session['last_generated_video'] = rel
            return rel


class HFSpaceI2VClient:
    """
    Image-to-Video via a public Hugging Face Space (no local GPU required).
    Default Space: 'multimodalart/stable-video-diffusion' (SVD img2vid-xt demo).

    Notes:
      - Uses gradio_client to call the Space API (api_name='video').
      - You can pass /static/... paths or data:image/... base64 strings.
      - The returned MP4 is saved into VID_DIR and a relative URL is returned.
    """
    def __init__(self, space_id: str = "multimodalart/stable-video-diffusion"):
        self.space_id = space_id
        self.client = None

    def _ensure_client(self):
        if Client is None:
            raise RuntimeError(
                "gradio_client not installed. Run: pip install --upgrade gradio_client"
            )
        if self.client is None:
            # If the Space is private, set HF_TOKEN in env and pass hf_token=...
            self.client = Client(self.space_id)  # public Space; no token required

    def _prepare_image_file(self, image_source: str) -> str:
        # Accept data URLs or local/static paths
        if image_source.startswith("data:image"):
            header, b64 = image_source.split(",", 1)
            suffix = ".png" if "png" in header else ".jpg"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(base64.b64decode(b64))
            tmp.flush(); tmp.close()
            return tmp.name
        if image_source.startswith("/static/"):
            return os.path.join(os.path.dirname(__file__), image_source.lstrip("/"))
        return image_source  # absolute/relative file path

    def generate(self, image_source: str, fps: int = 7,
                 motion_bucket_id: int = 127, noise_aug_strength: float = 0.02,
                 num_frames: int = 14) -> str:
        """
        The Space exposes (image, seed, randomize_seed, motion_bucket_id, fps_id)
        via api_name='video'. We map our arguments to the Space’s expected inputs.

        NOTE: num_frames is not exposed by this Space; clip length depends on fps.
        """
        self._ensure_client()
        src = self._prepare_image_file(image_source)

        # Call the Space function by name ('video'); see its app code. 
        # Predict returns [video_file, seed] or a single file path depending on gradio version.
        result = self.client.predict(
            api_name="video",
            image=handle_file(src),
            seed=0,                   # fixed seed for reproducibility
            randomize_seed=False,     # set True to vary motion each call
            motion_bucket_id=int(motion_bucket_id),
            fps_id=int(fps),
        )

        # Normalize output (can be str, list, or gradio_client.file.File)
        out = result[0] if isinstance(result, (list, tuple)) else result
        if hasattr(out, "path"):      # gradio_client file object
            out_path_src = out.path
        else:
            out_path_src = str(out)

        # Copy into your static videos folder
        ts = int(time.time() * 1000)
        filename = f"i2v_remote_{ts}.mp4"
        out_path_dst = os.path.join(VID_DIR, filename)
        os.makedirs(os.path.dirname(out_path_dst), exist_ok=True)
        shutil.copyfile(out_path_src, out_path_dst)

        rel = f"/static/outputs/videos/{filename}"
        session['last_generated_video'] = rel
        return rel


def parse_target_res(s: str, default=(1080, 1920)):
    """Parse target resolution string like '1080x1920' into (width, height) tuple"""
    try:
        w, h = s.lower().split('x')
        return int(w), int(h)
    except Exception:
        return default


def letterbox_to_canvas(clip: VideoFileClip, canvas_w=1080, canvas_h=1920):
    """Resize video clip to fit canvas while maintaining aspect ratio, add black bars if needed"""
    # keep aspect, fit inside, add black bars if needed
    target_aspect = canvas_w / canvas_h
    clip_aspect = clip.w / clip.h
    if clip_aspect > target_aspect:
        # wider: fit width, pad top/bottom
        new_w = canvas_w
        new_h = int(canvas_w / clip_aspect)
    else:
        # taller: fit height, pad left/right
        new_h = canvas_h
        new_w = int(canvas_h * clip_aspect)
    resized = clip.resize(newsize=(new_w, new_h))
    # position centered on a black canvas
    return resized.on_color(size=(canvas_w, canvas_h), color=(0,0,0), pos=("center","center"))


def mix_background_music(video_clip: VideoFileClip, music_path: str, loudness=-16.0):
    """
    Simple ducking by setting music volume relative to baseline.
    We don't compute LUFS; we just attenuate to ~podcast-safe level.
    """
    if not os.path.exists(music_path):
        return video_clip
    try:
        music = AudioFileClip(music_path)
        # loop if too short
        if music.duration < video_clip.duration:
            loops = int(video_clip.duration // music.duration) + 1
            music = concatenate_videoclips([music] * loops).subclip(0, video_clip.duration)  # type: ignore
        # naive attenuation: -16 dB ~ 0.158 in linear scale
        vol = max(0.05, min(0.5, 10 ** (loudness / 20.0)))
        music = music.volumex(vol)
        if video_clip.audio:
            out_audio = CompositeAudioClip([music, video_clip.audio.volumex(1.0)])
        else:
            out_audio = CompositeAudioClip([music])
        return video_clip.set_audio(out_audio)
    except Exception as e:
        print(f"[Reel] music mix failed: {e}")
        return video_clip
