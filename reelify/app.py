import os
import time
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    send_from_directory,
    url_for,
    redirect,
)

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)

from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------------------------------------------------------------
# Paths / config (Render-safe)
# -----------------------------------------------------------------------------

# BASE_DIR should be repo root (one level above reelify/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Use env vars on Render (fallback to dev)
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Feature flags: keep demo light on Render
# Set ENABLE_MEDIA=1 locally if you want SDXL/SVD endpoints active.
ENABLE_MEDIA = os.environ.get("ENABLE_MEDIA", "0") == "1"
# Optionally allow LLAMA locally; on Render keep it OFF
USE_LLAMA = os.environ.get("USE_LLAMA", "0") == "1"

# Output folders (only used if media enabled)
VID_DIR = os.path.join(STATIC_DIR, "outputs", "videos")
IMG_DIR = os.path.join(STATIC_DIR, "outputs", "images")
os.makedirs(VID_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Flask app setup
# -----------------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=TEMPLATES_DIR,
)
app.secret_key = SECRET_KEY

# -----------------------------------------------------------------------------
# Login manager setup
# -----------------------------------------------------------------------------

login_manager = LoginManager(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    def get_id(self):
        return str(self.id)


# Simple in-memory user store
_users_by_username = {
    "admin": User(
        id=1,
        username="admin",
        password_hash=generate_password_hash(os.environ.get("ADMIN_PASSWORD", "changeme123")),
    )
}
_users_by_id = {u.id: u for u in _users_by_username.values()}


@login_manager.user_loader
def load_user(user_id: str):
    try:
        uid = int(user_id)
    except ValueError:
        return None
    return _users_by_id.get(uid)


# -----------------------------------------------------------------------------
# Core services (trend + story only by default)
# -----------------------------------------------------------------------------

from .trends import TrendScraper
trend_scraper = TrendScraper()

# Import your story helpers.
# IMPORTANT: make sure your story.py uses GEMINI_API_KEY from env or accepts it.
from .story import (
    GeminiStoryGenerator,
    LlamaStoryGenerator,
    FacebookUrduTranslator,
    get_one_liner_for_trend,
)

from .evidence import fetch_evidence_for_trend
from .evaluation import evaluate_story_quality

# Story generator selection:
# - Render: Gemini
# - Local: optional Llama if you set USE_LLAMA=1 and you have Ollama running
if USE_LLAMA:
    story_generator = LlamaStoryGenerator(model_name=os.environ.get("LLAMA_MODEL", "llama3.1:8b"))
else:
    # Prefer Gemini on Render
    story_generator = GeminiStoryGenerator(model_name=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))

translator = FacebookUrduTranslator()

# Media services (optional)
t2i_generator = None
i2v_generator = None
parse_target_res = None
letterbox_to_canvas = None
mix_background_music = None
free_cuda = None

if ENABLE_MEDIA:
    # Heavy imports only when enabled
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    from .media import (
        SDXLImageGenerator,
        SVDImageToVideo,
        parse_target_res,
        letterbox_to_canvas,
        mix_background_music,
        free_cuda,
    )
    t2i_generator = SDXLImageGenerator(prefer_turbo=False)
    i2v_generator = SVDImageToVideo()


# -----------------------------------------------------------------------------
# Auth routes
# -----------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = _users_by_username.get(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("index"))

        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# -----------------------------------------------------------------------------
# UI routes
# -----------------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    return render_template("test.html")


# -----------------------------------------------------------------------------
# API routes (demo-ready)
# -----------------------------------------------------------------------------

@app.route("/api/scrape-trends", methods=["POST"])
@login_required
def scrape_trends():
    try:
        data = request.get_json() or {}
        location = data.get("location", "pakistan")

        trends = trend_scraper.scrape_trends(location)

        # Add one-liners (Gemini) — if this fails, we degrade gracefully
        for trend in trends:
            try:
                trend["description"] = get_one_liner_for_trend(trend["name"])
            except Exception:
                trend["description"] = f"Trending topic: {trend['name']}"

        session["current_trends"] = trends
        session["location"] = location

        return jsonify(
            {
                "success": True,
                "trends": trends,
                "location": location,
                "count": len(trends),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/generate-story", methods=["POST"])
@login_required
def generate_story():
    try:
        data = request.get_json() or {}
        tag = data.get("tag")
        description = data.get("description")

        if not tag or not description:
            return jsonify({"success": False, "error": "Missing tag or description"}), 400

        location = data.get("location") or session.get("location", "pakistan")

        evidence_articles = fetch_evidence_for_trend(
            trend_name=tag,
            location=location,
            max_articles=4,
            trend_description=description,
        )

        story = story_generator.generate_story(
            tag=tag,
            description=description,
            story_length="0.5-1 minute",
            evidence_articles=evidence_articles,
        )

        if not story:
            return jsonify({"success": False, "error": "Story generation failed"}), 500

        evaluation = evaluate_story_quality(
            story_text=story,
            evidence_articles=evidence_articles,
            tag=tag,
            description=description,
        )

        session["last_generated_story"] = story
        session["last_story_sources"] = evidence_articles

        return jsonify(
            {
                "success": True,
                "story": story,
                "tag": tag,
                "model_used": getattr(story_generator, "model_label", "unknown"),
                "location_used": location,
                "evidence_count": len(evidence_articles),
                "evidence_articles": evidence_articles,
                "evaluation": evaluation,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/translate-story", methods=["POST"])
@login_required
def translate_story():
    try:
        data = request.get_json() or {}
        text_to_translate = data.get("text") or session.get("last_generated_story")

        if not text_to_translate:
            return jsonify({"success": False, "error": "No text provided and no story found in session"}), 400

        translated_text = translator.translate(text_to_translate)

        return jsonify(
            {
                "success": True,
                "original_text": text_to_translate,
                "translated_text": translated_text,
                "source_language": "English",
                "target_language": "Urdu",
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"Translation failed: {str(e)}"}), 500


# -----------------------------------------------------------------------------
# Media endpoints (optional; disabled on Render by default)
# -----------------------------------------------------------------------------

def _media_disabled():
    return jsonify(
        {
            "success": False,
            "error": "Media endpoints are disabled on this deployment (ENABLE_MEDIA=0).",
        }
    ), 503


@app.route("/api/generate-image", methods=["POST"])
@login_required
def generate_image():
    if not ENABLE_MEDIA or t2i_generator is None:
        return _media_disabled()

    try:
        data = request.get_json() or {}
        prompt = data.get("prompt") or "cinematic b-roll, realistic scene, no text"

        rel_path = t2i_generator.generate(
            prompt=prompt,
            negative_prompt=data.get("negative_prompt"),
            width=int(data.get("width", 1024)),
            height=int(data.get("height", 1024)),
            steps=int(data.get("steps", 30)),
            guidance_scale=float(data.get("guidance_scale", 5.5)),
            seed=data.get("seed"),
        )

        t2i_generator.unload()
        return jsonify({"success": True, "image_url": rel_path, "prompt": prompt})
    except Exception as e:
        return jsonify({"success": False, "error": f"Image generation failed: {str(e)}"}), 500


@app.route("/api/image-to-video", methods=["POST"])
@login_required
def image_to_video():
    if not ENABLE_MEDIA or i2v_generator is None:
        return _media_disabled()

    try:
        data = request.get_json() or {}
        image_src = data.get("image") or session.get("last_generated_image")

        if not image_src:
            return jsonify({"success": False, "error": "No image provided and no previous image found."}), 400

        rel_path = i2v_generator.generate(
            image_source=image_src,
            fps=int(data.get("fps", 7)),
            motion_bucket_id=int(data.get("motion_bucket_id", 127)),
            noise_aug_strength=float(data.get("noise_aug_strength", 0.02)),
            num_frames=int(data.get("num_frames", 14)),
        )

        i2v_generator.unload()
        return jsonify({"success": True, "video_url": rel_path})
    except Exception as e:
        return jsonify({"success": False, "error": f"Image-to-video failed: {str(e)}"}), 500


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "ok",
            "enable_media": ENABLE_MEDIA,
            "using_llama": USE_LLAMA,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/static/outputs/<path:subpath>")
def serve_outputs(subpath):
    directory, filename = os.path.split(subpath)
    return send_from_directory(os.path.join(STATIC_DIR, "outputs", directory), filename)


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=True, host="0.0.0.0", port=port)
