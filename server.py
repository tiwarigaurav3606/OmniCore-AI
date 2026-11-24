import base64
import io
import os
import random
import re
import urllib.parse
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

app = FastAPI()

# Render par Environment Variable se key lega, aur local laptop par aapki key chalegi
API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=API_KEY)


# Aapke Groq account par jo bhi model active hai use khud dhoondh kar set karega
FAST_MODEL = ""
VISION_MODEL = ""
try:
    models_data = client.models.list().data
    valid_ids = [
        m.id for m in models_data
        if "whisper" not in m.id.lower()
        and "guard" not in m.id.lower()
        and "tts" not in m.id.lower()
        and "safeguard" not in m.id.lower()
    ]
    for preferred in [
        "llama-4-maverick-17b-128e-instruct",
        "llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]:
        if preferred in valid_ids:
            FAST_MODEL = preferred
            break
    if not FAST_MODEL and valid_ids:
        FAST_MODEL = valid_ids.pop(0)

    for v_pref in [
        "llama-4-scout-17b-16e-instruct",
        "llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    ]:
        if v_pref in valid_ids:
            VISION_MODEL = v_pref
            break
    if not VISION_MODEL:
        VISION_MODEL = FAST_MODEL
except Exception:
    FAST_MODEL = "openai/gpt-oss-120b"
    VISION_MODEL = "openai/gpt-oss-120b"

def clean_text(txt):
    txt = re.sub(r'<think>.*?</think>', '', txt, flags=re.DOTALL)
    txt = re.sub(r'\*\*(.*?)\*\*', r'\1', txt)
    txt = re.sub(r'###+\s*', '', txt)
    txt = re.sub(r'---+', '', txt)
    txt = txt.replace('$$', '').replace('$', '')
    txt = txt.replace('\\(', '').replace('\\)', '').replace('\\[', '').replace('\\]', '')

    for _ in range(3):
        txt = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', txt)
        txt = re.sub(r'\\sqrt\{([^{}]+)\}', r'√(\1)', txt)

    replacements = {
        '\\times': '×',
        '\\div': '÷',
        '\\cdot': '·',
        '\\pm': '±',
        '\\pi': 'π',
        '\\theta': 'θ',
        '\\alpha': 'α',
        '\\beta': 'β',
        '\\leq': '≤',
        '\\geq': '≥',
        '\\neq': '≠',
        '\\approx': '≈',
        '\\infty': '∞',
        '^2': '²',
        '^3': '³',
        '^{2}': '²',
        '^{3}': '³',
        '* ': '• '
    }
    for k, v in replacements.items():
        txt = txt.replace(k, v)

    txt = re.sub(r'\\([a-zA-Z]+)', r'\1', txt)
    return txt.strip()

def should_auto_search(msg, low):
    search_triggers = [
        "school", "college", "hospital", "mandir", "station", "rampus", "gorakhpur",
        "news", "today", "aaj", "current", "latest", "price", "weather", "score",
        "who is", "where is", "kahan hai", "kaun hai", "address", "location", "2025", "2026"
    ]
    return any(t in low for t in search_triggers)

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content_bytes = await file.read()
        fname = file.filename or "file"
        low_name = fname.lower()

        if low_name.endswith((".png", ".jpg", ".jpeg", ".webp")):
            b64 = base64.b64encode(content_bytes).decode("utf-8")
            mime = "image/png" if low_name.endswith(".png") else "image/jpeg"
            v_res = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract every word, number, and math equation from this image in simple plain text (use normal fractions like a/b and powers like x²)."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    ]
                }],
                temperature=0.1
            )
            extracted = v_res.choices.pop(0).message.content.strip()
            return JSONResponse({"ok": True, "filename": fname, "text": extracted})

        extracted = ""
        if low_name.endswith(".pdf"):
            if PdfReader:
                reader = PdfReader(io.BytesIO(content_bytes))
                pages = [f"Page {i+1}:\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages[:20]) if (p.extract_text() or "").strip()]
                extracted = "\n".join(pages)
            else:
                return JSONResponse({"ok": False, "error": "pypdf install nahi hai."})
        else:
            extracted = content_bytes.decode("utf-8", errors="ignore")

        extracted = extracted.strip()[:16000]
        if not extracted or len(extracted) < 10:
            return JSONResponse({"ok": False, "error": "Ye PDF scanned hai! Iska screenshot (JPG/PNG) upload karein."})

        return JSONResponse({"ok": True, "filename": fname, "text": extracted})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})

@app.post("/chat")
async def chat(req: Request):
    try:
        b = await req.json()
        msg = b.get("message", "").strip()
        history = b.get("history", [])
        file_context = b.get("file_context", "").strip()
        use_search = b.get("search", True)
        use_memory = b.get("memory", True)
        low = msg.lower()

        when_words = ["kab banaya", "kab bana", "when were you", "when did", "which year", "which month", "release date", "launch date"]
        if any(w in low for w in when_words):
            return JSONResponse({
                "reply": "Nov 2026 me unhone (Gaurav Tiwari Sir) deep research karke mujhe banaya tha.",
                "sources": []
            })

        creator_words = [
            "who create", "who created", "who invent", "who invented",
            "who built", "who build", "who made", "kisne banaya",
            "banaya hai", "creator", "developer", "owner", "founder"
        ]
        if any(w in low for w in creator_words):
            return JSONResponse({
                "reply": "Sir Gaurav Tiwari created me.",
                "sources": []
            })

        img_triggers = [
            "generate image", "generate an image", "create image", "create an image",
            "draw ", "photo banao", "image banao", "tasveer banao", "picture of",
            "pic banao", "painting of", "sketch of", "make an image", "make image", "image of"
        ]
        if any(t in low for t in img_triggers) and not file_context:
            try:
                p_res = client.chat.completions.create(
                    messages=[{"role": "user", "content": "Convert this into a short 6-word English image description (no quotes): " + msg}],
                    model=FAST_MODEL,
                    temperature=0.3
                )
                short_prompt = p_res.choices.pop(0).message.content.strip().replace('"', '')
            except Exception:
                short_prompt = msg

            seed = random.randint(1, 99999)
            encoded = urllib.parse.quote(short_prompt)
            img_url = "https://" + "image.pollinations.ai/prompt/" + encoded + "?width=768&height=768&model=flux&seed=" + str(seed) + "&nologo=true"
            return JSONResponse({
                "reply": "IMG_URL:" + img_url,
                "sources": [{"title": "Download HD Image", "url": img_url}]
            })

        ctx, srcs = "", []
        if use_search and should_auto_search(msg, low) and not file_context:
            try:
                search_query = msg
                if any(k in low for k in ["school", "college", "hospital", "mandir", "station", "rampus"]):
                    search_query = msg + " Gorakhpur India"
                with DDGS(timeout=3) as ddgs:
                    res = list(ddgs.text(search_query, max_results=3))
                if res:
                    ctx = "\n\nVERIFIED LIVE WEB FACTS (Use these exact facts, do not invent):\n" + "\n".join([f"- {x.get('title')}: {x.get('body')}" for x in res])
                    srcs = [{"title": x.get("title", "Source"), "url": x.get("href", "#")} for x in res]
            except Exception:
                pass

        messages = [
            {
                "role": "system",
                "content": (
                    "You are OmniCore AI, a super-fast, intelligent assistant created by Sir Gaurav Tiwari in Nov 2026. "
                    "STRICT WRITING & FORMULA RULES:\n"
                    "1. Write like a normal human teacher in clean, simple plain text. NEVER use markdown symbols like **, ###, ---, or $.\n"
                    "2. NEVER write LaTeX code like \\frac, \\sqrt, \\times, or ^{2}. Always write normal human-readable math formulas like: (a + b)² = a² + 2ab + b², Area = πr², or (1/2) × base × height.\n"
                    "3. Keep answers clear, direct, and accurate. Only mention your creator or creation date if explicitly asked."
                )
            }
        ]

        if use_memory:
            for h in history[:-1]:
                if not h.get("isImg") and not str(h.get("text", "")).startswith("IMG_URL:"):
                    r_type = "user" if h.get("role") == "user" else "assistant"
                    messages.append({"role": r_type, "content": h.get("text", "")})

        final_user_msg = msg + ctx
        if file_context:
            final_user_msg = f"[Attached Document]\n{file_context}\n\n[Question]\n{msg}{ctx}"

        messages.append({"role": "user", "content": final_user_msg})

        # Fast Call (Reasoning effort low rakhne se 0.5 second me jawab aayega)
        kwargs = {
            "messages": messages,
            "model": FAST_MODEL,
            "temperature": 0.2
        }

        res = client.chat.completions.create(**kwargs)
        reply = clean_text(res.choices.pop(0).message.content.strip())
        return JSONResponse({"reply": reply, "sources": srcs})
    except Exception as e:
        return JSONResponse({"reply": "Maaf kijiye, abhi response generate nahi ho paya.", "sources": [], "error": str(e)})
