import os
import uuid
import tempfile
import subprocess
from pathlib import Path
import wave
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gtts import gTTS
from supabase import create_client
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import whisper
import shutil

# ------------------------------
# FFmpeg path setup (portable)
# ------------------------------
def get_ffmpeg_path():
    try:
        import imageio_ffmpeg as iioff
        ffmpeg_path = iioff.get_ffmpeg_exe()
        print(f"✅ Using imageio-ffmpeg binary: {ffmpeg_path}")
        return ffmpeg_path
    except Exception as e:
        print("⚠️ imageio-ffmpeg not available:", e)

    # check ENV
    ffmpeg_path = os.getenv("FFMPEG_PATH")
    if ffmpeg_path and Path(ffmpeg_path).exists():
        print(f"✅ Using FFMPEG_PATH from env: {ffmpeg_path}")
        return ffmpeg_path

    # last fallback system ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"✅ Found system ffmpeg: {ffmpeg_path}")
        return ffmpeg_path

    raise RuntimeError("❌ FFmpeg not found. Please install or set FFMPEG_PATH.")

FFMPEG_PATH = get_ffmpeg_path()
os.environ["FFMPEG_BINARY"] = FFMPEG_PATH
print(f"🎬 Final FFmpeg path in use: {FFMPEG_PATH}")

# ------------------------------
# Setup Supabase
# ------------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "interview-audios")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------
# Setup MongoDB
# ------------------------------
MONGO_URI = os.getenv("MONGO_URL")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["recruiter-platform"]
candidates_collection = mongo_db["candidates"]
interviews_collection = mongo_db["interviews"]

# ------------------------------
# App Config
# ------------------------------
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app = FastAPI(title="Interview Voice Bot Backend")

# ------------------------------
# CORS
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Whisper (lazy load)
# ------------------------------
whisper_model = None
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("⏳ Loading Whisper model (base)...")
        whisper_model = whisper.load_model("base", device="cpu")  # force CPU
        print("✅ Whisper model ready")
    return whisper_model

# ------------------------------
# Questions
# ------------------------------
QUESTIONS = [
    "How many years of experience do you have?",
    "What is your current CTC?",
    "What is your expected CTC?",
    "Which is your current location?",
    "Are you open to relocation?",
    "What is your notice period?",
]

class StartRequest(BaseModel):
    name: str | None = None
    email: str | None = None

# ------------------------------
# Helper: Convert WebM → WAV
# ------------------------------
def convert_to_wav(input_path: str) -> str:
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    try:
        subprocess.run(
            [FFMPEG_PATH, "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("✅ FFmpeg conversion OK:", output_path)
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg failed:", e.stderr.decode())
        raise
    return output_path

# ------------------------------
# Routes
# ------------------------------

@app.post("/start_interview")
async def start_interview(req: StartRequest):
    candidate = candidates_collection.find_one({"email": req.email})
    if not candidate:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate with email {req.email} not found in MongoDB register database"
        )

    candidate_id = str(candidate["_id"])  # use string for Supabase

    existing = supabase.table("candidates").select("*").eq("candidate_id", candidate_id).execute()
    if not existing.data:
        supabase.table("candidates").insert({
            "candidate_id": candidate_id,
            "name": candidate.get("name"),
            "email": candidate.get("email")
        }).execute()
        print(f"✅ Candidate synced to Supabase: {candidate_id}")

    supabase.table("sessions").insert({
        "candidate_id": candidate_id,
        "q_index": 0
    }).execute()

    interviews_collection.insert_one({
        "candidate_id": ObjectId(candidate_id),
        "question_index": 0,
        "question": "Welcome! Let's start your interview.",
        "answer_text": None,
        "status": "pending",
        "answer_audio_url": None
    })

    return {
        "message": "Interview started",
        "candidate_id": candidate_id,
        "next_question_url": f"/question/{candidate_id}"
    }

@app.get("/question/{candidate_id}")
async def get_question(candidate_id: str):
    session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
    if not session_res.data:
        raise HTTPException(404, "Session not found")

    session = session_res.data[0]
    idx = session["q_index"]

    if idx >= len(QUESTIONS):
        return {"done": True, "message": "Interview finished"}

    question = QUESTIONS[idx]
    filename = f"q_{candidate_id}_{idx}.mp3"
    filepath = STATIC_DIR / filename

    if not filepath.exists():
        gTTS(text=question, lang="en").save(str(filepath))

    path_in_bucket = f"{candidate_id}/bot_q_{idx}.mp3"
    with open(filepath, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(path_in_bucket, f.read(), {"upsert": "true"})
    audio_url = supabase.storage.from_(BUCKET_NAME).get_public_url(path_in_bucket)

    return {
        "done": False,
        "question_index": idx,
        "question": question,
        "audio_url": audio_url
    }

@app.post("/submit_answer/{candidate_id}/{question_index}")
async def submit_answer(candidate_id: str, question_index: int, file: UploadFile = File(...)):
    session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
    if not session_res.data:
        raise HTTPException(404, "Session not found")
    session = session_res.data[0]

    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    tmp_input.write(await file.read())
    tmp_input.close()

    tmp_wav_path = convert_to_wav(tmp_input.name)

    try:
        with wave.open(tmp_wav_path, "rb") as wf:
            print("🎧 WAV info:", {
                "channels": wf.getnchannels(),
                "sample_width": wf.getsampwidth(),
                "framerate": wf.getframerate(),
                "nframes": wf.getnframes(),
                "duration": wf.getnframes() / float(wf.getframerate())
            })
    except Exception as e:
        print("⚠️ Could not read WAV:", e)

    text_answer = ""
    status = "error"
    try:
        model = get_whisper_model()
        audio = whisper.load_audio(tmp_wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        text_answer = result.text.strip()

        if not text_answer:
            print("⚠️ Empty transcript, retrying with longer padding...")
            audio = whisper.pad_or_trim(audio, length=30*16000)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            result = whisper.decode(model, mel, options)
            text_answer = result.text.strip()

        if text_answer:
            status = "ok"
        else:
            text_answer = "(Could not detect speech)"
            status = "error"

        print("📝 Whisper transcript:", text_answer)

    except Exception as e:
        print("❌ Whisper error:", e)
        text_answer = "(Transcription failed)"
        status = "error"

    path_in_bucket = f"{candidate_id}/{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
    with open(tmp_input.name, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(path_in_bucket, f.read())
    audio_url = supabase.storage.from_(BUCKET_NAME).get_public_url(path_in_bucket)

    os.remove(tmp_input.name)
    os.remove(tmp_wav_path)

    supabase.table("interviews").insert({
        "candidate_id": candidate_id,
        "question": QUESTIONS[question_index],
        "answer_text": text_answer,
        "status": status,
        "answer_audio_url": audio_url
    }).execute()

    qa_entry = [
        f"Q: {QUESTIONS[question_index]}",
        f"A: {text_answer}"
    ]

    interviews_collection.update_one(
        {"candidate_id": ObjectId(candidate_id)},
        {"$push": {"qa": qa_entry}},
        upsert=True
    )
    print(f"✅ Q/A saved in Mongo for candidate {candidate_id}, Q{question_index}")

    if status == "ok":
        supabase.table("sessions").update(
            {"q_index": session["q_index"] + 1}
        ).eq("candidate_id", candidate_id).execute()

    return {
        "answer_text": text_answer,
        "status": status,
        "next_question_url": f"/question/{candidate_id}"
    }

@app.get("/finish_interview/{candidate_id}")
async def finish_interview(candidate_id: str):
    res = supabase.table("interviews").select("*").eq("candidate_id", candidate_id.strip()).execute()
    supabase.table("sessions").delete().eq("candidate_id", candidate_id.strip()).execute()
    return {"candidate_id": candidate_id.strip(), "answers": res.data}

@app.get("/get_answers/{candidate_id}")
async def get_answers(candidate_id: str):
    clean_id = candidate_id.strip()
    try:
        doc = interviews_collection.find_one({"candidate_id": ObjectId(clean_id)})
    except:
        doc = None

    if doc and "qa" in doc:
        return {"candidate_id": clean_id, "qa": doc["qa"]}

    res = supabase.table("interviews").select("question, answer_text").eq("candidate_id", clean_id).execute()
    if not res.data:
        return {"candidate_id": clean_id, "qa": []}

    transcript = []
    for row in res.data:
        transcript.append([f"Q: {row['question']}", f"A: {row['answer_text']}"])

    return {"candidate_id": clean_id, "qa": transcript}

# ------------------------------
# Static files
# ------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
