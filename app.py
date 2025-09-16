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

# ------------------------------
# Ensure ffmpeg/ffprobe available to Whisper (Windows)
# ------------------------------
FFMPEG_PATH = r"C:\Users\abc\Downloads\fftf\ffmpeg-2025-09-04-git-2611874a50-essentials_build\ffmpeg-2025-09-04-git-2611874a50-essentials_build\bin\ffmpeg.exe"
if not os.path.exists(FFMPEG_PATH):
    raise RuntimeError(f"ffmpeg not found at {FFMPEG_PATH}")

# Whisper requires FFmpeg to be discoverable before import
os.environ["FFMPEG_BINARY"] = FFMPEG_PATH
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

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
    # 1️⃣ Fetch candidate from Mongo register database
    candidate = candidates_collection.find_one({"email": req.email})

    if not candidate:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate with email {req.email} not found in MongoDB register database"
        )

    # Extract _id (ObjectId) from Mongo
    candidate_id = str(candidate["_id"])  # use string for Supabase

    # 2️⃣ Ensure candidate exists in Supabase
    existing = supabase.table("candidates").select("*").eq("candidate_id", candidate_id).execute()

    if not existing.data:
        supabase.table("candidates").insert({
            "candidate_id": candidate_id,
            "name": candidate.get("name"),
            "email": candidate.get("email")
        }).execute()
        print(f"✅ Candidate synced to Supabase: {candidate_id}")

    # 3️⃣ Create new session in Supabase
    supabase.table("sessions").insert({
        "candidate_id": candidate_id,
        "q_index": 0
    }).execute()

    # 4️⃣ Create interview record in Mongo interviews collection
    interviews_collection.insert_one({
        "candidate_id": ObjectId(candidate_id),   # stored as ObjectId in Mongo
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

    # Generate TTS if not exists
    if not filepath.exists():
        gTTS(text=question, lang="en").save(str(filepath))

    # ✅ Upload to Supabase bucket if not already uploaded
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
    # Get session
    session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
    if not session_res.data:
        raise HTTPException(404, "Session not found")
    session = session_res.data[0]

    # Save uploaded audio temporarily
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    tmp_input.write(await file.read())
    tmp_input.close()

    # Convert WebM → WAV
    tmp_wav_path = convert_to_wav(tmp_input.name)

    # Debug audio info
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

    # Transcribe with Whisper
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

        # Retry once if empty
        if not text_answer:
            print("⚠️ Empty transcript, retrying with longer padding...")
            audio = whisper.pad_or_trim(audio, length=30*16000)  # 30s
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

    # Upload original audio
    path_in_bucket = f"{candidate_id}/{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
    with open(tmp_input.name, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(path_in_bucket, f.read())
    audio_url = supabase.storage.from_(BUCKET_NAME).get_public_url(path_in_bucket)

    # Cleanup temp files
    os.remove(tmp_input.name)
    os.remove(tmp_wav_path)

    # ✅ Save user answers in Supabase
    supabase.table("interviews").insert({
        "candidate_id": candidate_id,
        "question": QUESTIONS[question_index],
        "answer_text": text_answer,
        "status": status,
        "answer_audio_url": audio_url
    }).execute()

    # ✅ Save user answers in Mongo (Q/A format inside single doc)
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

    # Move to next question only if transcription succeeded
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

# ------------------------------
# NEW ROUTE: Fetch all questions + answers as transcript
# ------------------------------
@app.get("/get_answers/{candidate_id}")
async def get_answers(candidate_id: str):
    clean_id = candidate_id.strip()

    # ✅ First check Mongo
    try:
        doc = interviews_collection.find_one({"candidate_id": ObjectId(clean_id)})
    except:
        doc = None

    if doc and "qa" in doc:
        return {
            "candidate_id": clean_id,
            "qa": doc["qa"]
        }

    # fallback → Supabase if nothing in Mongo
    res = supabase.table("interviews").select("question, answer_text").eq("candidate_id", clean_id).execute()
    if not res.data:
        return {"candidate_id": clean_id, "qa": []}

    transcript = []
    for row in res.data:
        transcript.append([f"Q: {row['question']}", f"A: {row['answer_text']}"])

    return {
        "candidate_id": clean_id,
        "qa": transcript
    }

# ------------------------------
# Static files
# ------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
