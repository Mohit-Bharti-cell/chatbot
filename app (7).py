import os
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
import whisper
import time

# ------------------------------
# FFMPEG Path (Windows)
# ------------------------------
FFMPEG_PATH = r"C:\Users\HP\OneDrive\Desktop\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
if not os.path.exists(FFMPEG_PATH):
    raise RuntimeError(f"ffmpeg not found at {FFMPEG_PATH}")

os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

# ------------------------------
# Supabase Config
# ------------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "interview-audios")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------
# FastAPI Setup
# ------------------------------
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app = FastAPI(title="Interview Voice Bot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Whisper
# ------------------------------
whisper_model = None
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("⏳ Loading Whisper model...")
        whisper_model = whisper.load_model("base", device="cpu")
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
    name: str
    email: str
    phone_number: str | None = None

# ------------------------------
# Helpers
# ------------------------------
def convert_to_wav(input_path: str) -> str:
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    subprocess.run(
        [FFMPEG_PATH, "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return output_path

# ------------------------------
# Routes
# ------------------------------
@app.post("/start_interview")
async def start_interview(req: StartRequest):
    # Check if candidate already has a candidate_id
    existing = supabase.table("responses").select("candidate_id").eq("email", req.email).execute()

    if existing.data:
        candidate_id = existing.data[0]["candidate_id"]
        print(f"ℹ️ Candidate already exists: {candidate_id}")
    else:
        # Create a new candidate_id (timestamp-based)
        candidate_id = int(time.time())
        print(f"✅ New candidate ID generated: {candidate_id}")

        # Insert a placeholder row to store candidate info (no question/answer yet)
        supabase.table("responses").insert({
            "candidate_id": candidate_id,
            "name": req.name,
            "email": req.email,
            "phone_number": req.phone_number,
            "q_index": -1  # means not yet started
        }).execute()

    return {
        "message": "Interview started",
        "candidate_id": candidate_id,
        "next_question_url": f"/question/{candidate_id}"
    }

@app.get("/question/{candidate_id}")
async def get_question(candidate_id: int):
    # Get the last answered index
    last = supabase.table("responses").select("q_index").eq("candidate_id", candidate_id).order("q_index", desc=True).limit(1).execute()
    idx = (last.data[0]["q_index"] if last.data else -1) + 1

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
async def submit_answer(candidate_id: int, question_index: int, file: UploadFile = File(...)):
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    tmp_input.write(await file.read())
    tmp_input.close()

    tmp_wav_path = convert_to_wav(tmp_input.name)

    text_answer = "(error)"
    try:
        model = get_whisper_model()
        audio = whisper.load_audio(tmp_wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        text_answer = result.text.strip() or "(Could not detect speech)"
    except Exception as e:
        print("❌ Whisper error:", e)
        text_answer = "(transcription failed)"

    path_in_bucket = f"{candidate_id}/answer_q{question_index}{os.path.splitext(file.filename)[1]}"
    with open(tmp_input.name, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(path_in_bucket, f.read(), {"upsert": "true"})
    audio_url = supabase.storage.from_(BUCKET_NAME).get_public_url(path_in_bucket)

    os.remove(tmp_input.name)
    os.remove(tmp_wav_path)

    # Insert into responses (single table)
    insert_res = supabase.table("responses").insert({
        "candidate_id": candidate_id,
        "question": QUESTIONS[question_index],
        "answer_text": text_answer,
        "answer_audio_url": audio_url,
        "q_index": question_index
    }).execute()

    if not insert_res.data:
        raise HTTPException(500, "❌ Failed to save interview answer")

    return {
        "answer_text": text_answer,
        "status": "ok",
        "answer_audio_url": audio_url,
        "next_question_url": f"/question/{candidate_id}"
    }

@app.get("/finish_interview/{candidate_id}")
async def finish_interview(candidate_id: int):
    res = supabase.table("responses").select("*").eq("candidate_id", candidate_id).order("q_index").execute()
    return {"candidate_id": candidate_id, "answers": res.data}

@app.get("/get_answers/{candidate_id}")
async def get_answers(candidate_id: int):
    res = supabase.table("responses").select("question, answer_text").eq("candidate_id", candidate_id).order("q_index").execute()

    if not res.data:
        return {"candidate_id": candidate_id, "answers": []}

    transcript = []
    for row in res.data:
        if row["question"]:
            transcript.append(f"Q: {row['question']}")
        if row["answer_text"]:
            transcript.append(f"A: {row['answer_text']}")

    return {
        "candidate_id": candidate_id,
        "answers": res.data,
        "transcript": transcript
    }

# ------------------------------
# Static files
# ------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
