from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
import os
from hybrid_dna_qde_steganography import embed_hybrid_dna_qde, extract_hybrid_dna_qde

app = FastAPI()

# Allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/embed")
async def embed_secret(
    coverImage: UploadFile = File(...),
    secretFile: UploadFile = File(...),
    password: str = Form(...)
):
    with tempfile.TemporaryDirectory() as tmpdir:
        cover_path = os.path.join(tmpdir, "cover.png")
        secret_path = os.path.join(tmpdir, "secret")
        output_path = os.path.join(tmpdir, "stego.png")

        with open(cover_path, "wb") as f:
            shutil.copyfileobj(coverImage.file, f)
        with open(secret_path, "wb") as f:
            shutil.copyfileobj(secretFile.file, f)

        embed_hybrid_dna_qde(cover_path, secret_path, output_path, password)

        return StreamingResponse(open(output_path, "rb"), media_type="image/png")

@app.post("/api/extract")
async def extract_secret(
    coverImage: UploadFile = File(...),
    password: str = Form(...)
):
    with tempfile.TemporaryDirectory() as tmpdir:
        stego_path = os.path.join(tmpdir, "stego.png")
        with open(stego_path, "wb") as f:
            shutil.copyfileobj(coverImage.file, f)

        # NOTE: Bit length is fixed or must be extracted via index file if known
        bit_len = 4096 * 8  # Replace with actual logic if needed
        extract_hybrid_dna_qde(stego_path, bit_len, password)

        recovered_file = os.path.join(os.getcwd(), "recovered_secret")
        return StreamingResponse(open(recovered_file, "rb"), media_type="application/octet-stream")
