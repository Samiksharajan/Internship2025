import warnings
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cosine

# Configuration
SIMILARITY_THRESHOLD = 0.75  # 75%

# Get embeddings from audio
def get_embedding(audio_path, encoder):
    print(f"\n🔊 Processing audio: {audio_path}")
    wav = preprocess_wav(Path(audio_path))
    embed = encoder.embed_utterance(wav)
    return embed

# Cosine similarity to determine speaker similarity
def compute_similarity(embedding1, embedding2):
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def main():
    audio1 = "Recording.m4a"  # Replace with your actual file
    audio2 = "Recording (2).m4a"  # Replace with your actual file

    encoder = VoiceEncoder()

    emb1 = get_embedding(audio1, encoder)
    emb2 = get_embedding(audio2, encoder)

    similarity_score = compute_similarity(emb1, emb2)
    percentage = round(similarity_score * 100, 2)

    print(f"\n🎧 Similarity score: {percentage:.2f}%")

    if similarity_score >= SIMILARITY_THRESHOLD:
        print("✅ Match: Likely same speaker.")
    else:
        print("❌ Not a match: Likely different speakers.")

if __name__ == "__main__":
    main()