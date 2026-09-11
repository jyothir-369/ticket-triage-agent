import os, httpx
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("GEMINI_API_KEY")
print("Key prefix:", key[:10] + "..." if key else "MISSING")
print("Key length:", len(key) if key else 0)
print()

r = httpx.get(
    f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
    timeout=15,
)
print("Status:", r.status_code)

if r.status_code == 200:
    data = r.json()
    models = [m["name"] for m in data.get("models", [])]
    print(f"Total models available: {len(models)}")
    flash_models = [m for m in models if "flash" in m.lower()]
    print("Flash models (first 5):")
    for m in flash_models[:5]:
        print("  -", m)
elif r.status_code == 400:
    print("ERROR: API key is invalid or malformed")
    print("Response:", r.text[:300])
else:
    print("Response:", r.text[:300])
