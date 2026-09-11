import asyncio
from dotenv import load_dotenv
load_dotenv()

async def test_all():
    from src.config import get_settings
    s = get_settings()

    # Neon
    try:
        import asyncpg
        url = s.database_url.replace("postgresql+asyncpg://", "postgresql://").split("?")[0]
        conn = await asyncpg.connect(url, ssl="require", timeout=15)
        n = await conn.fetchval("SELECT COUNT(*) FROM tickets")
        await conn.close()
        print(f"[OK] Neon ({n} tickets)")
    except Exception as e:
        print(f"[FAIL] Neon: {e}")

    # Qdrant
    try:
        from qdrant_client import QdrantClient
        q = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=15)
        info = q.get_collection(s.qdrant_collection)
        print(f"[OK] Qdrant ({info.points_count} points, {info.config.params.vectors.size} dims)")
    except Exception as e:
        print(f"[FAIL] Qdrant: {e}")

    # Redis
    try:
        import redis.asyncio as redis
        r = redis.from_url(s.redis_url, socket_connect_timeout=10)
        pong = await r.ping()
        await r.aclose()
        print(f"[OK] Redis (PING={pong})")
    except Exception as e:
        print(f"[FAIL] Redis: {e}")

    # Gemini
    try:
        import httpx
        r = httpx.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={s.gemini_api_key}",
            timeout=15,
        )
        if r.status_code == 200:
            models = r.json().get("models", [])
            valid = [m for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
            print(f"[OK] Gemini ({len(valid)} generative models available)")
            print(f"     Configured model '{s.gemini_model}' present: {any(s.gemini_model in m['name'] for m in valid)}")
        else:
            print(f"[FAIL] Gemini: HTTP {r.status_code} — {r.text[:200]}")
    except Exception as e:
        print(f"[FAIL] Gemini: {e}")

    # OpenRouter
    try:
        r = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {s.openrouter_api_key}"},
            timeout=15,
        )
        print(f"[OK] OpenRouter ({len(r.json().get('data', []))} models)")
    except Exception as e:
        print(f"[FAIL] OpenRouter: {e}")

asyncio.run(test_all())
