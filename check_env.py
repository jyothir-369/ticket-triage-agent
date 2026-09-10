from src.config import get_settings
s = get_settings()

def mask(v):
    if not v: return "MISSING"
    if len(v) < 20: return v
    return v[:20] + "..." + v[-4:]

print("--- INFRASTRUCTURE ---")
print(f"ENVIRONMENT               {s.environment}")
print(f"DATABASE_URL              {s.database_url[:60]}...")
print(f"DATABASE_USE_NULLPOOL     {s.database_use_nullpool}")
print(f"DATABASE_SSL_MODE         {s.database_ssl_mode}")
print(f"REDIS_URL                 {s.redis_url[:50]}...")
print(f"QDRANT_URL                {s.qdrant_url[:60] if s.qdrant_url else 'MISSING'}")
print(f"QDRANT_API_KEY            {mask(s.qdrant_api_key)}")
print()
print("--- LLM ---")
print(f"LLM_PROVIDER              {s.llm_provider}")
print(f"LLM_FALLBACK_PROVIDER     {s.llm_fallback_provider}")
print(f"GEMINI_API_KEY            {mask(s.gemini_api_key)}")
print(f"GEMINI_MODEL              {s.gemini_model}")
print(f"OPENROUTER_API_KEY        {mask(s.openrouter_api_key)}")
print(f"OPENROUTER_MODEL          {s.openrouter_model}")
print(f"OPENROUTER_BASE_URL       {s.openrouter_base_url}")
