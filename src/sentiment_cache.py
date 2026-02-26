import json
from pathlib import Path

CACHE_PATH = Path("data/sentiment_cache.json")
CACHE_PATH.parent.mkdir(exist_ok=True)

def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

def get_cached(key):
    cache = load_cache()
    return cache.get(key)

def set_cached(key, value):
    cache = load_cache()
    cache[key] = value
    save_cache(cache)
