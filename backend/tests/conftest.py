import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["CACHE_TYPE"] = "local"
