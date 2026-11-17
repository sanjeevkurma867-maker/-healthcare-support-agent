"""Settings. Thresholds were chosen on the development split of the evaluation set (see eval/)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
DB_PATH = Path(os.environ.get("HSA_DB", ROOT / "hsa.db"))
SECRET = os.environ.get("HSA_SECRET", "dev-only-secret-change-me")

TOKEN_TTL_SECONDS = 8 * 3600
MAX_LOGIN_FAILURES = 5
LOGIN_WINDOW_SECONDS = 15 * 60
MAX_MESSAGE_CHARS = 1000
TOOL_CALL_BUDGET = 3          # tool calls allowed per message

RRF_K = 60                    # Reciprocal Rank Fusion constant
TOP_K = 5                     # chunks kept after fusion
MAX_SENTENCES = 3             # sentences quoted in an answer
SPECIFIC_COVERAGE = 0.90      # path A: share of question terms found in the top chunk ...
SPECIFIC_COSINE = 0.20        # ... plus this best-sentence cosine
MIN_COSINE = 0.15             # best-sentence cosine needed to answer
MIN_MASS = 1.5                # summed IDF weight of question terms found in the top chunk
MIN_MATCHED = 2               # number of distinct question terms found in the top chunk
