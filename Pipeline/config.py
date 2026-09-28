"""Central configuration for the agentic XAI pipeline. Change values here, not inside the modules."""
import os

# Run everything from the repository root (the folder that contains MedPix-2-0/ and pipeline/)
DATA_ROOT = os.environ.get("MEDPIX_ROOT", "MedPix-2-0")
SPLITS_DIR = os.path.join(DATA_ROOT, "splitted_dataset")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
KG_GRAPHS_DIR = os.path.join(DATA_ROOT, "KG", "graphs")
FAISS_DIR = os.path.join(DATA_ROOT, "MedPix", "vectorstore-history-linq-embed-mistral")

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_WEIGHTS = os.path.join(PIPELINE_DIR, "weights", "first_pass_cnn.pt")
OUTPUT_DIR = "outputs/pipeline"

# Label spaces (same targets as DR-Minerva: modality + location category)
MODALITIES = ["CT", "MR"]
LOCATIONS = ["Head", "Thorax", "Abdomen", "Spine and Muscles", "Reproductive and Urinary System"]
IMG_SIZE = 128

# ---- Route decision -------------------------------------------------------
# short path if the first-pass model is confident AND image/text models agree on location
ROUTE_CONF_THRESHOLD = 0.70
# if calibrate_router.py has been run, use the threshold it picked on the dev split
_router_json = os.path.join(PIPELINE_DIR, "weights", "router.json")
if os.path.exists(_router_json):
    import json as _json
    ROUTE_CONF_THRESHOLD = _json.load(open(_router_json))["threshold"]

# ---- Evidence gathering ---------------------------------------------------
RETRIEVAL_CORPUS_SPLIT = "train_1"   # same corpus as RAG_costruction_history.py
RETRIEVAL_BACKEND = "bm25"           # "bm25" (CPU, no download) or "faiss" (Week-3 Linq-Embed-Mistral index)
K_SHORT = 4                          # neighbours on the short path
K_LONG = 8                           # neighbours on the long path
KG_GRAPH_NAME = "hybrid-train-10tripsphi3mini"   # folder under MedPix-2-0/KG/graphs/
KG_MAX_TRIPLETS = 12

# ---- Synthesizer ----------------------------------------------------------
SYNTH_BACKEND = "template"           # "template" (no LLM) or "phi3" (GPU)
LLM_ID = "microsoft/Phi-3-mini-4k-instruct"

# ---- Verifier -------------------------------------------------------------
MAX_RETRIES = 1                      # verifier can send a case back once (short -> long path)
DELETION_FRACTION = 0.20             # Grad-CAM faithfulness: occlude top-20% pixels
