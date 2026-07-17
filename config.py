import os 

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FASHIONPEDIA_IMAGE_DIR=os.path.join(PROJECT_ROOT,"data","images")
FASHIONPEDIA_ANNOTATION=os.path.join(PROJECT_ROOT,"data","instances_attributes_train2020.json")
CHROMA_PERSIST_DIR=os.path.join(PROJECT_ROOT,"data","chroma_store")

#Chroma collecection name
GLOBAL_COLLECTION="image_global"  # one vector per image (whole-scene embedding)
REGION_COLLECTION="germent_regions" ## one vector per detected garment instance

#MODELS
FASHION_CLIP_MODEL = "patrickjohncyh/fashion-clip"
FALLBACK_CLIP_MODEL = "openai/clip-vit-base-patch32"


GROQ_MODEL = "llama-3.3-70b-versatile"


WEIGHT_GLOBAL_SIM = 0.45       # whole-image <-> full query text similarity (style/vibe)
WEIGHT_COMPOSITIONAL = 0.40    # per-garment (color, category) region matching
WEIGHT_ENVIRONMENT = 0.15      # environment tag match bonus

TOP_K_DEFAULT = 10
GLOBAL_CANDIDATE_POOL = 100 