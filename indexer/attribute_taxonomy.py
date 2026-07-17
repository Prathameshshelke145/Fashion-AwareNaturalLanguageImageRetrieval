"""
Controlled vocabularies used across indexing and retrieval.

Why a fixed taxonomy instead of letting a captioning model free-associate?
Free-text captions are exactly what causes CLIP's compositionality problem --
"red" and "shirt" become loosely associated tokens in one big embedding.
By extracting attributes into *structured slots* (garment_category, color,
environment) at index time, and parsing the query into the same slots at
retrieval time, we get exact/near-exact matching for the hard cases, and use
the dense embedding only for the fuzzy "vibe" part of the query.

The garment categories mirror Fashionpedia's apparel taxonomy (a subset of
the 46 main categories -- Fashionpedia also has ~19 part categories like
"collar" / "sleeve" which we deliberately exclude, they're too fine-grained
to be useful as *retrieval* units).

IMPORTANT: GARMENT_CATEGORIES is the index vocabulary -- every value must be
reachable from FASHIONPEDIA_CATEGORY_MAP, or that category can never actually
appear in the index and any query for it will silently return zero results.
Terms a user might type that AREN'T distinct categories in the underlying
data (e.g. "raincoat" -- Fashionpedia has no raincoat/coat distinction) go in
GARMENT_SYNONYMS instead, and query_parser.py should resolve a parsed
garment term through GARMENT_SYNONYMS before it's matched against the index.
"""

GARMENT_CATEGORIES = [
    "shirt", "blouse", "t-shirt", "sweater", "cardigan", "jacket", "coat",
    "blazer", "vest", "hoodie", "top", "dress", "jumpsuit",
    "pants", "jeans", "shorts", "skirt", "leggings", "suit", "tie",
    "bow tie", "scarf", "glasses", "hat", "cap", "gloves", "shoe", "boot",
    "sneaker", "sock", "bag", "belt", "watch",
]

# Fashionpedia's raw category names -> our simplified retrieval vocabulary.
# (Fashionpedia uses names like "shirt, blouse" or "cardigan" combined categories;
# this map normalizes those into single terms a user would actually type.)
FASHIONPEDIA_CATEGORY_MAP = {
    "shirt, blouse": "shirt",
    "top, t-shirt, sweatshirt": "t-shirt",
    "sweater": "sweater",
    "cardigan": "cardigan",
    "jacket": "jacket",
    "vest": "vest",
    "pants": "pants",
    "shorts": "shorts",
    "skirt": "skirt",
    "coat": "coat",
    "dress": "dress",
    "jumpsuit": "jumpsuit",
    "cape": "coat",
    "glasses": "glasses",
    "hat": "hat",
    "headband, head covering, hair accessory": "hat",
    "tie": "tie",
    "glove": "gloves",
    "watch": "watch",
    "belt": "belt",
    "leg warmer": "leggings",
    "tights, stockings": "leggings",
    "sock": "sock",
    "shoe": "shoe",
    "bag, wallet": "bag",
    "scarf": "scarf",
}

# Terms a user is likely to type that map onto an EXISTING index category
# rather than being their own category. Fashionpedia doesn't distinguish
# these at annotation granularity, so a query for "raincoat" against an
# index built with only "coat" categories would otherwise silently return
# nothing. query_parser.py should apply this lookup (case-insensitive) to
# every parsed garment term before it's matched against the index.
GARMENT_SYNONYMS = {
    "raincoat": "coat",
    "trench coat": "coat",
    "trenchcoat": "coat",
    "parka": "coat",
    "overcoat": "coat",
    "windbreaker": "jacket",
    "bomber jacket": "jacket",
    "denim jacket": "jacket",
    "jean jacket": "jacket",
    "gown": "dress",
    "sundress": "dress",
    "sunglasses": "glasses",
    "shades": "glasses",
    "trainers": "sneaker",
    "sneakers": "sneaker",
    "boots": "boot",
    "handbag": "bag",
    "purse": "bag",
    "necktie": "tie",
    "bowtie": "bow tie",
    "beanie": "hat",
    "trousers": "pants",
    "slacks": "pants",
}

# Curated color-name -> RGB reference table for nearest-neighbor color naming.
# Kept intentionally fashion-relevant (e.g. "navy" and "beige" instead of only
# primary colors) since eval query #1 depends on precise color naming.
COLOR_NAME_TO_RGB = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "red": (200, 30, 30),
    "maroon": (128, 0, 0),
    "pink": (240, 150, 180),
    "orange": (240, 130, 30),
    "yellow": (235, 210, 40),
    "bright yellow": (255, 225, 0),
    "gold": (200, 165, 60),
    "beige": (222, 202, 168),
    "brown": (110, 70, 40),
    "tan": (190, 160, 110),
    "green": (60, 130, 70),
    "olive": (100, 100, 40),
    "teal": (30, 130, 130),
    "blue": (40, 80, 190),
    "navy": (20, 30, 80),
    "sky blue": (120, 180, 230),
    "purple": (110, 60, 150),
    "lavender": (180, 160, 220),
    "cream": (245, 235, 210),
    "silver": (192, 192, 192),
}

# Environment vocabulary, matched zero-shot against the FashionCLIP text
# tower (see environment_classifier.py).
#
# Previously this was {office, urban street, park, home} -- a vocabulary
# that fits generic lifestyle photography, not Fashionpedia. Fashionpedia's
# val split is overwhelmingly studio product shots, runway/show photography,
# and street-style/editorial captures; there is very little genuine office,
# park-bench, or living-room imagery in it. Classifying every image into the
# old 4-way vocabulary forced each one into the *least-wrong* option rather
# than a genuinely matching one, which produces noisy, low-signal tags.
#
# This vocabulary instead reflects how fashion photography actually gets
# shot, and adds an explicit "unknown" escape hatch (see
# environment_classifier.py) for images that don't clearly match any label,
# rather than forcing a bad guess with a misleadingly confident score.
ENVIRONMENTS = {
    "studio": "a fashion product photo shot in a plain studio against a "
              "solid, seamless backdrop, no outdoor scenery",
    "runway": "a runway fashion show, a model walking a catwalk in front "
              "of an audience with stage lighting",
    "street style": "a candid street-style photo of a person walking "
                     "outdoors on a city sidewalk or urban street",
    "editorial": "a stylized editorial fashion magazine photo with an "
                 "artistic set, dramatic lighting, or elaborate backdrop",
}

# Style keywords are deliberately NOT hard-matched -- they're subjective /
# compositional in a different way (a "casual weekend outfit" isn't one
# garment+color, it's a gestalt). We let the fashion-tuned CLIP global
# embedding handle these via zero-shot similarity instead of a lookup table.
STYLE_HINTS = [
    "formal", "casual", "professional", "business", "weekend", "athletic",
    "streetwear", "elegant", "sporty", "relaxed",
]
