"""
LLM-based structured query parsing via langchain-groq's ChatGroq +
`.with_structured_output()` (Pydantic structured outputs).

Why this on top of the rule-based parser (query_parser_rules.py)?
The rule-based parser is fast, free, and works well for the target query
style, but it's a closed-vocabulary positional heuristic -- it breaks on
phrasing outside its assumptions:
  - negation: "a shirt with no tie"
  - implied attributes: "business casual" implying a blazer without the
    word "blazer" ever appearing
  - unusual word order or descriptive clauses: "the coat she wears over her
    blue dress"
  - colors/garments outside its fixed vocabulary lists

For those, this module makes a single Groq call constrained to return a
typed Pydantic object. `ChatGroq.with_structured_output(ParsedQuery)` binds
the Pydantic schema as a tool call under the hood and parses the model's
response straight back into a `ParsedQuery` instance -- no manual JSON
parsing, no prompt-format fragility.

Both this and query_parser_rules.py return the same dict shape (see
retriever/query_parser.py, the dispatcher that picks between them), so
retriever/search.py never needs to know which one ran.

Setup:
    pip install langchain-groq
    export GROQ_API_KEY=your_key_here     # https://console.groq.com/keys
"""
import os
import sys 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
from typing import List, Optional

from pydantic import BaseModel, Field

from config import GROQ_MODEL


class Garment(BaseModel):
    category: str = Field(
        description="Garment type, e.g. 'shirt', 'tie', 'raincoat', 'pants'. Singular, lowercase."
    )
    color: Optional[str] = Field(
        default=None, description="Color of THIS SPECIFIC garment if mentioned, else null."
    )


class ParsedQuery(BaseModel):
    garments: List[Garment] = Field(
        default_factory=list,
        description="Every distinct garment mentioned, each bound to its own color if specified.",
    )
    environment: Optional[str] = Field(
        default=None,
        description="One of: street style, stduio,editorial,runway. Null if no location is implied.",
    )
    style_hints: List[str] = Field(
        default_factory=list,
        description="Short style/vibe adjectives, e.g. casual, formal, professional, athletic.",
    )


_SYSTEM_PROMPT = """You extract structured fashion-search attributes from a user's natural language query.

Rules:
- Bind colors to the SPECIFIC garment they describe. "a red tie and a white shirt"
  means tie=red, shirt=white -- never swap or merge them.
- Only include a garment if it is actually mentioned or strongly implied
  (e.g. "business suit" implies a suit).
- environment must be exactly one of: office, urban street, park, home -- or
  null if the query doesn't imply a location.
- style_hints should be short adjectives (casual, formal, professional,
  athletic, streetwear, elegant), not full phrases.
- Do not invent attributes that aren't in or implied by the query."""


class LLMQueryParser:
    def __init__(self, api_key: str = None, model: str = None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "No Groq API key found. Set the GROQ_API_KEY environment variable "
                "or pass api_key=... explicitly."
            )
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=model or GROQ_MODEL, api_key=api_key, temperature=0)
        structured_llm = llm.with_structured_output(ParsedQuery)
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", "{query}")]
        )
        self.chain = prompt | structured_llm

    def parse(self, query: str) -> ParsedQuery:
        return self.chain.invoke({"query": query})

    def parse_as_dict(self, query: str) -> dict:
        """Same shape as query_parser_rules.parse_query() for drop-in compatibility."""
        parsed = self.parse(query)
        return {
            "garments": [{"color": g.color, "category": g.category} for g in parsed.garments],
            "environment": parsed.environment,
            "style_hints": parsed.style_hints,
            "raw_text": query,
        }


if __name__ == "__main__":
    parser = LLMQueryParser()
    tests = [
        "A person in a bright yellow raincoat.",
        "A shirt with no tie, business casual.",
        "The coat she wears over her blue dress on a rainy street.",
        "A red tie and a white shirt in a formal setting.",
    ]
    for t in tests:
        print(t)
        print(" ->", parser.parse_as_dict(t))
        print()
