"""
Unified query-parsing entry point -- this is the only module retriever/search.py
imports from.

Tries the Groq LLM-based structured parser first (handles negation, implied
attributes, and more natural phrasing than the rule-based parser can).
Falls back to the free, zero-dependency, zero-latency rule-based parser
(query_parser_rules.py) if:
  - no GROQ_API_KEY is set, or
  - the API call fails for any reason (network error, rate limit, invalid
    response, etc.)

Retrieval should never hard-fail just because an external API call failed --
degrading to the rule-based parser keeps the system usable, just with the
narrower rule-based coverage until the LLM path is available again.

The LLM client is initialized lazily and cached (not re-created per query),
and initialization is only attempted once per process -- if it fails once
(e.g. missing key), we don't retry Groq init on every subsequent query.
"""
import logging
import os

from retriever.query_parser_rules import parse_query as _parse_query_rules

logger = logging.getLogger(__name__)

_llm_parser = None
_llm_init_attempted = False


def _get_llm_parser():
    global _llm_parser, _llm_init_attempted
    if _llm_init_attempted:
        return _llm_parser
    _llm_init_attempted = True

    if not os.environ.get("GROQ_API_KEY"):
        logger.info("GROQ_API_KEY not set -- using rule-based query parser only.")
        return None

    try:
        from retriever.query_parser_llm import LLMQueryParser

        _llm_parser = LLMQueryParser()
        logger.info("LLM query parser (Groq) initialized.")
    except Exception as e:
        logger.warning(f"Could not initialize LLM query parser ({e}); using rule-based fallback.")
        _llm_parser = None

    return _llm_parser


def parse_query(query: str) -> dict:
    parser = _get_llm_parser()
    if parser is not None:
        try:
            return parser.parse_as_dict(query)
        except Exception as e:
            logger.warning(f"LLM parse failed ({e}); falling back to rule-based parser for this query.")
    return _parse_query_rules(query)


if __name__ == "__main__":
    tests = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting.",
    ]
    for t in tests:
        print(t)
        print(" ->", parse_query(t))
        print()
