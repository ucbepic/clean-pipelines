"""NPI entity-resolution pipeline (modular library + CLI).

Resolve officer mentions to POST employment records, fetching candidates from the NPI
API. Works from a mentions CSV (batch) or directly from a name (single), via the same
PostMatcher. County / agency_type are optional — used when the data has them.

    from prap_post_entity_resolution.resolve import PostMatcher, build_mention
    matcher = PostMatcher(api_url="http://localhost:8001")
    verdict = matcher.resolve_one(build_mention({...}))
    results = matcher.resolve_batch(mentions)
"""

from .io import build_mention, read_mentions, write_outputs
from .pipeline import MentionResult, PostMatcher

__all__ = [
    "PostMatcher",
    "MentionResult",
    "build_mention",
    "read_mentions",
    "write_outputs",
]
