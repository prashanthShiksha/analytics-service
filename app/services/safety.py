"""
Content safety checker — non-LLM flagging for abusive language and sensitive PII patterns.

Used as Step 3 in the thematic classification pipeline.
A statement is Flagged if EITHER the regex/keyword pass OR the pattern check trips.
"""
import re
import logging

logger = logging.getLogger("analytics_service.services.safety")

# --- Regex patterns for sensitive PII ---
PII_PATTERNS = [
    # Email addresses
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    # Phone numbers (Indian 10-digit, with optional +91/0 prefix)
    re.compile(r'(?:\+91[\-\s]?|0)?[6-9]\d{9}\b'),
    # Aadhaar numbers (12-digit, optionally space/dash separated in groups of 4)
    re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'),
    # PAN card (Indian format: 5 letters, 4 digits, 1 letter)
    re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),
]

# --- Abusive / profane keyword blocklist (lowercase, expandable) ---
# This is a starter set — extend as needed for the domain.
ABUSIVE_KEYWORDS = {
    "fuck", "shit", "damn", "bastard", "bitch", "asshole", "slut", "whore",
    "dick", "cunt", "piss", "crap", "nigger", "faggot", "retard",
    # Hindi profanity (transliterated)
    "madarchod", "bhenchod", "chutiya", "gaand", "lund", "randi", "saala",
    "harami", "kamina", "bhosdike",
}


def check_pii_patterns(text: str) -> bool:
    """Returns True if any PII regex pattern matches in the text."""
    for pattern in PII_PATTERNS:
        if pattern.search(text):
            logger.info("PII pattern detected in statement")
            return True
    return False


def check_abusive_language(text: str) -> bool:
    """Returns True if any abusive keyword is found in the text (case-insensitive, word-boundary)."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    matched = words & ABUSIVE_KEYWORDS
    if matched:
        logger.info(f"Abusive keyword(s) detected: {matched}")
        return True
    return False


def is_flagged(text: str) -> bool:
    """
    Main safety check entry point.
    Returns True if the statement should be marked content_quality = 'Flagged'.
    Short-circuits: runs the cheap keyword check first; only runs regex PII if keyword didn't flag.
    """
    if check_abusive_language(text):
        return True
    if check_pii_patterns(text):
        return True
    return False
