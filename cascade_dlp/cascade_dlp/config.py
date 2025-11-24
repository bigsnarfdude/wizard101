"""
Configuration for Cascade DLP v2.
"""

# GLiNER2 model
MODEL_NAME = "fastino/gliner2-base-v1"

# PII types for zero-shot detection
PII_TYPES = [
    # Personal identifiers
    "person name",
    "first name",
    "last name",
    "email address",
    "phone number",
    "social security number",
    "passport number",
    "driver license number",

    # Financial
    "credit card number",
    "bank account number",
    "iban",

    # Location
    "street address",
    "city",
    "zip code",
    "country",

    # Digital
    "ip address",
    "mac address",
    "username",
    "password",
    "api key",
    "access token",

    # Demographics
    "date of birth",
    "age",
    "gender",

    # Other
    "vehicle registration",
    "medical record number",
    "tax id",
]

# Regex patterns for deterministic detection (Stage 0)
SECRET_PATTERNS = {
    "AWS_ACCESS_KEY": {
        "pattern": r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}",
        "confidence": 0.95,
    },
    "GITHUB_TOKEN": {
        "pattern": r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}",
        "confidence": 0.95,
    },
    "JWT": {
        "pattern": r"eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*",
        "confidence": 0.90,
    },
    "RSA_PRIVATE_KEY": {
        "pattern": r"-----BEGIN RSA PRIVATE KEY-----[\s\S]*?-----END RSA PRIVATE KEY-----",
        "confidence": 1.0,
    },
    "OPENSSH_PRIVATE_KEY": {
        "pattern": r"-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]*?-----END OPENSSH PRIVATE KEY-----",
        "confidence": 1.0,
    },
}

# Policy actions
POLICY_ACTIONS = {
    "password": "BLOCK",
    "api key": "BLOCK",
    "access token": "BLOCK",
    "social security number": "BLOCK",
    "credit card number": "REDACT",
    "email address": "REDACT",
    "phone number": "REDACT",
    "person name": "REDACT",
    "street address": "REDACT",
}

# Default action for unlisted types
DEFAULT_ACTION = "REDACT"

# Redaction style: "type", "generic", "hash", "partial"
REDACTION_STYLE = "type"
