"""
Secret/Credential Detector for cascade_dlp.

Patterns based on SecretBench (15,084 verified secrets across 8 categories).
Reference: github.com/setu1421/SecretBench
"""

import re
from typing import List
from dataclasses import dataclass


@dataclass
class Detection:
    """Detection result."""
    detector_name: str
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    metadata: dict


# SecretBench-derived patterns
SECRET_PATTERNS = {
    # API Keys
    "AWS_ACCESS_KEY": {
        "pattern": r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}",
        "confidence": 0.95,
        "description": "AWS Access Key ID",
    },
    "AWS_SECRET_KEY": {
        "pattern": r"(?i)aws[_\-\.]?secret[_\-\.]?(?:access)?[_\-\.]?key['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?",
        "confidence": 0.90,
        "description": "AWS Secret Access Key",
    },
    "GITHUB_TOKEN": {
        "pattern": r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}",
        "confidence": 0.95,
        "description": "GitHub Personal Access Token",
    },
    "GITHUB_OAUTH": {
        "pattern": r"gho_[A-Za-z0-9_]{36}",
        "confidence": 0.95,
        "description": "GitHub OAuth Token",
    },
    "GITLAB_TOKEN": {
        "pattern": r"glpat-[A-Za-z0-9\-_]{20,}",
        "confidence": 0.95,
        "description": "GitLab Personal Access Token",
    },
    "SLACK_TOKEN": {
        "pattern": r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*",
        "confidence": 0.95,
        "description": "Slack Token",
    },
    "SLACK_WEBHOOK": {
        "pattern": r"https://hooks\.slack\.com/services/T[A-Z0-9]{8}/B[A-Z0-9]{8}/[A-Za-z0-9]{24}",
        "confidence": 0.95,
        "description": "Slack Webhook URL",
    },
    "STRIPE_KEY": {
        "pattern": r"(?:sk|pk)_(?:test|live)_[A-Za-z0-9]{24,}",
        "confidence": 0.95,
        "description": "Stripe API Key",
    },
    "GOOGLE_API_KEY": {
        "pattern": r"AIza[0-9A-Za-z\-_]{35}",
        "confidence": 0.90,
        "description": "Google API Key",
    },
    "GOOGLE_OAUTH": {
        "pattern": r"[0-9]+-[a-z0-9_]{32}\.apps\.googleusercontent\.com",
        "confidence": 0.90,
        "description": "Google OAuth Client ID",
    },
    "HEROKU_API_KEY": {
        "pattern": r"(?i)heroku[_\-\.]?api[_\-\.]?key['\"]?\s*[:=]\s*['\"]?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})['\"]?",
        "confidence": 0.90,
        "description": "Heroku API Key",
    },
    "TWILIO_API_KEY": {
        "pattern": r"SK[a-z0-9]{32}",
        "confidence": 0.85,
        "description": "Twilio API Key",
    },
    "SENDGRID_API_KEY": {
        "pattern": r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}",
        "confidence": 0.95,
        "description": "SendGrid API Key",
    },
    "MAILCHIMP_API_KEY": {
        "pattern": r"[a-f0-9]{32}-us[0-9]{1,2}",
        "confidence": 0.90,
        "description": "Mailchimp API Key",
    },
    "NPM_TOKEN": {
        "pattern": r"(?i)(?:npm_)?[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}",
        "confidence": 0.85,
        "description": "NPM Token",
    },

    # Private Keys
    "RSA_PRIVATE_KEY": {
        "pattern": r"-----BEGIN RSA PRIVATE KEY-----[\s\S]*?-----END RSA PRIVATE KEY-----",
        "confidence": 1.0,
        "description": "RSA Private Key",
    },
    "DSA_PRIVATE_KEY": {
        "pattern": r"-----BEGIN DSA PRIVATE KEY-----[\s\S]*?-----END DSA PRIVATE KEY-----",
        "confidence": 1.0,
        "description": "DSA Private Key",
    },
    "EC_PRIVATE_KEY": {
        "pattern": r"-----BEGIN EC PRIVATE KEY-----[\s\S]*?-----END EC PRIVATE KEY-----",
        "confidence": 1.0,
        "description": "EC Private Key",
    },
    "OPENSSH_PRIVATE_KEY": {
        "pattern": r"-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]*?-----END OPENSSH PRIVATE KEY-----",
        "confidence": 1.0,
        "description": "OpenSSH Private Key",
    },
    "PGP_PRIVATE_KEY": {
        "pattern": r"-----BEGIN PGP PRIVATE KEY BLOCK-----[\s\S]*?-----END PGP PRIVATE KEY BLOCK-----",
        "confidence": 1.0,
        "description": "PGP Private Key",
    },

    # Authentication Tokens
    "JWT": {
        "pattern": r"eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*",
        "confidence": 0.90,
        "description": "JSON Web Token",
    },
    "BEARER_TOKEN": {
        "pattern": r"(?i)bearer\s+[a-zA-Z0-9_\-\.=]+",
        "confidence": 0.80,
        "description": "Bearer Token",
    },
    "BASIC_AUTH": {
        "pattern": r"(?i)basic\s+[a-zA-Z0-9+/=]{10,}",
        "confidence": 0.80,
        "description": "Basic Auth Token",
    },

    # Database Connection Strings
    "POSTGRES_URI": {
        "pattern": r"postgres(?:ql)?://[^\s'\"]+:[^\s'\"]+@[^\s'\"]+",
        "confidence": 0.90,
        "description": "PostgreSQL Connection String",
    },
    "MYSQL_URI": {
        "pattern": r"mysql://[^\s'\"]+:[^\s'\"]+@[^\s'\"]+",
        "confidence": 0.90,
        "description": "MySQL Connection String",
    },
    "MONGODB_URI": {
        "pattern": r"mongodb(?:\+srv)?://[^\s'\"]+:[^\s'\"]+@[^\s'\"]+",
        "confidence": 0.90,
        "description": "MongoDB Connection String",
    },
    "REDIS_URI": {
        "pattern": r"redis://[^\s'\"]+:[^\s'\"]+@[^\s'\"]+",
        "confidence": 0.90,
        "description": "Redis Connection String",
    },

    # Passwords in Code
    "PASSWORD_ASSIGNMENT": {
        "pattern": r"(?i)(?:password|passwd|pwd|secret)['\"]?\s*[:=]\s*['\"]([^'\"]{8,})['\"]",
        "confidence": 0.70,
        "description": "Password Assignment",
    },
    "PASSWORD_IN_URL": {
        "pattern": r"(?i)://[^:]+:([^@]+)@",
        "confidence": 0.75,
        "description": "Password in URL",
    },
    "PASSWORD_DISCLOSURE": {
        "pattern": r"(?i)(?:your|the|my|temp|temporary|default|initial|new)\s+password\s+(?:is|:)\s*['\"]?([A-Za-z0-9!@#$%^&*_\-+=]{6,})['\"]?",
        "confidence": 0.80,
        "description": "Natural language password disclosure",
    },
    "PASSWORD_LABELED": {
        "pattern": r"(?i)password\s*:\s*['\"]?([A-Za-z0-9!@#$%^&*_\-+=]{6,})['\"]?",
        "confidence": 0.75,
        "description": "Labeled password value",
    },

    # Generic Secrets
    "GENERIC_API_KEY": {
        "pattern": r"(?i)(?:api[_\-\.]?key|apikey)['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
        "confidence": 0.70,
        "description": "Generic API Key",
    },
    "GENERIC_SECRET": {
        "pattern": r"(?i)(?:secret|token)[_\-\.]?(?:key)?['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
        "confidence": 0.65,
        "description": "Generic Secret",
    },
}


class SecretDetector:
    """Detect secrets and credentials in text."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.name = "SecretDetector"
        self.confidence_threshold = confidence_threshold
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self.compiled_patterns = {}
        for name, config in SECRET_PATTERNS.items():
            self.compiled_patterns[name] = {
                "regex": re.compile(config["pattern"]),
                "confidence": config["confidence"],
                "description": config["description"],
            }

    def detect(self, text: str) -> List[Detection]:
        """
        Detect secrets in text.

        Args:
            text: Input text to scan

        Returns:
            List of Detection objects
        """
        detections = []

        for secret_type, config in self.compiled_patterns.items():
            for match in config["regex"].finditer(text):
                confidence = config["confidence"]

                # Skip if below threshold
                if confidence < self.confidence_threshold:
                    continue

                # Get the matched text
                matched_text = match.group(0)

                # For patterns with capture groups, extract the secret
                if match.lastindex:
                    matched_text = match.group(1)

                detection = Detection(
                    detector_name=self.name,
                    entity_type=secret_type,
                    text=matched_text,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    metadata={
                        "description": config["description"],
                        "pattern_name": secret_type,
                    },
                )
                detections.append(detection)

        return detections


def main():
    """Test the secret detector."""
    detector = SecretDetector()

    test_cases = [
        # AWS
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'",

        # GitHub
        "GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",

        # Stripe
        "stripe_key = sk_test_REPLACE_WITH_YOUR_KEY",

        # JWT
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",

        # Database
        "postgres://user:password123@localhost:5432/mydb",

        # Private Key
        """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGcYbBuMz5N5G8wVwZkXR
-----END RSA PRIVATE KEY-----""",

        # Password
        "password = 'supersecretpassword123'",

        # Clean text (should have no detections)
        "Hello, this is a normal message without any secrets.",
    ]

    print("=" * 60)
    print("SECRET DETECTOR TEST")
    print("=" * 60)

    for text in test_cases:
        print(f"\nInput: {text[:60]}...")
        detections = detector.detect(text)

        if detections:
            for d in detections:
                print(f"  Found: {d.entity_type}")
                print(f"    Text: {d.text[:40]}...")
                print(f"    Confidence: {d.confidence}")
        else:
            print("  No secrets detected")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
