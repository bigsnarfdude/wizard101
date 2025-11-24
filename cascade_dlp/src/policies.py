"""
Policy engine for cascade_dlp.

Decides what action to take based on detections and context.
"""

from typing import List, Callable
from enum import Enum
from dataclasses import dataclass

from context import RequestContext, Destination, UserRole, SensitivityLevel
from detectors.secret_detector import Detection


class Action(Enum):
    """Actions the policy engine can take."""
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    ALERT = "alert"
    REVIEW = "review"  # Human review required


@dataclass
class PolicyDecision:
    """Result of policy evaluation."""
    action: Action
    reason: str
    policy_name: str
    detection: Detection


# Type for policy functions
PolicyFunction = Callable[[Detection, RequestContext], PolicyDecision | None]


class PolicyEngine:
    """
    Evaluate detections against policies and decide actions.

    Policies are evaluated in order. First matching policy wins.
    """

    def __init__(self):
        self.policies: List[tuple[str, PolicyFunction]] = []
        self._load_default_policies()

    def _load_default_policies(self):
        """Load default security policies."""
        # Order matters - first match wins

        # 1. Always block credentials externally
        self.add_policy("block_external_credentials", self._block_external_credentials)

        # 2. Always block private keys
        self.add_policy("block_private_keys", self._block_private_keys)

        # 3. Redact PII for external requests
        self.add_policy("redact_external_pii", self._redact_external_pii)

        # 4. Allow admins to see everything
        self.add_policy("allow_admin", self._allow_admin)

        # 5. Allow internal with audit
        self.add_policy("allow_internal", self._allow_internal)

        # 6. Default: redact unknown
        self.add_policy("default_redact", self._default_redact)

    def add_policy(self, name: str, policy_fn: PolicyFunction):
        """Add a policy to the engine."""
        self.policies.append((name, policy_fn))

    def evaluate(self, detection: Detection, context: RequestContext) -> PolicyDecision:
        """
        Evaluate a detection against all policies.

        Returns the decision from the first matching policy.
        """
        for name, policy_fn in self.policies:
            decision = policy_fn(detection, context)
            if decision:
                return decision

        # Default: block if no policy matched
        return PolicyDecision(
            action=Action.BLOCK,
            reason="No policy matched - blocking by default",
            policy_name="default_block",
            detection=detection
        )

    def evaluate_all(
        self,
        detections: List[Detection],
        context: RequestContext
    ) -> List[PolicyDecision]:
        """Evaluate all detections and return decisions."""
        return [self.evaluate(d, context) for d in detections]

    # Default Policy Implementations

    def _block_external_credentials(
        self,
        detection: Detection,
        context: RequestContext
    ) -> PolicyDecision | None:
        """Block credentials in external responses."""
        credential_types = {
            "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "GITHUB_TOKEN", "GITLAB_TOKEN",
            "SLACK_TOKEN", "STRIPE_KEY", "GOOGLE_API_KEY", "HEROKU_API_KEY",
            "SENDGRID_API_KEY", "NPM_TOKEN", "POSTGRES_URI", "MYSQL_URI",
            "MONGODB_URI", "REDIS_URI", "PASSWORD_ASSIGNMENT", "PASSWORD_IN_URL",
            "GENERIC_API_KEY", "GENERIC_SECRET", "JWT", "BEARER_TOKEN", "BASIC_AUTH"
        }

        if detection.entity_type in credential_types:
            if context.destination == Destination.EXTERNAL:
                return PolicyDecision(
                    action=Action.BLOCK,
                    reason=f"Credential {detection.entity_type} blocked for external destination",
                    policy_name="block_external_credentials",
                    detection=detection
                )
        return None

    def _block_private_keys(
        self,
        detection: Detection,
        context: RequestContext
    ) -> PolicyDecision | None:
        """Always block private keys."""
        key_types = {
            "RSA_PRIVATE_KEY", "DSA_PRIVATE_KEY", "EC_PRIVATE_KEY",
            "OPENSSH_PRIVATE_KEY", "PGP_PRIVATE_KEY"
        }

        if detection.entity_type in key_types:
            return PolicyDecision(
                action=Action.BLOCK,
                reason=f"Private key {detection.entity_type} always blocked",
                policy_name="block_private_keys",
                detection=detection
            )
        return None

    def _redact_external_pii(
        self,
        detection: Detection,
        context: RequestContext
    ) -> PolicyDecision | None:
        """Redact PII in external responses."""
        pii_types = {
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
            "IBAN_CODE", "IP_ADDRESS", "LOCATION", "DATE_TIME", "NRP",
            "MEDICAL_LICENSE", "US_BANK_NUMBER", "URL"
        }

        if detection.entity_type in pii_types:
            if context.destination == Destination.EXTERNAL:
                return PolicyDecision(
                    action=Action.REDACT,
                    reason=f"PII {detection.entity_type} redacted for external destination",
                    policy_name="redact_external_pii",
                    detection=detection
                )
        return None

    def _allow_admin(
        self,
        detection: Detection,
        context: RequestContext
    ) -> PolicyDecision | None:
        """Allow admins to see everything (with audit)."""
        if context.is_admin():
            return PolicyDecision(
                action=Action.ALLOW,
                reason=f"Admin access allowed for {detection.entity_type}",
                policy_name="allow_admin",
                detection=detection
            )
        return None

    def _allow_internal(
        self,
        detection: Detection,
        context: RequestContext
    ) -> PolicyDecision | None:
        """Allow internal requests with lower sensitivity."""
        if context.is_internal():
            if context.sensitivity_level in (SensitivityLevel.PUBLIC, SensitivityLevel.INTERNAL):
                return PolicyDecision(
                    action=Action.ALLOW,
                    reason=f"Internal access allowed for {detection.entity_type}",
                    policy_name="allow_internal",
                    detection=detection
                )
        return None

    def _default_redact(
        self,
        detection: Detection,
        context: RequestContext
    ) -> PolicyDecision | None:
        """Default: redact anything else."""
        return PolicyDecision(
            action=Action.REDACT,
            reason=f"Default redaction for {detection.entity_type}",
            policy_name="default_redact",
            detection=detection
        )


def main():
    """Test policy engine."""
    from context import create_external_context, create_internal_context, create_admin_context

    engine = PolicyEngine()

    # Test detections
    detections = [
        Detection(
            detector_name="test",
            entity_type="AWS_ACCESS_KEY",
            text="AKIAIOSFODNN7EXAMPLE",
            start=0, end=20,
            confidence=0.95,
            metadata={}
        ),
        Detection(
            detector_name="test",
            entity_type="EMAIL_ADDRESS",
            text="john@example.com",
            start=0, end=16,
            confidence=1.0,
            metadata={}
        ),
        Detection(
            detector_name="test",
            entity_type="PERSON",
            text="John Smith",
            start=0, end=10,
            confidence=0.85,
            metadata={}
        ),
    ]

    contexts = [
        ("External User", create_external_context("user123")),
        ("Internal Staff", create_internal_context("staff456")),
        ("Admin", create_admin_context("admin789")),
    ]

    print("=" * 70)
    print("POLICY ENGINE TEST")
    print("=" * 70)

    for detection in detections:
        print(f"\n{'─' * 70}")
        print(f"Detection: {detection.entity_type} = '{detection.text}'")
        print(f"{'─' * 70}")

        for context_name, context in contexts:
            decision = engine.evaluate(detection, context)
            print(f"  {context_name:20} → {decision.action.value:8} ({decision.policy_name})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
