"""
Request context for cascade_dlp policy decisions.

Provides contextual information about the request for policy evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional, Set
from enum import Enum


class Destination(Enum):
    """Where the response is going."""
    INTERNAL = "internal"      # Internal dashboard, admin
    EXTERNAL = "external"      # Public API, user-facing
    LOGGING = "logging"        # Audit logs
    STORAGE = "storage"        # Database, file storage


class UserRole(Enum):
    """User permission level."""
    ADMIN = "admin"
    STAFF = "staff"
    USER = "user"
    ANONYMOUS = "anonymous"
    SYSTEM = "system"


class SensitivityLevel(Enum):
    """Data sensitivity classification."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class RequestContext:
    """
    Context for a DLP request.

    Contains information about who is requesting, where data is going,
    and what permissions apply.
    """
    # Request identification
    request_id: str = ""

    # User context
    user_id: str = ""
    user_role: UserRole = UserRole.USER
    user_email: str = ""

    # Destination context
    destination: Destination = Destination.EXTERNAL

    # Data context
    sensitivity_level: SensitivityLevel = SensitivityLevel.INTERNAL
    allowed_entity_types: Set[str] = field(default_factory=set)

    # Additional metadata
    application: str = ""
    ip_address: str = ""
    session_id: str = ""

    def is_internal(self) -> bool:
        """Check if request is internal."""
        return self.destination == Destination.INTERNAL

    def is_admin(self) -> bool:
        """Check if user has admin privileges."""
        return self.user_role in (UserRole.ADMIN, UserRole.SYSTEM)

    def can_view_entity_type(self, entity_type: str) -> bool:
        """Check if user can view this entity type."""
        if not self.allowed_entity_types:
            return True  # No restrictions
        return entity_type in self.allowed_entity_types


@dataclass
class DataProvenance:
    """
    Track where data came from and who can see it.

    Used for fine-grained access control decisions.
    """
    # Source information
    sources: Set[str] = field(default_factory=set)  # Where data came from
    source_sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL

    # Access control
    allowed_readers: Set[str] = field(default_factory=set)  # Who can see this
    owner_id: str = ""

    # Classification
    data_types: Set[str] = field(default_factory=set)  # pii, credential, etc.

    def can_access(self, user_id: str, user_role: UserRole) -> bool:
        """Check if user can access this data."""
        # Admins can access everything
        if user_role in (UserRole.ADMIN, UserRole.SYSTEM):
            return True

        # Owner can access their own data
        if user_id == self.owner_id:
            return True

        # Check allowed readers
        if self.allowed_readers and user_id in self.allowed_readers:
            return True

        return False


def create_external_context(user_id: str = "", request_id: str = "") -> RequestContext:
    """Create context for external API requests."""
    return RequestContext(
        request_id=request_id,
        user_id=user_id,
        user_role=UserRole.USER,
        destination=Destination.EXTERNAL,
        sensitivity_level=SensitivityLevel.CONFIDENTIAL,
    )


def create_internal_context(user_id: str = "", request_id: str = "") -> RequestContext:
    """Create context for internal requests."""
    return RequestContext(
        request_id=request_id,
        user_id=user_id,
        user_role=UserRole.STAFF,
        destination=Destination.INTERNAL,
        sensitivity_level=SensitivityLevel.INTERNAL,
    )


def create_admin_context(user_id: str = "", request_id: str = "") -> RequestContext:
    """Create context for admin requests."""
    return RequestContext(
        request_id=request_id,
        user_id=user_id,
        user_role=UserRole.ADMIN,
        destination=Destination.INTERNAL,
        sensitivity_level=SensitivityLevel.PUBLIC,
    )
