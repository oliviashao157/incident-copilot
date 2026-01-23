"""Category definitions and keyword mappings for incident classification."""

from src.schema import Category

# Keywords associated with each category for rule-based classification
CATEGORY_KEYWORDS: dict[Category, list[str]] = {
    Category.LATENCY: [
        "latency",
        "slow",
        "timeout",
        "delay",
        "response time",
        "performance",
        "lag",
        "p99",
        "p95",
        "percentile",
        "slo",
        "sli",
        "degraded",
        "sluggish",
    ],
    Category.OUTAGE: [
        "outage",
        "down",
        "unavailable",
        "offline",
        "crash",
        "failure",
        "unreachable",
        "502",
        "503",
        "500",
        "service down",
        "not responding",
        "dead",
        "broken",
    ],
    Category.DEPLOYMENT: [
        "deploy",
        "deployment",
        "release",
        "rollout",
        "rollback",
        "upgrade",
        "version",
        "canary",
        "blue-green",
        "ci/cd",
        "pipeline",
        "build",
        "helm",
        "kubernetes deploy",
    ],
    Category.CONFIG: [
        "config",
        "configuration",
        "misconfiguration",
        "setting",
        "env",
        "environment variable",
        "yaml",
        "json",
        "toml",
        "properties",
        "secret",
        "credential",
        "flag",
        "feature flag",
    ],
    Category.CAPACITY: [
        "capacity",
        "scale",
        "scaling",
        "memory",
        "cpu",
        "disk",
        "storage",
        "quota",
        "limit",
        "resource",
        "oom",
        "out of memory",
        "throttle",
        "autoscale",
        "hpa",
        "vpa",
    ],
    Category.DATA: [
        "data",
        "database",
        "db",
        "query",
        "sql",
        "migration",
        "schema",
        "replication",
        "backup",
        "restore",
        "corruption",
        "inconsistent",
        "deadlock",
        "transaction",
        "redis",
        "postgres",
        "mysql",
        "mongodb",
    ],
    Category.SECURITY: [
        "security",
        "vulnerability",
        "cve",
        "exploit",
        "breach",
        "unauthorized",
        "auth",
        "authentication",
        "authorization",
        "permission",
        "access",
        "ssl",
        "tls",
        "certificate",
        "injection",
        "xss",
    ],
    Category.DEPENDENCY: [
        "dependency",
        "upstream",
        "downstream",
        "third-party",
        "external",
        "api",
        "integration",
        "vendor",
        "library",
        "package",
        "module",
        "import",
        "sdk",
        "client",
    ],
    Category.NETWORK: [
        "network",
        "dns",
        "tcp",
        "udp",
        "http",
        "https",
        "ssl",
        "connection",
        "socket",
        "port",
        "firewall",
        "load balancer",
        "proxy",
        "ingress",
        "egress",
        "vpc",
        "subnet",
        "route",
    ],
}

# GitHub label to category mapping
LABEL_CATEGORY_MAP: dict[str, Category] = {
    # Latency
    "performance": Category.LATENCY,
    "slow": Category.LATENCY,
    "latency": Category.LATENCY,
    # Outage
    "bug": Category.OUTAGE,
    "crash": Category.OUTAGE,
    "outage": Category.OUTAGE,
    # Deployment
    "deployment": Category.DEPLOYMENT,
    "ci/cd": Category.DEPLOYMENT,
    "release": Category.DEPLOYMENT,
    # Config
    "configuration": Category.CONFIG,
    "config": Category.CONFIG,
    # Capacity
    "scaling": Category.CAPACITY,
    "resources": Category.CAPACITY,
    "memory": Category.CAPACITY,
    # Data
    "database": Category.DATA,
    "data": Category.DATA,
    # Security
    "security": Category.SECURITY,
    "vulnerability": Category.SECURITY,
    "cve": Category.SECURITY,
    # Dependency
    "dependency": Category.DEPENDENCY,
    "upstream": Category.DEPENDENCY,
    # Network
    "network": Category.NETWORK,
    "networking": Category.NETWORK,
    "dns": Category.NETWORK,
}


def infer_category_from_labels(labels: list[str]) -> Category:
    """Infer category from a list of labels using keyword matching."""
    labels_lower = [label.lower() for label in labels]

    for label in labels_lower:
        if label in LABEL_CATEGORY_MAP:
            return LABEL_CATEGORY_MAP[label]

    # Try partial matching
    for label in labels_lower:
        for keyword, category in LABEL_CATEGORY_MAP.items():
            if keyword in label or label in keyword:
                return category

    return Category.UNKNOWN


def infer_category_from_text(text: str) -> tuple[Category, float]:
    """Infer category from text using keyword matching. Returns category and confidence."""
    text_lower = text.lower()
    category_scores: dict[Category, int] = {cat: 0 for cat in Category}

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                category_scores[category] += 1

    max_score = max(category_scores.values())
    if max_score == 0:
        return Category.UNKNOWN, 0.0

    best_category = max(category_scores, key=lambda k: category_scores[k])
    # Simple confidence: normalize by max possible matches
    confidence = min(max_score / 3.0, 1.0)  # Cap at 1.0

    return best_category, confidence
