"""LLM-based synthetic incident data generator."""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.config import LLMProvider, get_settings
from src.schema import Category, Incident, IncidentSource, Severity

# Templates for each category - used when LLM is not available
INCIDENT_TEMPLATES: dict[Category, list[dict]] = {
    Category.LATENCY: [
        {
            "title": "API response times increased to {time}ms on {service}",
            "description": "We're seeing elevated P99 latency on the {service} service. Response times have increased from normal baseline of {baseline}ms to {time}ms. This started around {start_time}. {users} users are affected. The issue appears to be related to {cause}.",
            "resolution": "Identified slow database queries and added missing index on {table} table. Deployed fix and latency returned to normal within {fix_time} minutes.",
        },
        {
            "title": "High latency on {service} endpoint affecting user experience",
            "description": "Multiple alerts triggered for slow responses on /{endpoint} endpoint. SLO breach detected with {percent}% of requests exceeding {threshold}ms. Investigation shows {cause}.",
            "resolution": "Implemented query optimization and added caching layer. Response times improved by {improvement}%.",
        },
    ],
    Category.OUTAGE: [
        {
            "title": "{service} service completely down - 503 errors",
            "description": "CRITICAL: {service} is returning 503 errors for all requests. Impact: {impact}. Started at {start_time}. All {pods} pods are showing unhealthy status. Error logs show: {error}",
            "resolution": "Root cause was {cause}. Restarted affected pods and service recovered. Implementing additional health checks.",
        },
        {
            "title": "Production {service} unavailable - customers impacted",
            "description": "Complete outage of {service} in production. Customer-facing impact started at {start_time}. Approximately {customers} customers affected. Error: {error}",
            "resolution": "Identified {cause}. Applied emergency patch and rolled back problematic changes. Service restored after {duration} minutes.",
        },
    ],
    Category.DEPLOYMENT: [
        {
            "title": "Failed deployment of {service} v{version}",
            "description": "Deployment of {service} version {version} failed during rollout. {percent}% of pods stuck in CrashLoopBackOff. Error: {error}. Previous version: {prev_version}.",
            "resolution": "Rolled back to {prev_version}. Root cause: {cause}. Fixed in {fix_version}.",
        },
        {
            "title": "Canary deployment of {service} showing errors",
            "description": "Canary deployment detecting {error_rate}% error rate on new version {version}. Baseline error rate is {baseline}%. Canary weight at {weight}%.",
            "resolution": "Halted canary rollout. Identified {cause} in new code path. Fixed and redeployed successfully.",
        },
    ],
    Category.CONFIG: [
        {
            "title": "Misconfigured {config_type} causing {service} failures",
            "description": "Incorrect {config_type} configuration deployed to {service}. Affected environments: {envs}. Error: {error}. Configuration change made at {change_time}.",
            "resolution": "Reverted configuration to previous known-good state. Updated validation to prevent similar issues.",
        },
        {
            "title": "Missing environment variable breaking {service}",
            "description": "Service {service} failing to start due to missing {env_var} environment variable. Affected pods: {pods}. This was introduced in recent config change.",
            "resolution": "Added missing {env_var} to deployment configuration. Documented required environment variables.",
        },
    ],
    Category.CAPACITY: [
        {
            "title": "OOM kills on {service} pods - memory limit exceeded",
            "description": "{service} pods being OOM killed. Memory usage spiked to {memory}GB against limit of {limit}GB. {pods} pods affected. Cause appears to be {cause}.",
            "resolution": "Increased memory limits to {new_limit}GB. Identified memory leak in {component} and deployed fix.",
        },
        {
            "title": "CPU throttling on {service} affecting performance",
            "description": "High CPU throttling detected on {service}. CPU utilization at {cpu}% with frequent throttling. Request queue backing up. HPA at max replicas ({replicas}).",
            "resolution": "Scaled up cluster nodes and increased HPA max replicas. Optimized {component} to reduce CPU usage.",
        },
    ],
    Category.DATA: [
        {
            "title": "Database replication lag on {db} affecting reads",
            "description": "Replication lag on {db} replica reached {lag} seconds. Read queries returning stale data. Primary at {primary_util}% CPU. {queries} slow queries identified.",
            "resolution": "Killed long-running queries and optimized slow query patterns. Replication caught up within {catch_up} minutes.",
        },
        {
            "title": "Failed database migration on {db}",
            "description": "Migration {migration_id} failed on {db}. Error: {error}. Table {table} in inconsistent state. Affects {percent}% of records.",
            "resolution": "Manually fixed inconsistent records. Applied corrected migration. Added migration validation tests.",
        },
    ],
    Category.SECURITY: [
        {
            "title": "SSL certificate expired on {service}",
            "description": "SSL certificate for {domain} expired. All HTTPS traffic to {service} failing. Certificate expired at {expire_time}. Monitoring alert missed.",
            "resolution": "Renewed certificate and deployed. Implemented certificate expiry monitoring with {alert_days} day warning.",
        },
        {
            "title": "Authentication failures spike on {service}",
            "description": "Abnormal spike in authentication failures: {failures} failures in {window} minutes. Normal rate: {normal_rate}. Source IPs: {ips}. Possible {attack_type}.",
            "resolution": "Implemented rate limiting and blocked suspicious IPs. Added additional monitoring for auth anomalies.",
        },
    ],
    Category.DEPENDENCY: [
        {
            "title": "Upstream {dependency} API returning errors",
            "description": "External dependency {dependency} returning {error_code} errors. Started at {start_time}. {percent}% of our requests failing. Impact: {impact}.",
            "resolution": "Implemented circuit breaker for {dependency}. Added fallback mechanism. Dependency recovered after {duration}.",
        },
        {
            "title": "Third-party {service} rate limiting our requests",
            "description": "{service} API rate limiting triggered. Receiving 429 errors. Current usage: {usage} requests/min. Limit: {limit} requests/min.",
            "resolution": "Implemented request batching and caching to reduce API calls. Requested rate limit increase from provider.",
        },
    ],
    Category.NETWORK: [
        {
            "title": "DNS resolution failures for {domain}",
            "description": "Intermittent DNS resolution failures for {domain}. {percent}% of requests timing out. Affects services: {services}. DNS provider: {provider}.",
            "resolution": "Switched to backup DNS resolver. Worked with {provider} to resolve issue. Added DNS caching.",
        },
        {
            "title": "Load balancer health check failures on {service}",
            "description": "Load balancer marking {service} instances as unhealthy. {unhealthy} of {total} instances failing health checks. Health check timeout: {timeout}s.",
            "resolution": "Adjusted health check timeouts and endpoints. Fixed slow startup affecting health check response.",
        },
    ],
}

SERVICES = [
    "user-service",
    "payment-api",
    "auth-service",
    "notification-service",
    "order-processor",
    "inventory-service",
    "search-api",
    "recommendation-engine",
    "analytics-pipeline",
    "gateway",
]

DATABASES = ["postgres-primary", "mysql-cluster", "mongodb-atlas", "redis-cache", "elasticsearch"]


def _generate_synthetic_incident_local(category: Category, seed: int) -> Incident:
    """Generate a synthetic incident using templates (no LLM)."""
    random.seed(seed)

    templates = INCIDENT_TEMPLATES.get(category, INCIDENT_TEMPLATES[Category.OUTAGE])
    template = random.choice(templates)

    # Generate random values for placeholders
    service = random.choice(SERVICES)
    db = random.choice(DATABASES)
    start_time = (datetime.now() - timedelta(hours=random.randint(1, 48))).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    replacements = {
        "{service}": service,
        "{db}": db,
        "{time}": str(random.randint(500, 5000)),
        "{baseline}": str(random.randint(50, 200)),
        "{start_time}": start_time,
        "{users}": str(random.randint(100, 10000)),
        "{cause}": random.choice(
            ["database connection pool exhaustion", "increased traffic", "memory pressure", "GC pauses"]
        ),
        "{table}": random.choice(["users", "orders", "transactions", "sessions"]),
        "{fix_time}": str(random.randint(5, 30)),
        "{endpoint}": random.choice(["api/v1/users", "api/v2/orders", "health", "metrics"]),
        "{percent}": str(random.randint(10, 90)),
        "{threshold}": str(random.randint(200, 1000)),
        "{improvement}": str(random.randint(50, 90)),
        "{impact}": random.choice(
            ["all customers affected", "degraded checkout flow", "API consumers impacted"]
        ),
        "{pods}": str(random.randint(3, 20)),
        "{error}": random.choice(
            [
                "Connection refused",
                "OOMKilled",
                "ImagePullBackOff",
                "CrashLoopBackOff",
                "Timeout exceeded",
            ]
        ),
        "{customers}": str(random.randint(1000, 100000)),
        "{duration}": str(random.randint(5, 120)),
        "{version}": f"{random.randint(1, 5)}.{random.randint(0, 20)}.{random.randint(0, 100)}",
        "{prev_version}": f"{random.randint(1, 5)}.{random.randint(0, 20)}.{random.randint(0, 100)}",
        "{fix_version}": f"{random.randint(1, 5)}.{random.randint(0, 20)}.{random.randint(0, 100)}",
        "{error_rate}": str(random.uniform(1, 25)),
        "{weight}": str(random.choice([5, 10, 20, 25])),
        "{config_type}": random.choice(
            ["database connection", "feature flag", "rate limit", "timeout"]
        ),
        "{envs}": random.choice(["production", "staging, production", "all environments"]),
        "{change_time}": start_time,
        "{env_var}": random.choice(["DATABASE_URL", "API_KEY", "SECRET_KEY", "REDIS_HOST"]),
        "{memory}": str(random.uniform(4, 16)),
        "{limit}": str(random.choice([2, 4, 8])),
        "{new_limit}": str(random.choice([4, 8, 16])),
        "{component}": random.choice(["request handler", "cache layer", "serializer", "parser"]),
        "{cpu}": str(random.randint(85, 99)),
        "{replicas}": str(random.randint(10, 50)),
        "{lag}": str(random.randint(30, 300)),
        "{primary_util}": str(random.randint(70, 99)),
        "{queries}": str(random.randint(5, 50)),
        "{catch_up}": str(random.randint(5, 30)),
        "{migration_id}": f"V{random.randint(100, 999)}",
        "{domain}": f"{service}.example.com",
        "{expire_time}": start_time,
        "{alert_days}": str(random.choice([14, 30, 60])),
        "{failures}": str(random.randint(1000, 50000)),
        "{window}": str(random.choice([5, 10, 15])),
        "{normal_rate}": str(random.randint(10, 100)),
        "{ips}": f"{random.randint(1, 255)}.{random.randint(0, 255)}.x.x",
        "{attack_type}": random.choice(["brute force attack", "credential stuffing", "bot activity"]),
        "{dependency}": random.choice(["Stripe", "Twilio", "SendGrid", "AWS S3", "Cloudflare"]),
        "{error_code}": random.choice(["500", "502", "503", "504"]),
        "{usage}": str(random.randint(500, 2000)),
        "{provider}": random.choice(["Route53", "Cloudflare", "Google Cloud DNS"]),
        "{services}": ", ".join(random.sample(SERVICES, k=random.randint(2, 4))),
        "{unhealthy}": str(random.randint(1, 5)),
        "{total}": str(random.randint(5, 10)),
        "{timeout}": str(random.choice([5, 10, 30])),
    }

    title = template["title"]
    description = template["description"]
    resolution = template.get("resolution", "")

    for key, value in replacements.items():
        title = title.replace(key, value)
        description = description.replace(key, value)
        resolution = resolution.replace(key, value)

    # Determine severity based on category and keywords
    severity = Severity.MEDIUM
    if category == Category.OUTAGE or "CRITICAL" in description:
        severity = random.choice([Severity.CRITICAL, Severity.HIGH])
    elif category in [Category.SECURITY, Category.DATA]:
        severity = random.choice([Severity.HIGH, Severity.MEDIUM])
    elif category in [Category.LATENCY, Category.CAPACITY]:
        severity = random.choice([Severity.MEDIUM, Severity.HIGH])
    else:
        severity = random.choice([Severity.LOW, Severity.MEDIUM])

    # Create incident
    created_at = datetime.now() - timedelta(days=random.randint(1, 365))
    resolved_at = created_at + timedelta(minutes=random.randint(10, 480)) if resolution else None

    return Incident(
        id=f"syn-{uuid.uuid4().hex[:8]}",
        title=title,
        description=description,
        category=category,
        severity=severity,
        source=IncidentSource.SYNTHETIC,
        created_at=created_at,
        resolved_at=resolved_at,
        resolution=resolution if resolution else None,
        labels=[category.value, severity.value],
        metadata={"generator": "template", "seed": seed},
    )


class SyntheticGenerator:
    """Generate synthetic incident data using LLM or templates."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.settings = get_settings()
        self._llm_client = None

    def _get_llm_client(self):
        """Lazily initialize LLM client."""
        if self._llm_client is not None:
            return self._llm_client

        if self.settings.llm_provider == LLMProvider.ANTHROPIC:
            import anthropic

            self._llm_client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        else:
            import openai

            self._llm_client = openai.OpenAI(api_key=self.settings.openai_api_key)

        return self._llm_client

    def _generate_with_llm(self, category: Category, seed: int) -> Incident:
        """Generate a synthetic incident using LLM."""
        prompt = f"""Generate a realistic incident report for a software system.

Category: {category.value}
Requirements:
- Create a realistic incident title (1 line, specific and actionable)
- Create a detailed description (2-4 paragraphs) including:
  - What happened
  - When it started
  - Impact on users/systems
  - Initial observations
- Create a resolution description (1-2 paragraphs) with:
  - Root cause
  - Steps taken to resolve
  - Preventive measures

Output as JSON with these exact fields:
{{
    "title": "...",
    "description": "...",
    "resolution": "...",
    "severity": "critical|high|medium|low"
}}

Be specific with service names, metrics, and technical details. Make it sound like a real incident from a production system."""

        try:
            if self.settings.llm_provider == LLMProvider.ANTHROPIC:
                response = self._get_llm_client().messages.create(
                    model=self.settings.llm_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text
            else:
                response = self._get_llm_client().chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                )
                content = response.choices[0].message.content

            # Parse JSON from response
            # Try to extract JSON from the response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            severity_map = {
                "critical": Severity.CRITICAL,
                "high": Severity.HIGH,
                "medium": Severity.MEDIUM,
                "low": Severity.LOW,
            }

            created_at = datetime.now() - timedelta(days=random.randint(1, 365))
            resolved_at = created_at + timedelta(minutes=random.randint(10, 480))

            return Incident(
                id=f"syn-{uuid.uuid4().hex[:8]}",
                title=data["title"],
                description=data["description"],
                category=category,
                severity=severity_map.get(data.get("severity", "medium").lower(), Severity.MEDIUM),
                source=IncidentSource.SYNTHETIC,
                created_at=created_at,
                resolved_at=resolved_at,
                resolution=data.get("resolution"),
                labels=[category.value],
                metadata={"generator": "llm", "seed": seed},
            )

        except Exception as e:
            # Fallback to template generation
            print(f"LLM generation failed: {e}, falling back to template")
            return _generate_synthetic_incident_local(category, seed)

    def generate(
        self,
        count: int = 100,
        categories: Optional[list[Category]] = None,
        seed: int = 42,
    ) -> list[Incident]:
        """Generate synthetic incidents.

        Args:
            count: Total number of incidents to generate
            categories: Categories to generate (defaults to all except UNKNOWN)
            seed: Random seed for reproducibility

        Returns:
            List of generated incidents
        """
        random.seed(seed)

        if categories is None:
            categories = [c for c in Category if c != Category.UNKNOWN]

        incidents = []
        incidents_per_category = count // len(categories)
        remainder = count % len(categories)

        for i, category in enumerate(categories):
            n = incidents_per_category + (1 if i < remainder else 0)
            for j in range(n):
                incident_seed = seed + i * 1000 + j
                if self.use_llm:
                    try:
                        incident = self._generate_with_llm(category, incident_seed)
                    except Exception:
                        incident = _generate_synthetic_incident_local(category, incident_seed)
                else:
                    incident = _generate_synthetic_incident_local(category, incident_seed)
                incidents.append(incident)

        return incidents

    def save_to_jsonl(self, incidents: list[Incident], output_path: Path) -> None:
        """Save incidents to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for incident in incidents:
                f.write(incident.model_dump_json() + "\n")
