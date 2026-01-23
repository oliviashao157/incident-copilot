"""Streamlit UI for Incident Copilot."""

import requests
import streamlit as st

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Incident Copilot",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 AI Incident Copilot")
st.markdown(
    "Analyze incidents using AI-powered classification and RAG-based similar incident retrieval."
)


def check_api_health():
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def analyze_incident(title: str, description: str, top_k: int = 5):
    """Call the analyze endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={
                "title": title,
                "description": description,
                "top_k": top_k,
            },
            timeout=60,
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_incidents(query: str, top_k: int = 10):
    """Call the search endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


# Sidebar - API Status
with st.sidebar:
    st.header("System Status")

    healthy, health_data = check_api_health()

    if healthy:
        st.success("API Connected")
        st.metric("Index Size", health_data.get("index_size", 0))
        st.metric(
            "Classifier",
            "Loaded" if health_data.get("classifier_loaded") else "Not loaded",
        )
        st.caption(f"Version: {health_data.get('version', 'unknown')}")
    else:
        st.error("API Not Available")
        st.caption("Start the API with: `make run-api`")
        st.caption(f"Error: {health_data.get('error', 'Unknown')}")

    st.divider()
    st.header("Settings")
    top_k = st.slider("Similar incidents to retrieve", 1, 10, 5)

# Main content
tab1, tab2 = st.tabs(["📝 Analyze Incident", "🔍 Search Incidents"])

with tab1:
    st.header("New Incident Analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        title = st.text_input(
            "Incident Title",
            placeholder="e.g., High latency on payment API",
        )

        description = st.text_area(
            "Incident Description",
            height=200,
            placeholder="Describe the incident in detail...\n\nInclude:\n- What happened\n- When it started\n- Impact on users\n- Error messages observed",
        )

        if st.button("🔍 Analyze", type="primary", disabled=not healthy):
            if not title or not description:
                st.warning("Please provide both title and description")
            else:
                with st.spinner("Analyzing incident..."):
                    result = analyze_incident(title, description, top_k)

                if result.get("success"):
                    st.session_state["analysis_result"] = result["result"]
                else:
                    st.error(f"Analysis failed: {result.get('error')}")

    with col2:
        if "analysis_result" in st.session_state:
            result = st.session_state["analysis_result"]

            # Classification results
            st.subheader("Classification")
            col_cat, col_sev, col_conf = st.columns(3)

            with col_cat:
                category = result.get("predicted_category", "unknown")
                st.metric("Category", category.upper())

            with col_sev:
                severity = result.get("predicted_severity", "unknown")
                severity_colors = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                    "unknown": "⚪",
                }
                st.metric(
                    "Severity",
                    f"{severity_colors.get(severity, '⚪')} {severity.upper()}",
                )

            with col_conf:
                confidence = result.get("category_confidence", 0)
                st.metric("Confidence", f"{confidence:.0%}")

    # Full-width analysis results below
    if "analysis_result" in st.session_state:
        result = st.session_state["analysis_result"]

        st.divider()

        # Analysis summary
        st.subheader("📊 Analysis Summary")
        st.info(result.get("analysis_summary", "No summary available"))

        # Root cause hypothesis
        st.subheader("🔍 Root Cause Hypothesis")
        st.write(result.get("root_cause_hypothesis", "No hypothesis available"))

        # Recommended actions
        st.subheader("✅ Recommended Actions")
        actions = result.get("recommended_actions", [])
        if actions:
            for i, action in enumerate(actions, 1):
                st.markdown(f"{i}. {action}")
        else:
            st.write("No recommendations available")

        # Impact assessment
        st.subheader("💥 Estimated Impact")
        st.write(result.get("estimated_impact", "Unknown"))

        # Similar incidents
        st.subheader("📋 Similar Past Incidents")
        similar = result.get("similar_incidents", [])

        if similar:
            for i, sim in enumerate(similar):
                incident = sim.get("incident", {})
                score = sim.get("similarity_score", 0)

                with st.expander(
                    f"**{incident.get('title', 'Unknown')}** (Similarity: {score:.0%})"
                ):
                    cols = st.columns([1, 1, 1])
                    with cols[0]:
                        st.caption(f"ID: {incident.get('id', 'N/A')}")
                    with cols[1]:
                        st.caption(f"Category: {incident.get('category', 'unknown')}")
                    with cols[2]:
                        st.caption(f"Severity: {incident.get('severity', 'unknown')}")

                    st.markdown("**Description:**")
                    desc = incident.get("description", "")
                    st.write(desc[:500] + "..." if len(desc) > 500 else desc)

                    if incident.get("resolution"):
                        st.markdown("**Resolution:**")
                        st.success(incident["resolution"])
        else:
            st.write("No similar incidents found")

        # Citations
        citations = result.get("citations", [])
        if citations:
            st.subheader("📚 Citations")
            for cit in citations:
                with st.expander(f"Reference: {cit.get('incident_id', 'Unknown')}"):
                    st.markdown(f"**Relevant Text:** {cit.get('text', 'N/A')}")
                    st.markdown(f"**Relevance:** {cit.get('relevance', 'N/A')}")

with tab2:
    st.header("Search Past Incidents")

    search_query = st.text_input(
        "Search Query",
        placeholder="Search for incidents by keywords...",
    )

    search_k = st.slider("Number of results", 5, 50, 10, key="search_k")

    if st.button("🔍 Search", disabled=not healthy):
        if not search_query:
            st.warning("Please enter a search query")
        else:
            with st.spinner("Searching..."):
                results = search_incidents(search_query, search_k)

            if results.get("success"):
                st.subheader(f"Found {len(results.get('results', []))} incidents")

                for res in results.get("results", []):
                    with st.expander(
                        f"**{res.get('title', 'Unknown')}** "
                        f"({res.get('category', 'unknown')} / {res.get('severity', 'unknown')}) "
                        f"- Score: {res.get('similarity_score', 0):.0%}"
                    ):
                        st.caption(f"ID: {res.get('incident_id', 'N/A')}")
            else:
                st.error(f"Search failed: {results.get('error')}")

# Footer
st.divider()
st.caption(
    "AI Incident Copilot - Built with FastAPI, Streamlit, FAISS, and Claude/OpenAI"
)
