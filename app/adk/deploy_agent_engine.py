"""Optional Vertex AI Agent Engine deployment entry point.

Run this only when Google Cloud credentials and Agent Engine configuration are
available. It is not imported by the FastAPI application.
"""

from __future__ import annotations

import os

from app.adk.agent import root_agent


def deploy() -> object:
    """Deploy ``root_agent`` to Vertex AI Agent Engine."""
    try:
        import vertexai
        from vertexai import agent_engines
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install google-cloud-aiplatform[adk,agent_engines] to deploy to Agent Engine."
        ) from exc

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    staging_bucket = os.getenv("GOOGLE_CLOUD_STAGING_BUCKET")
    api_key = os.getenv("VERTEX_AI_EXPRESS_API_KEY")

    if api_key:
        vertexai.init(api_key=api_key)
    else:
        if not project_id or not staging_bucket:
            raise RuntimeError(
                "Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_STAGING_BUCKET, "
                "or set VERTEX_AI_EXPRESS_API_KEY for Express Mode."
            )
        vertexai.init(
            project=project_id,
            location=location,
            staging_bucket=staging_bucket,
        )

    app = agent_engines.AdkApp(agent=root_agent, enable_tracing=True)
    return agent_engines.create(
        agent_engine=app,
        display_name=os.getenv("AGENT_ENGINE_DISPLAY_NAME", "TriageAI Complaint Supervisor"),
        requirements=[
            "google-adk>=1.0.0",
            "google-genai>=1.0.0",
            "google-cloud-aiplatform[adk,agent_engines]>=1.111",
        ],
    )


if __name__ == "__main__":
    remote_app = deploy()
    print(f"Agent Engine deployed: {remote_app.resource_name}")
