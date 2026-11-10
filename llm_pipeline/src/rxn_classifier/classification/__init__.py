"""Multi-agent reaction classification pipeline.

Modules:
- models: records passed between stages, and the config
- llm_client: minimal LLM client wrappers
- agents: hierarchy, detailed, verifier, generator and aggregator agents
- pipeline: runs the agents in order and builds the output
- api: config-driven entry point
"""
