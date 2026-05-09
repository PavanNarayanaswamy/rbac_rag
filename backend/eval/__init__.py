"""RAG + RBAC evaluation harness.

Submodules
----------
golden_dataset : Hand-labelled test cases.
metrics        : Retrieval, generation, and RBAC isolation metrics.
rbac_tests     : Black-box security tests against the live FastAPI app.
llm_judge      : Optional RAGAS-style faithfulness / relevance judge (Ollama).
run_evaluation : Orchestrator + report generator.
"""
