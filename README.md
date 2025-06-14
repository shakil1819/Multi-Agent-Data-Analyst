<div align="center">

  [![pytest Action](https://github.com/IbraheemTuffaha/python-fastapi-template/actions/workflows/pytest.yml/badge.svg)](https://github.com/IbraheemTuffaha/python-fastapi-template/actions/workflows/pytest.yml)
  [![format Action](https://github.com/IbraheemTuffaha/python-fastapi-template/actions/workflows/format.yml/badge.svg)](https://github.com/IbraheemTuffaha/python-fastapi-template/actions/workflows/format.yml)
  [![Python Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FIbraheemTuffaha%2Fpython-fastapi-template%2Fmain%2Fpyproject.toml&query=project.requires-python&label=python&color=greenlime)](https://github.com/IbraheemTuffaha/python-fastapi-template/blob/main/pyproject.toml)
  [![GitHub License](https://img.shields.io/github/license/IbraheemTuffaha/python-fastapi-template?color=greenlime)](https://github.com/IbraheemTuffaha/python-fastapi-template/blob/main/LICENSE)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shakil1819/Multi-Agent-Data-Analyst)
</div>
<div align="center">
  <a href="https://idx.google.com/import?url=https%3A%2F%2Fgithub.com%2Fshakil1819%2FgetChief-assignment">
    <picture>
      <source
        media="(prefers-color-scheme: dark)"
        srcset="https://cdn.idx.dev/btn/open_dark_32.svg">
      <source
        media="(prefers-color-scheme: light)"
        srcset="https://cdn.idx.dev/btn/open_light_32.svg">
      <img
        height="32"
        alt="Open in IDX"
        src="https://cdn.idx.dev/btn/open_purple_32.svg">
    </picture>
  </a>
</div>

![image](https://github.com/user-attachments/assets/e124c4a7-18ea-49f2-8726-f16d1207e63f)

# Multi-Agent Data Analysis System
![bfb52a27-471e-4baf-aa8d-40f7313baec4](https://github.com/user-attachments/assets/034f958e-a947-4164-9e81-f8c44b472925)


This Agentic system is designed to handle complex data analysis tasks by breaking them down into smaller subtasks and delegating them to specialized agents. The system uses a multi-agent architecture to improve performance and scalability.

## System Components

- **Agent Orchestrator**: The main controller that coordinates the work of the agents.
- **Data Analyst Agent**: Handles data analysis tasks.
- **SQL Data Analyst Agent**: Specialized in SQL-based data analysis.
- **Visualization Server**: Manages the visualization of data analysis results.

## System Design Diagram:
![screencapture-mermaid-live-view-2025-04-15-01_00_08](https://github.com/user-attachments/assets/2e56f0dc-1865-4dc5-a292-a6ff6fe317f6)


## Setup

1. Install dependencies:
```bash
uv venv
uv sync
```

2. Set up environment variables:
```bash
cp .env.example .env # OPENAI_API_KEY
```

3. Run the application:
```bash
python main.py
```

## Usage

1. Access the API at `http://localhost:8080`.
2. Use the endpoints provided by the API to interact with the system.

## Testing

1. Run tests:
```bash
python -m pytest
```

2. Run coverage report:
```bash
python -m pytest --cov=app
```

## License

MIT License

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Plotly](https://plotly.com/)
- [OpenAI](https://openai.com/)
- [IDX](https://idx.dev/)
- [Pydantic_AI](https://ai.pydantic.dev/)
- [LangGraph](https://github.com/langchain-ai/langgraph)


