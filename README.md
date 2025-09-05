# LLM Agent API

A FastAPI backend service that provides intelligent agent capabilities through integration with Large Language Models and agent frameworks.

## Features

- FastAPI-based REST API
- LLM integration (OpenAI, Anthropic, etc.)
- Agent capabilities via LangChain
- Configurable tool system
- Comprehensive logging and monitoring

## Installation

```bash
# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Running the Application

```bash
# Run with uvicorn
uvicorn app.main:app --reload

# Or run directly
python app/main.py
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

```bash
# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy app/
```