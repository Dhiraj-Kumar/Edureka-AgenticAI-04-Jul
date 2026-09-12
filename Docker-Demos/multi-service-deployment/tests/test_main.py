"""Tests for the FastAPI chat agent app in main.py.

The OpenAI/LangChain agent and the Postgres checkpointer are mocked so no
real network or database calls are made.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def mock_agent_infra():
    """Patch PostgresSaver and create_agent used inside the /chat endpoint."""
    with patch("main.PostgresSaver") as mock_saver_cls, \
            patch("main.create_agent") as mock_create_agent:

        mock_checkpointer = MagicMock()
        # Support: with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        mock_saver_cls.from_conn_string.return_value.__enter__.return_value = mock_checkpointer
        mock_saver_cls.from_conn_string.return_value.__exit__.return_value = False

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        yield {
            "saver_cls": mock_saver_cls,
            "checkpointer": mock_checkpointer,
            "create_agent": mock_create_agent,
            "agent": mock_agent,
        }


def _set_agent_reply(mock_agent_infra, reply_text):
    mock_agent_infra["agent"].invoke.return_value = {
        "messages": [MagicMock(content=reply_text)]
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_success(client, mock_agent_infra):
    _set_agent_reply(mock_agent_infra, "Hello there! 😄")

    response = client.post(
        "/chat", json={"message": "Hi", "thread_id": "abc123"})

    assert response.status_code == 200
    body = response.json()
    assert body == {"reply": "Hello there! 😄", "thread_id": "abc123"}


def test_chat_uses_default_thread_id(client, mock_agent_infra):
    _set_agent_reply(mock_agent_infra, "Sure thing!")

    response = client.post("/chat", json={"message": "Hi"})

    assert response.status_code == 200
    assert response.json()["thread_id"] == "default"


def test_chat_empty_message_returns_400(client, mock_agent_infra):
    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "message cannot be empty"
    mock_agent_infra["create_agent"].assert_not_called()


def test_chat_invokes_agent_with_expected_arguments(client, mock_agent_infra):
    _set_agent_reply(mock_agent_infra, "Done ✅")

    client.post(
        "/chat", json={"message": "What's the weather?", "thread_id": "t1"})

    mock_agent_infra["checkpointer"].setup.assert_called_once()
    mock_agent_infra["create_agent"].assert_called_once_with(
        model=main.llm,
        system_prompt=main.SYSTEM_PROMPT,
        checkpointer=mock_agent_infra["checkpointer"],
    )
    mock_agent_infra["agent"].invoke.assert_called_once_with(
        {"messages": [{"role": "user", "content": "What's the weather?"}]},
        config={"configurable": {"thread_id": "t1"}},
    )


def test_chat_does_not_call_real_openai(client, mock_agent_infra):
    """Ensure the agent factory (which wraps the OpenAI model) is only ever
    invoked through the mock, never hitting the real OpenAI API."""
    _set_agent_reply(mock_agent_infra, "Mocked reply")

    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    mock_agent_infra["create_agent"].assert_called_once()
    mock_agent_infra["agent"].invoke.assert_called_once()
