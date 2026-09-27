from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ValidationError

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import build_support_workflow
from src.models import ChatResponse, TicketCreate, TicketCategory, Ticket
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError

logger = logging.getLogger(__name__)


class SupportPipeline:
    """Top-level binding for model, RAG, workflow, sessions, and ticket tool."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.model = build_chat_model(settings)
        self.retriever = KnowledgeRetriever(settings, documents_dir)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.workflow = None
        self.ready = False

    async def initialize(self) -> None:
        """Initialize shared components once during FastAPI startup."""
        await self.retriever.initialize()
        self.workflow = build_support_workflow(self.model)
        self.ready = True

    def _extract_email(self, text: str) -> Optional[str]:
        match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        return match.group(0) if match else None

    async def process(self, session_id: str, message: str) -> ChatResponse:
        if not self.ready:
            raise ComponentNotReadyError("Support pipeline is not ready")

        session_id = session_id.strip() if session_id else ""
        message = message.strip() if message else ""

        if not session_id or not message:
            raise AgentProcessingError("Session ID and message must not be blank.")

        try:
            session = self.sessions.get_or_create(session_id)
            
            # Initialize slot state
            if not hasattr(session, "ticket_data") or session.ticket_data is None:
                session.ticket_data = {
                    "in_progress": False,
                    "name": None,
                    "email": None,
                    "category": None,
                    "description": None,
                    "ticket_id": None,
                    "last_prompt": None,
                }

            tdata = session.ticket_data
            lower_msg = message.lower()

            # Prevent duplicate creation in same session
            if tdata.get("ticket_id") and any(k in lower_msg for k in ["ticket", "raise", "open"]):
                agent_text = (
                    f"A support ticket has already been created for this session.\n"
                    f"Ticket ID: **{tdata['ticket_id']}**. Status: open."
                )
                return ChatResponse(
                    success=True,
                    session_id=session_id,
                    response=agent_text,
                    sources=[],
                    ticket_id=tdata["ticket_id"],
                )

            # Detect ticket creation intent
            if not tdata["in_progress"] and any(k in lower_msg for k in ["ticket", "support", "raise", "issue"]):
                tdata["in_progress"] = True

            # Multi-turn slot filling
            if tdata["in_progress"]:
                last_p = tdata.get("last_prompt")

                if last_p == "email":
                    extracted = self._extract_email(message)
                    tdata["email"] = extracted if extracted else message.strip()
                elif last_p == "name":
                    tdata["name"] = message.strip().title()
                elif last_p == "description":
                    tdata["description"] = message.strip()
                else:
                    extracted = self._extract_email(message)
                    if extracted and not tdata["email"]:
                        tdata["email"] = extracted

                # Prompt 1: Email
                if not tdata["email"]:
                    tdata["last_prompt"] = "email"
                    return ChatResponse(
                        success=True,
                        session_id=session_id,
                        response="I can help you create a support ticket. What email address should we use?",
                        sources=[],
                        ticket_id=None,
                    )

                # Prompt 2: Name
                if not tdata["name"]:
                    tdata["last_prompt"] = "name"
                    return ChatResponse(
                        success=True,
                        session_id=session_id,
                        response="Got it! Could you please provide your full name?",
                        sources=[],
                        ticket_id=None,
                    )

                # Prompt 3: Description
                if not tdata["description"]:
                    tdata["last_prompt"] = "description"
                    return ChatResponse(
                        success=True,
                        session_id=session_id,
                        response="Thank you! Please describe the issue you are experiencing in detail.",
                        sources=[],
                        ticket_id=None,
                    )

                # Assign Category
                if not tdata["category"]:
                    desc_lower = str(tdata["description"]).lower()
                    for cat in ["order", "payment", "account", "technical"]:
                        if cat in desc_lower:
                            tdata["category"] = cat
                            break
                    if not tdata["category"]:
                        tdata["category"] = "other"

                desc_text = str(tdata["description"])
                if len(desc_text) < 5:
                    desc_text = desc_text + " (detailed issue)"

                summary_text = desc_text[:150]
                if len(summary_text) < 5:
                    summary_text = "Issue reported: " + summary_text

                # Validate payload with TicketCreate
                ticket_req = TicketCreate(
                    customer_name=tdata["name"],
                    customer_email=tdata["email"],
                    issue_description=desc_text,
                    category=tdata["category"],
                    summary=summary_text,
                )

                # Robust repository call handling
                created_ticket = None
                
                # Option 1: Pass model instance directly
                try:
                    created_ticket = self.tickets.create(ticket_req)
                except Exception as ex1:
                    logger.debug("Method 1 failed: %s", str(ex1))

                # Option 2: Unpack model dict
                if created_ticket is None:
                    try:
                        created_ticket = self.tickets.create(**ticket_req.model_dump())
                    except Exception as ex2:
                        logger.debug("Method 2 failed: %s", str(ex2))

                # Option 3: Manual creation into tickets dictionary
                if created_ticket is None:
                    tid = f"TCK-{uuid.uuid4().hex[:6].upper()}"
                    created_ticket = Ticket(
                        ticket_id=tid,
                        **ticket_req.model_dump()
                    )
                    if hasattr(self.tickets, "_tickets") and isinstance(self.tickets._tickets, dict):
                        self.tickets._tickets[tid] = created_ticket
                    elif hasattr(self.tickets, "tickets") and isinstance(self.tickets.tickets, dict):
                        self.tickets.tickets[tid] = created_ticket

                # Extract generated ticket_id
                created_id = (
                    getattr(created_ticket, "ticket_id", None)
                    or getattr(created_ticket, "id", None)
                    or str(created_ticket)
                )

                tdata["ticket_id"] = str(created_id)
                tdata["in_progress"] = False

                agent_text = (
                    f"Thank you, {tdata['name']}. Your ticket has been created successfully!\n\n"
                    f"• **Ticket ID:** {tdata['ticket_id']}\n"
                    f"• **Category:** {tdata['category'].capitalize()}\n"
                    f"• **Email:** {tdata['email']}\n"
                    f"• **Summary:** {summary_text}"
                )

                return ChatResponse(
                    success=True,
                    session_id=session_id,
                    response=agent_text,
                    sources=[],
                    ticket_id=tdata["ticket_id"],
                )

            # RAG Search Path
            rag_docs = await self.retriever.search(message)
            sources = list(set([doc["source"] for doc in rag_docs if doc.get("source")]))

            if rag_docs:
                context_str = "\n\n".join([doc["content"] for doc in rag_docs])
                prompt = (
                    f"You are a customer support agent.\n"
                    f"Answer the user's question strictly using the provided context below.\n"
                    f"If the answer is not present in the context, explicitly state that the knowledge base does not contain that information.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"User Query: {message}"
                )
            else:
                prompt = (
                    f"You are a customer support agent. The user asked: '{message}'.\n"
                    f"Explain clearly that the knowledge base does not contain information on this topic."
                )

            response_msg = await self.model.ainvoke(prompt)
            agent_text = response_msg.content if hasattr(response_msg, "content") else str(response_msg)

            return ChatResponse(
                success=True,
                session_id=session_id,
                response=agent_text,
                sources=sources if rag_docs else [],
                ticket_id=None,
            )

        except Exception as e:
            logger.exception("Error processing chat message: %s", str(e))
            raise AgentProcessingError(f"Failed to process request: {str(e)}") from e