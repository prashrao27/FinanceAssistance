"""LLM-based workflow routing to the six finance specialists."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agents.specialists import AgentType


class RouteDecision(BaseModel):
    """Structured result returned by the workflow router."""

    agent: Literal[
        "finance_qa",
        "portfolio_analysis",
        "market_analysis",
        "goal_planning",
        "news_synthesizer",
        "tax_education",
    ] = Field(description="The single specialist best suited to answer the request.")
    rationale: str = Field(description="Short explanation for the specialist choice.")


ROUTER_INSTRUCTIONS = f"""
Route the user's request to exactly one best-fit specialist:
{AgentType.FINANCE_QA.value}: general financial education and finance concepts.
{AgentType.PORTFOLIO_ANALYSIS.value}: analyze holdings, allocation, or diversification.
{AgentType.MARKET_ANALYSIS.value}: current prices, company fundamentals, or market trends.
{AgentType.GOAL_PLANNING.value}: savings goals, milestones, timelines, or planning scenarios.
{AgentType.NEWS_SYNTHESIZER.value}: financial news, article summaries, or sentiment.
{AgentType.TAX_EDUCATION.value}: tax concepts, tax-advantaged accounts, or tax education.
Choose tax education for tax/account-type questions, even if they mention a goal.
Choose market analysis for current ticker/price questions. Choose portfolio analysis
when the user asks to assess existing holdings. Do not answer the query here.
""".strip()
