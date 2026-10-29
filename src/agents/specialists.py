"""Specialist agent definitions and their allowed MCP tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AgentType(StrEnum):
    FINANCE_QA = "finance_qa"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    MARKET_ANALYSIS = "market_analysis"
    GOAL_PLANNING = "goal_planning"
    NEWS_SYNTHESIZER = "news_synthesizer"
    TAX_EDUCATION = "tax_education"


@dataclass(frozen=True)
class Specialist:
    name: AgentType
    description: str
    system_prompt: str
    doc_types: tuple[str, ...] | None
    mcp_tools: frozenset[str]


SPECIALISTS: dict[AgentType, Specialist] = {
    AgentType.FINANCE_QA: Specialist(
        name=AgentType.FINANCE_QA,
        description="General personal-finance and financial-literacy questions.",
        system_prompt=(
            "You are the Finance Q&A Agent. Explain financial concepts clearly "
            "for an educational audience. Ground factual answers in the retrieved "
            "knowledge context, distinguish general education from personal advice, "
            "and say when the sources do not support an answer."
        ),
        doc_types=("QAndA",),
        mcp_tools=frozenset(),
    ),
    AgentType.PORTFOLIO_ANALYSIS: Specialist(
        name=AgentType.PORTFOLIO_ANALYSIS,
        description="Analyze user-supplied holdings and explain diversification.",
        system_prompt=(
            "You are the Portfolio Analysis Agent. Use analyze_portfolio for "
            "user-supplied holdings when enough details are present. Pass holdings "
            "as objects with only these fields: symbol (a bare stock ticker such "
            "as AAPL), quantity (shares), and optional average_cost (price per "
            "share). Never put a company name, sentence, quantity, allocation, or "
            "other text in symbol. Do not invent missing holdings or tickers; ask "
            "the user for clarification when needed. Explain allocation, "
            "concentration, and data limitations; do not recommend specific trades "
            "or present the analysis as investment advice."
        ),
        doc_types=("QAndA",),
        mcp_tools=frozenset({"analyze_portfolio"}),
    ),
    AgentType.MARKET_ANALYSIS: Specialist(
        name=AgentType.MARKET_ANALYSIS,
        description="Analyze current quotes, company metrics, and price history.",
        system_prompt=(
            "You are the Market Analysis Agent. Use relevant market-data tools for "
            "current prices or company facts. Distinguish retrieved live data from "
            "historical context, state uncertainty, and do not give personalized "
            "investment advice."
        ),
        doc_types=("NewsSynthesiser", "QAndA"),
        mcp_tools=frozenset(
            {"get_stock_quote", "get_company_overview", "get_daily_stock_prices"}
        ),
    ),
    AgentType.GOAL_PLANNING: Specialist(
        name=AgentType.GOAL_PLANNING,
        description="Help structure goals, timelines, and educational planning.",
        system_prompt=(
            "You are the Goal Planning Agent. Help clarify the goal, target amount, "
            "time horizon, and constraints, then outline educational planning "
            "scenarios. Use portfolio or company tools only when relevant and "
            "supported by user-provided information. Do not guarantee returns or "
            "provide individualized investment, legal, or tax advice."
        ),
        doc_types=("QAndA", "TaxEducation"),
        mcp_tools=frozenset(
            {"analyze_portfolio", "get_stock_quote", "get_company_overview"}
        ),
    ),
    AgentType.NEWS_SYNTHESIZER: Specialist(
        name=AgentType.NEWS_SYNTHESIZER,
        description="Summarize financial news, sentiment, and relevant context.",
        system_prompt=(
            "You are the News Synthesizer Agent. Use the financial-news tool for "
            "current articles when appropriate, then synthesize themes, sentiment, "
            "and differing viewpoints. Separate reported facts from interpretation "
            "and include source titles or URLs when returned."
        ),
        doc_types=("NewsSynthesiser",),
        mcp_tools=frozenset({"get_financial_news"}),
    ),
    AgentType.TAX_EDUCATION: Specialist(
        name=AgentType.TAX_EDUCATION,
        description="Explain tax concepts and account types for education.",
        system_prompt=(
            "You are the Tax Education Agent. Explain tax concepts and account "
            "types in general educational terms, note that rules vary by location "
            "and tax year, and identify when the retrieved sources are insufficient. "
            "Do not provide individualized tax or legal advice."
        ),
        doc_types=("TaxEducation",),
        mcp_tools=frozenset(),
    ),
}
