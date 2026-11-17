"""Streamlit frontend for the Finance Assistant API."""

from __future__ import annotations

import math
from typing import Any

import httpx
import pandas as pd
import streamlit as st


class ApiError(RuntimeError):
    """An API request failed or returned an unexpected response."""


def _request_api(
    method: str,
    api_url: str,
    path: str,
    *,
    timeout: float = 30,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        response = httpx.request(
            method,
            f"{api_url.rstrip('/')}{path}",
            timeout=timeout,
            **kwargs,
        )
    except (httpx.RequestError, httpx.InvalidURL) as exc:
        raise ApiError(
            f"Could not connect to the Finance Assistant API at {api_url}: {exc}"
        ) from exc

    if response.is_error:
        try:
            error_body = response.json()
            detail = (
                error_body.get("detail", response.text)
                if isinstance(error_body, dict)
                else response.text
            )
        except ValueError:
            detail = response.text
        raise ApiError(f"API returned HTTP {response.status_code}: {detail}")

    try:
        result = response.json()
    except ValueError as exc:
        raise ApiError("The API returned an invalid JSON response.") from exc
    if not isinstance(result, dict):
        raise ApiError("The API returned an unexpected response format.")
    return result


def _format_number(value: Any, *, prefix: str = "", suffix: str = "") -> str:
    try:
        return f"{prefix}{float(value):,.2f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _render_chat(api_url: str, user_id: str, thread_id: str) -> None:
    conversation = (user_id, thread_id)
    if st.session_state.get("active_conversation") != conversation:
        st.session_state.active_conversation = conversation
        st.session_state.chat_messages = []
        st.session_state.history_error = None
        try:
            result = _request_api(
                "GET",
                api_url,
                "/chat/history",
                params={"user_id": user_id, "thread_id": thread_id},
            )
            st.session_state.chat_messages = [
                message
                for turn in result.get("turns", [])
                for message in (
                    {"role": "user", "content": turn["query"]},
                    {"role": "assistant", "content": turn["answer"]},
                )
            ]
        except (ApiError, KeyError, TypeError) as exc:
            st.session_state.history_error = str(exc)

    st.subheader("Finance Assistant")
    st.caption(f"Conversation: {user_id} / {thread_id}")
    if st.session_state.history_error:
        st.error(f"Could not load conversation history: {st.session_state.history_error}")
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a finance question")
    if prompt is None:
        return
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = _request_api(
                    "POST",
                    api_url,
                    "/chat",
                    timeout=120,
                    json={
                        "message": prompt,
                        "user_id": user_id,
                        "thread_id": thread_id,
                    },
                )
                answer = result["answer"]
                if not isinstance(answer, str):
                    raise ApiError("The API returned an invalid assistant response.")
            except (ApiError, KeyError, TypeError) as exc:
                st.error(f"Could not get an assistant response: {exc}")
                return
        st.markdown(answer)
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )


def _render_portfolio(api_url: str) -> None:
    st.subheader("Portfolio analysis")
    st.caption(
        "Enter ticker symbols and share quantities. Average cost is optional; "
        "analysis uses current market quotes."
    )
    with st.form("portfolio_form"):
        holdings = st.data_editor(
            pd.DataFrame(
                [{"symbol": "AAPL", "quantity": 10.0, "average_cost": None}]
            ),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "symbol": st.column_config.TextColumn("Ticker", required=True),
                "quantity": st.column_config.NumberColumn(
                    "Shares",
                    min_value=0.000001,
                    required=True,
                ),
                "average_cost": st.column_config.NumberColumn(
                    "Average cost / share",
                    min_value=0.0,
                    help="Optional purchase price per share.",
                ),
            },
        )
        submitted = st.form_submit_button("Analyze portfolio", type="primary")

    if not submitted:
        return

    positions: list[dict[str, Any]] = []
    for _, row in holdings.iterrows():
        raw_symbol = row.get("symbol")
        symbol = (
            ""
            if raw_symbol is None or pd.isna(raw_symbol)
            else str(raw_symbol).strip()
        )
        quantity = row.get("quantity")
        average_cost = row.get("average_cost")
        if not symbol and pd.isna(quantity) and pd.isna(average_cost):
            continue
        if (
            not symbol
            or pd.isna(quantity)
            or not math.isfinite(float(quantity))
            or float(quantity) <= 0
        ):
            st.error("Every holding needs a ticker and a positive share quantity.")
            return
        if not pd.isna(average_cost) and (
            not math.isfinite(float(average_cost)) or float(average_cost) < 0
        ):
            st.error("Average cost cannot be negative.")
            return
        holding: dict[str, Any] = {
            "symbol": symbol.upper(),
            "quantity": float(quantity),
        }
        if not pd.isna(average_cost):
            holding["average_cost"] = float(average_cost)
        positions.append(holding)

    if not positions:
        st.error("Add at least one holding to analyze.")
        return

    try:
        result = _request_api(
            "POST",
            api_url,
            "/portfolio/analysis",
            timeout=120,
            json={"holdings": positions},
        )
    except ApiError as exc:
        st.error(f"Portfolio analysis failed: {exc}")
        return

    st.markdown("#### Portfolio summary")
    columns = st.columns(3)
    columns[0].metric(
        "Market value",
        _format_number(result.get("total_market_value"), prefix="$"),
    )
    if "total_cost_basis" in result:
        columns[1].metric(
            "Cost basis",
            _format_number(result.get("total_cost_basis"), prefix="$"),
        )
        columns[2].metric(
            "Total gain / loss",
            _format_number(result.get("total_gain_loss"), prefix="$"),
        )
    else:
        columns[1].metric("Positions", len(result.get("positions", [])))

    position_data = result.get("positions", [])
    if not isinstance(position_data, list) or not position_data:
        st.info("The API returned no portfolio positions.")
        return
    frame = pd.DataFrame(position_data)
    display_columns = [
        column
        for column in (
            "symbol",
            "quantity",
            "current_price",
            "market_value",
            "allocation_percent",
            "average_cost",
            "gain_loss",
            "gain_loss_percent",
        )
        if column in frame.columns
    ]
    st.dataframe(frame[display_columns], hide_index=True, width="stretch")
    if "allocation_percent" in frame and "symbol" in frame:
        st.markdown("#### Allocation")
        st.bar_chart(frame.set_index("symbol")["allocation_percent"])
    if result.get("disclaimer"):
        st.caption(result["disclaimer"])


def _render_market(api_url: str) -> None:
    st.subheader("Market overview")
    with st.form("market_form"):
        symbol = st.text_input("Ticker symbol", value="AAPL", max_chars=20)
        outputsize = st.selectbox("Price history", ["compact", "full"])
        submitted = st.form_submit_button("Load market data", type="primary")

    if not submitted:
        st.info("Enter a ticker and load market data to view its dashboard.")
        return

    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        st.error("Enter a ticker symbol.")
        return

    try:
        quote_result = _request_api(
            "GET",
            api_url,
            f"/market/quote/{normalized_symbol}",
        )
        overview = _request_api(
            "GET",
            api_url,
            f"/market/overview/{normalized_symbol}",
        )
        history = _request_api(
            "GET",
            api_url,
            f"/market/history/{normalized_symbol}",
            params={"outputsize": outputsize},
            timeout=60,
        )
    except ApiError as exc:
        st.error(f"Could not load market data: {exc}")
        return

    quote = quote_result.get("quote", {})
    st.markdown(
        f"#### {overview.get('Name', normalized_symbol)} "
        f"({quote_result.get('symbol', normalized_symbol)})"
    )
    quote_columns = st.columns(4)
    quote_columns[0].metric("Price", _format_number(quote.get("05. price"), prefix="$"))
    quote_columns[1].metric(
        "Change",
        _format_number(quote.get("09. change"), prefix="$"),
    )
    quote_columns[2].metric("Change %", quote.get("10. change percent", "—"))
    quote_columns[3].metric("Volume", quote.get("06. volume", "—"))

    fundamentals = [
        ("Sector", "Sector"),
        ("Market cap", "MarketCapitalization"),
        ("P/E ratio", "PERatio"),
        ("52-week high", "52WeekHigh"),
        ("52-week low", "52WeekLow"),
    ]
    fundamental_columns = st.columns(len(fundamentals))
    for column, (label, key) in zip(fundamental_columns, fundamentals):
        value = overview.get(key, "—")
        if key == "MarketCapitalization" and value != "—":
            value = _format_number(value, prefix="$")
        column.metric(label, str(value))

    prices = history.get("prices", [])
    if not isinstance(prices, list) or not prices:
        st.info("No daily prices were returned for this ticker.")
        return
    frame = pd.DataFrame(prices)
    if not {"date", "close"}.issubset(frame.columns):
        st.error("The API returned incomplete daily price data.")
        return
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).set_index("date")
    if frame.empty:
        st.info("No valid daily prices were returned for this ticker.")
        return
    st.markdown("#### Daily closing price")
    st.line_chart(frame["close"])
    if "volume" in frame.columns:
        st.markdown("#### Daily volume")
        st.bar_chart(frame["volume"])
    with st.expander("Daily price data"):
        st.dataframe(frame.reset_index(), hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="Finance Assistant",
        layout="wide",
    )
    st.title("Finance Assistant")
    st.caption("Chat with your finance assistant and explore portfolio and market data.")

    with st.sidebar:
        st.header("Connection")
        api_url = st.text_input(
            "FastAPI base URL",
            value="http://127.0.0.1:8080",
            help="The API server URL, not the MCP server URL.",
        ).strip().rstrip("/")
        if st.button("Check API connection"):
            try:
                health = _request_api("GET", api_url, "/health", timeout=5)
                if health.get("status") == "ok":
                    st.success("Finance Assistant API is online.")
                else:
                    st.error(f"Unexpected health response: {health}")
            except ApiError as exc:
                st.error(str(exc))

        st.divider()
        st.header("Conversation")
        user_id = st.text_input("User ID", value="local-user", max_chars=128).strip()
        thread_id = st.text_input(
            "Conversation ID",
            value="default",
            max_chars=128,
        ).strip()
        if st.button("Clear conversation", disabled=not user_id or not thread_id):
            try:
                result = _request_api(
                    "DELETE",
                    api_url,
                    "/chat/history",
                    params={"user_id": user_id, "thread_id": thread_id},
                )
                if st.session_state.get("active_conversation") == (user_id, thread_id):
                    st.session_state.chat_messages = []
                st.success(f"Deleted {result.get('deleted_turns', 0)} saved turn(s).")
            except ApiError as exc:
                st.error(f"Could not clear conversation: {exc}")
        st.caption("Conversation IDs are not authentication credentials.")

    chat_tab, portfolio_tab, market_tab = st.tabs(
        ["Chat", "Portfolio", "Market"]
    )
    with chat_tab:
        _render_chat(api_url, user_id, thread_id)
    with portfolio_tab:
        _render_portfolio(api_url)
    with market_tab:
        _render_market(api_url)


if __name__ == "__main__":
    main()
