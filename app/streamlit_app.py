"""
Streamlit Frontend — PostgreSQL AI Agent
Communicates with the MCP server over stdio to test connections,
list schemas, and run AI-powered natural-language queries.
"""

import json
import os
import sys

import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PostgreSQL AI Agent",
    page_icon="🐘",
    layout="wide",
)

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_PATH = os.path.join(PROJECT_ROOT, "server", "mcp_server.py")
PYTHON = sys.executable  # same interpreter running streamlit


# ── MCP client helper ────────────────────────────────────────────────────────

def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Spawn the MCP server as a subprocess, send one tool call, return result."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    import asyncio

    async def _call():
        params = StdioServerParameters(
            command=PYTHON,
            args=[SERVER_PATH],
            env={**os.environ},
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                # result.content is a list of TextContent objects
                text = result.content[0].text if result.content else "{}"
                return json.loads(text)

    return asyncio.run(_call())


# ── response renderer ────────────────────────────────────────────────────────

def _render_response(data: dict):
    """Render an agent response: answer, SQL, summary, chart."""
    answer = data.get("answer", "")
    sql = data.get("sql", "")
    summary = data.get("summary", "")
    chart_json = data.get("chart_json", "")

    if answer:
        st.markdown(answer)

    if sql:
        with st.expander("SQL Query", expanded=False):
            st.code(sql, language="sql")

    if summary:
        st.caption(summary)

    if chart_json:
        try:
            import plotly.io as pio
            fig = pio.from_json(chart_json)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render chart: {exc}")


# ── session state defaults ───────────────────────────────────────────────────

if "connected" not in st.session_state:
    st.session_state.connected = False
if "credentials" not in st.session_state:
    st.session_state.credentials = {}
if "schemas" not in st.session_state:
    st.session_state.schemas = []
if "selected_schema" not in st.session_state:
    st.session_state.selected_schema = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── sidebar: credentials & schema ────────────────────────────────────────────

with st.sidebar:
    st.header("Database Connection")

    host = st.text_input("Host", value="localhost")
    port = st.number_input("Port", value=5432, min_value=1, max_value=65535)
    database = st.text_input("Database", value="postgres")
    user = st.text_input("User", value="postgres")
    password = st.text_input("Password", type="password")

    if st.button("Connect", use_container_width=True):
        with st.spinner("Testing connection…"):
            resp = call_mcp_tool(
                "test_connection",
                {
                    "host": host,
                    "port": int(port),
                    "database": database,
                    "user": user,
                    "password": password,
                },
            )
        if resp.get("success"):
            st.session_state.connected = True
            st.session_state.credentials = {
                "host": host,
                "port": int(port),
                "database": database,
                "user": user,
                "password": password,
            }
            # fetch schemas
            with st.spinner("Loading schemas…"):
                schema_resp = call_mcp_tool(
                    "list_schemas",
                    st.session_state.credentials,
                )
            if schema_resp.get("success"):
                st.session_state.schemas = schema_resp["schemas"]
            else:
                st.error(f"Failed to list schemas: {schema_resp.get('message')}")
            st.success("Connected!")
        else:
            st.session_state.connected = False
            st.error(f"Connection failed: {resp.get('message')}")

    if st.session_state.connected and st.session_state.schemas:
        st.divider()
        schema = st.selectbox(
            "Schema",
            options=st.session_state.schemas,
            index=(
                st.session_state.schemas.index("public")
                if "public" in st.session_state.schemas
                else 0
            ),
        )
        st.session_state.selected_schema = schema

    if st.session_state.connected:
        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── main area ────────────────────────────────────────────────────────────────

st.title("PostgreSQL AI Agent")

if not st.session_state.connected:
    st.info("Enter your PostgreSQL credentials in the sidebar and click **Connect**.")
    st.stop()

if not st.session_state.selected_schema:
    st.info("Select a schema from the sidebar.")
    st.stop()

# ── render chat history ──────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if "data" in msg:
                _render_response(msg["data"])
            else:
                st.markdown(msg.get("content", ""))

# ── chat input ───────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your database…"):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = call_mcp_tool(
                    "ask_agent",
                    {
                        **st.session_state.credentials,
                        "schema": st.session_state.selected_schema,
                        "question": prompt,
                    },
                )
            except Exception as exc:
                result = {
                    "success": False,
                    "answer": f"Error communicating with agent: {exc}",
                    "sql": "",
                    "summary": "",
                    "chart_json": "",
                }

        if result.get("success"):
            _render_response(result)
            st.session_state.chat_history.append(
                {"role": "assistant", "data": result}
            )
        else:
            err = result.get("answer", "Unknown error")
            st.error(err)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": err}
            )
