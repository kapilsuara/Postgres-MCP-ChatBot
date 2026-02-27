"""
MCP Server — PostgreSQL AI Agent
Exposes 3 MCP tools over stdio:
  test_connection, list_schemas, ask_agent
The ask_agent tool runs a multi-turn agent loop using Claude Opus 4.6
with extended thinking and 7 internal tools for schema discovery,
SQL execution, and chart generation.
"""

import json
import logging
import os
import re
import sys
import textwrap
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

import anthropic
import psycopg
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ── bootstrap ────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "..", "logs", "app.log")
        ),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger(__name__)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

mcp = FastMCP("postgres-agent")

# ── helpers ──────────────────────────────────────────────────────────────────

DDL_PATTERN = re.compile(
    r"\b(DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b", re.IGNORECASE
)
MAX_RESULT_CHARS = 12_000
MAX_AGENT_TURNS = 25
SELECT_ROW_LIMIT = 500


def _json_default(obj: Any) -> Any:
    """Serialise types that json.dumps does not handle natively."""
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, memoryview):
        return obj.tobytes().decode("utf-8", errors="replace")
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… (truncated at {limit} chars)"


def _connect(creds: dict) -> psycopg.Connection:
    return psycopg.connect(
        host=creds["host"],
        port=int(creds.get("port", 5432)),
        dbname=creds["database"],
        user=creds["user"],
        password=creds["password"],
        connect_timeout=10,
    )

#prevent sql injection
def _safe_table_name(schema: str, table: str) -> str:
    """Quote identifiers to prevent SQL injection."""
    s = psycopg.sql.Identifier(schema)
    t = psycopg.sql.Identifier(table)
    return psycopg.sql.SQL("{}.{}").format(s, t)


# ── PostgresAgent ────────────────────────────────────────────────────────────


class PostgresAgent:
    """Multi-turn agent that uses Claude extended thinking to answer
    natural-language questions against a PostgreSQL database."""

    INTERNAL_TOOLS = [
        {
            "name": "list_tables",
            "description": "List all tables in the selected schema. Call this first to discover what data is available.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "describe_table",
            "description": "Get column names, data types, nullable flags, and constraints for a table.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Table name (without schema prefix)",
                    }
                },
                "required": ["table_name"],
            },
        },
        {
            "name": "preview_table",
            "description": "Return 3 sample rows from a table so you can understand the actual data values.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Table name (without schema prefix)",
                    }
                },
                "required": ["table_name"],
            },
        },
        {
            "name": "run_sql",
            "description": (
                "Execute a SQL statement. SELECT queries return rows as JSON. "
                "DML (INSERT/UPDATE/DELETE) is allowed and committed. "
                "DDL (DROP/ALTER/CREATE/TRUNCATE/GRANT/REVOKE) is blocked."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL statement to execute",
                    }
                },
                "required": ["sql"],
            },
        },
        {
            "name": "analyze_table_statistics",
            "description": "Get row count, distinct counts, and null counts for every column in a table.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Table name (without schema prefix)",
                    }
                },
                "required": ["table_name"],
            },
        },
        {
            "name": "generate_chart",
            "description": (
                "Build a Plotly chart and return its JSON representation. "
                "Use this when a visual would help the user understand the data. "
                "Supported chart types: bar, line, scatter, pie, histogram, heatmap."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": [
                            "bar",
                            "line",
                            "scatter",
                            "pie",
                            "histogram",
                            "heatmap",
                        ],
                        "description": "Type of Plotly chart to create",
                    },
                    "chart_title": {
                        "type": "string",
                        "description": "Title for the chart",
                    },
                    "x_data": {
                        "type": "array",
                        "description": "X-axis values",
                    },
                    "y_data": {
                        "type": "array",
                        "description": "Y-axis values",
                    },
                    "x_label": {
                        "type": "string",
                        "description": "X-axis label",
                    },
                    "y_label": {
                        "type": "string",
                        "description": "Y-axis label",
                    },
                    "labels": {
                        "type": "array",
                        "description": "Labels for pie chart slices",
                    },
                    "values": {
                        "type": "array",
                        "description": "Values for pie chart slices",
                    },
                    "z_data": {
                        "type": "array",
                        "description": "Z-axis values (for heatmap — 2D array)",
                    },
                },
                "required": ["chart_type", "chart_title"],
            },
        },
        {
            "name": "final_answer",
            "description": (
                "Return the final answer to the user. You MUST call this tool "
                "exactly once to deliver the completed response."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The natural-language answer to the user's question",
                    },
                    "sql": {
                        "type": "string",
                        "description": "The SQL query used (if any)",
                    },
                    "summary": {
                        "type": "string",
                        "description": "A brief summary of what was found",
                    },
                    "chart_json": {
                        "type": "string",
                        "description": "Plotly figure JSON string (if a chart was generated)",
                    },
                },
                "required": ["answer"],
            },
        },
    ]

    # ── construction ─────────────────────────────────────────────────────────
    def __init__(self, creds: dict, schema: str):
        self.creds = creds
        self.schema = schema
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── internal tool dispatch ───────────────────────────────────────────────

    def _exec_tool(self, name: str, inp: dict) -> str:
        """Execute an internal tool and return a string result."""
        try:
            if name == "list_tables":
                return self._tool_list_tables()
            if name == "describe_table":
                return self._tool_describe_table(inp["table_name"])
            if name == "preview_table":
                return self._tool_preview_table(inp["table_name"])
            if name == "run_sql":
                return self._tool_run_sql(inp["sql"])
            if name == "analyze_table_statistics":
                return self._tool_analyze_table_statistics(inp["table_name"])
            if name == "generate_chart":
                return self._tool_generate_chart(inp)
            if name == "final_answer":
                return "__FINAL__"
            return f"Unknown tool: {name}"
        except Exception as exc:
            log.exception("Tool %s failed", name)
            return f"ERROR: {exc}"

    # ── tool implementations ─────────────────────────────────────────────────

    def _tool_list_tables(self) -> str:
        with _connect(self.creds) as conn:
            rows = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s AND table_type = 'BASE TABLE' "
                "ORDER BY table_name",
                (self.schema,),
            ).fetchall()
        tables = [r[0] for r in rows]
        return json.dumps(tables, default=_json_default)

    def _tool_describe_table(self, table: str) -> str:
        with _connect(self.creds) as conn:
            cols = conn.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (self.schema, table),
            ).fetchall()

            constraints = conn.execute(
                """
                SELECT tc.constraint_name, tc.constraint_type,
                       kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.table_schema = %s AND tc.table_name = %s
                """,
                (self.schema, table),
            ).fetchall()

        result = {
            "columns": [
                {
                    "name": c[0],
                    "type": c[1],
                    "nullable": c[2],
                    "default": c[3],
                }
                for c in cols
            ],
            "constraints": [
                {"name": c[0], "type": c[1], "column": c[2]} for c in constraints
            ],
        }
        return _truncate(json.dumps(result, default=_json_default))

    def _tool_preview_table(self, table: str) -> str:
        safe = _safe_table_name(self.schema, table)
        query = psycopg.sql.SQL("SELECT * FROM {} LIMIT 3").format(safe)
        with _connect(self.creds) as conn:
            cur = conn.execute(query)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
        data = [dict(zip(cols, row)) for row in rows]
        return _truncate(json.dumps(data, default=_json_default))

    def _tool_run_sql(self, sql: str) -> str:
        # Block DDL
        if DDL_PATTERN.search(sql):
            return "ERROR: DDL statements (DROP/ALTER/CREATE/TRUNCATE/GRANT/REVOKE) are not allowed."

        # Auto-add LIMIT for SELECT without one
        stripped = sql.strip().rstrip(";")
        if stripped.upper().startswith("SELECT") and "LIMIT" not in stripped.upper():
            sql = f"{stripped} LIMIT {SELECT_ROW_LIMIT}"

        is_select = stripped.upper().startswith("SELECT")

        with _connect(self.creds) as conn:
            cur = conn.execute(sql)
            if is_select:
                cols = [d.name for d in cur.description]
                rows = cur.fetchall()
                data = [dict(zip(cols, row)) for row in rows]
                result = json.dumps(data, default=_json_default)
            else:
                conn.commit()
                result = f"Statement executed successfully. Rows affected: {cur.rowcount}"

        return _truncate(result)

    def _tool_analyze_table_statistics(self, table: str) -> str:
        safe = _safe_table_name(self.schema, table)

        with _connect(self.creds) as conn:
            # row count
            row_count = conn.execute(
                psycopg.sql.SQL("SELECT COUNT(*) FROM {}").format(safe)
            ).fetchone()[0]

            # column info
            cols = conn.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (self.schema, table),
            ).fetchall()

            stats = {"row_count": row_count, "columns": {}}
            for (col_name,) in cols:
                col_id = psycopg.sql.Identifier(col_name)
                q = psycopg.sql.SQL(
                    "SELECT COUNT(DISTINCT {col}) AS distinct_count, "
                    "SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count "
                    "FROM {tbl}"
                ).format(col=col_id, tbl=safe)
                r = conn.execute(q).fetchone()
                stats["columns"][col_name] = {
                    "distinct": r[0],
                    "nulls": r[1],
                }

        return _truncate(json.dumps(stats, default=_json_default))

    def _tool_generate_chart(self, inp: dict) -> str:
        import plotly.graph_objects as go

        chart_type = inp["chart_type"]
        title = inp["chart_title"]
        fig = go.Figure()

        if chart_type == "bar":
            fig.add_trace(
                go.Bar(
                    x=inp.get("x_data", []),
                    y=inp.get("y_data", []),
                )
            )
        elif chart_type == "line":
            fig.add_trace(
                go.Scatter(
                    x=inp.get("x_data", []),
                    y=inp.get("y_data", []),
                    mode="lines+markers",
                )
            )
        elif chart_type == "scatter":
            fig.add_trace(
                go.Scatter(
                    x=inp.get("x_data", []),
                    y=inp.get("y_data", []),
                    mode="markers",
                )
            )
        elif chart_type == "pie":
            fig.add_trace(
                go.Pie(
                    labels=inp.get("labels", []),
                    values=inp.get("values", []),
                )
            )
        elif chart_type == "histogram":
            fig.add_trace(go.Histogram(x=inp.get("x_data", [])))
        elif chart_type == "heatmap":
            fig.add_trace(
                go.Heatmap(
                    z=inp.get("z_data", []),
                    x=inp.get("x_data", []),
                    y=inp.get("y_data", []),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=inp.get("x_label", ""),
            yaxis_title=inp.get("y_label", ""),
            template="plotly_white",
        )

        chart_json = fig.to_json()
        return chart_json

    # ── agent loop ───────────────────────────────────────────────────────────

    def run(self, question: str) -> dict:
        """Run the full agent loop and return the final answer dict."""
        system_prompt = textwrap.dedent(f"""\
            You are a PostgreSQL data analyst agent. You help users explore and
            query a PostgreSQL database using the tools provided.

            Current schema: {self.schema}

            Workflow:
            1. Start by calling list_tables to see what's available.
            2. Use describe_table and preview_table to understand the data.
            3. Write SQL queries using run_sql to answer the question.
            4. If a query errors, fix it and retry.
            5. When a visual would help, use generate_chart.
            6. Always finish by calling final_answer with your complete response.

            Rules:
            - Always prefix table names with the schema: {self.schema}.table_name
            - DDL is blocked; DML (INSERT/UPDATE/DELETE) is allowed.
            - SELECT queries auto-get LIMIT {SELECT_ROW_LIMIT} if none specified.
            - Keep answers concise and data-driven.
        """)

        messages: list[dict] = [{"role": "user", "content": question}]

        for turn in range(MAX_AGENT_TURNS):
            log.info("Agent turn %d", turn + 1)

            api_kwargs: dict[str, Any] = {
                "model": CLAUDE_MODEL,
                "max_tokens": 16000,
                "system": system_prompt,
                "tools": self.INTERNAL_TOOLS,
                "messages": messages,
            }

            # Extended thinking on first turn and every 5th turn
            if turn == 0 or turn % 5 == 0:
                api_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 10000,
                }

            response = self.client.messages.create(**api_kwargs)
            log.info("Stop reason: %s", response.stop_reason)

            # Build the assistant message preserving thinking blocks
            assistant_content = []
            for block in response.content:
                if block.type == "thinking":
                    assistant_content.append({
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.signature,
                    })
                elif block.type == "text":
                    assistant_content.append({
                        "type": "text",
                        "text": block.text,
                    })
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            messages.append({"role": "assistant", "content": assistant_content})

            # If model stopped without tool use, force a final_answer nudge
            if response.stop_reason == "end_turn":
                # Extract text if any
                text_parts = [
                    b.text for b in response.content if b.type == "text"
                ]
                return {
                    "answer": "\n".join(text_parts) or "No answer produced.",
                    "sql": "",
                    "summary": "",
                    "chart_json": "",
                }

            # Process tool calls
            if response.stop_reason == "tool_use":
                tool_results = []
                final_data = None

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    log.info("Calling tool: %s", block.name)
                    result = self._exec_tool(block.name, block.input)

                    if result == "__FINAL__":
                        final_data = block.input
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Answer delivered.",
                        })
                    else:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})

                if final_data is not None:
                    return {
                        "answer": final_data.get("answer", ""),
                        "sql": final_data.get("sql", ""),
                        "summary": final_data.get("summary", ""),
                        "chart_json": final_data.get("chart_json", ""),
                    }

        return {"answer": "Agent reached maximum turns without a final answer.", "sql": "", "summary": "", "chart_json": ""}


# ── MCP tools ────────────────────────────────────────────────────────────────


@mcp.tool()
def test_connection(
    host: str, port: int, database: str, user: str, password: str
) -> str:
    """Test PostgreSQL connectivity with the given credentials."""
    try:
        creds = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        with _connect(creds) as conn:
            conn.execute("SELECT 1")
        return json.dumps({"success": True, "message": "Connection successful"})
    except Exception as exc:
        log.exception("Connection test failed")
        return json.dumps({"success": False, "message": str(exc)})


@mcp.tool()
def list_schemas(
    host: str, port: int, database: str, user: str, password: str
) -> str:
    """List all non-system schemas in the database."""
    try:
        creds = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        with _connect(creds) as conn:
            rows = conn.execute(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN ('information_schema', 'pg_catalog', "
                "'pg_toast') ORDER BY schema_name"
            ).fetchall()
        schemas = [r[0] for r in rows]
        return json.dumps({"success": True, "schemas": schemas})
    except Exception as exc:
        log.exception("list_schemas failed")
        return json.dumps({"success": False, "message": str(exc)})


@mcp.tool()
def ask_agent(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    schema: str,
    question: str,
) -> str:
    """Run the AI agent to answer a natural-language question about the database."""
    creds = {
        "host": host,
        "port": port,
        "database": database,
        "user": user,
        "password": password,
    }
    try:
        agent = PostgresAgent(creds, schema)
        result = agent.run(question)
        return json.dumps({"success": True, **result}, default=_json_default)
    except Exception as exc:
        log.exception("ask_agent failed")
        return json.dumps({"success": False, "answer": str(exc), "sql": "", "summary": "", "chart_json": ""})


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting MCP server (stdio) …")
    mcp.run(transport="stdio")
