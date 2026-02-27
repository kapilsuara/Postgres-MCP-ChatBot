# Postgres MCP ChatBot

A natural-language AI chat interface for PostgreSQL databases. Ask questions about your data in plain English and get answers with SQL queries and interactive charts — powered by Claude and the Model Context Protocol (MCP).

## How It Works

```
User (Streamlit Chat UI)
        │
        ▼
   MCP Client ──stdio──▶ MCP Server
                              │
                              ▼
                     PostgresAgent (Claude AI)
                        │           │
                        ▼           ▼
                   PostgreSQL    Plotly Charts
```

1. Enter your PostgreSQL credentials in the sidebar and connect.
2. Select a schema and type a question in plain English (e.g., *"What are the top 10 customers by revenue?"*).
3. The AI agent autonomously explores your schema, writes SQL, executes it, and returns an answer — optionally with an interactive chart.

## Features

- **Natural language to SQL** — Claude analyzes your schema and writes accurate queries
- **Interactive charts** — automatic Plotly visualizations (bar, line, scatter, pie, histogram, heatmap)
- **Safe by default** — DDL statements (`DROP`, `ALTER`, `TRUNCATE`, etc.) are blocked; SELECT queries are auto-limited to 500 rows
- **Agentic reasoning** — multi-turn tool use with extended thinking for complex questions
- **Schema-aware** — discovers tables, columns, constraints, and statistics before querying

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| AI | Claude (Anthropic API) |
| Protocol | Model Context Protocol (MCP) |
| Database | PostgreSQL (psycopg 3) |
| Charts | Plotly |

## Prerequisites

- Python 3.10+
- A PostgreSQL database
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Postgres-MCP-ChatBot.git
cd Postgres-MCP-ChatBot

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

## Usage

```bash
streamlit run app/streamlit_app.py
```

The app opens in your browser. Fill in your PostgreSQL connection details in the sidebar, click **Connect**, select a schema, and start chatting.

> The MCP server runs automatically as a subprocess — no separate server startup needed.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | *required* |
| `CLAUDE_MODEL` | Claude model to use | `claude-opus-4-6` |

## Project Structure

```
├── app/
│   └── streamlit_app.py    # Streamlit chat UI + MCP client
├── server/
│   └── mcp_server.py       # MCP server + AI agent with tool use
├── logs/
│   └── app.log             # Runtime logs
├── requirements.txt
├── .env                    # API key (not committed)
└── .gitignore
```

## Agent Tools

The AI agent has access to these internal tools when answering your questions:

| Tool | Purpose |
|------|---------|
| `list_tables` | Discover tables in the selected schema |
| `describe_table` | Get columns, types, and constraints |
| `preview_table` | Sample 3 rows from a table |
| `analyze_table_statistics` | Row counts, distinct values, null counts |
| `run_sql` | Execute read-only SQL queries |
| `generate_chart` | Create Plotly visualizations |
| `final_answer` | Return the structured response |

## License

[MIT](LICENSE)
