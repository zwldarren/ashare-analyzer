# Ashare Analyzer

An intelligent A-share stock analysis system powered by LLMs with multi-agent architecture. Automatically analyzes your watchlist and delivers a "Decision Dashboard" to Discord, Telegram, or Email.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**[中文文档](README_CN.md)**

## Features

- **Multi-Agent Architecture**: Specialized AI agents collaborate to provide comprehensive analysis
  - Technical Analysis Agent - Chart patterns, indicators, trend analysis
  - Fundamental Analysis Agent - Financial statements, valuation metrics
  - News Sentiment Agent - News aggregation and sentiment scoring
  - Chip Distribution Agent - Institutional/retail fund flow analysis
  - Risk Management Agent - Position sizing, risk assessment
  - Portfolio Manager Agent - Final decision synthesis

- **Multi-Source Data**: Aggregates data from Akshare, Baostock, Tushare, eFinance, etc.

- **Flexible Notifications**: Push results to Discord, Telegram, or Email

- **Multiple AI Providers**: Supports 100+ LLM providers via LiteLLM (DeepSeek, OpenAI, Gemini, Claude, etc.)

- **Scheduled Execution**: Run daily analysis automatically

- **Flexible Configuration**: TOML config file with environment variable overrides

## Quick Start

### Prerequisites

- Python 3.13+

### Installation

```bash
# Install with uv (recommended)
uv tool install ashare-analyzer

# Or run directly with uvx (no installation needed)
uvx ashare-analyzer

# Or install with pip
pip install ashare-analyzer
```

### Configuration

Configuration can be done via TOML file or environment variables.

**Configuration Priority** (lowest to highest):
1. `~/.ashare-analyzer/config.toml` - Base configuration
2. Environment variables / `.env` - Override TOML
3. CLI arguments - Highest priority

#### Option 1: TOML Config File (Recommended)

```bash
# Create config directory
mkdir -p ~/.ashare-analyzer

# Copy example config
cp config.example.toml ~/.ashare-analyzer/config.toml

# Edit config
vim ~/.ashare-analyzer/config.toml
```

Minimal `config.toml`:

```toml
stock_list = ["600519", "300750"]

[ai]
llm_model = "deepseek/deepseek-reasoner"
llm_api_key = "your_api_key_here"
```

#### Option 2: Environment Variables

```bash
# Stock watchlist (comma-separated)
STOCK_LIST=600519,300750,002594

# AI Model (LiteLLM format: provider/model-name)
LLM_MODEL=deepseek/deepseek-reasoner
LLM_API_KEY=your_api_key_here
```

📖 **[Full Configuration Guide](docs/configuration.md)** - All config options with examples

### Usage

```bash
# Run analysis
ashare-analyzer

# Debug mode (verbose logging)
ashare-analyzer --debug

# Analyze specific stocks
ashare-analyzer --stocks 600519,300750

# Override AI model settings via CLI
ashare-analyzer --model openai/gpt-5 --api-key your_key --base-url https://api.openai.com/v1

# Scheduled mode (runs daily at configured time)
ashare-analyzer --schedule

# Dry run (fetch data only, no AI analysis)
ashare-analyzer --dry-run

# Skip notifications
ashare-analyzer --no-notify

# Show help
ashare-analyzer --help
```

## Configuration Details

> 📖 **[Full Configuration Guide](docs/configuration.md)** - Complete reference with all options

### AI Model Configuration

Supports 100+ providers via LiteLLM format:

| Provider | Model Example | API Key Source |
|----------|---------------|----------------|
| DeepSeek | `deepseek/deepseek-reasoner` | [platform.deepseek.com](https://platform.deepseek.com/) |
| OpenAI | `openai/gpt-5.2` | [platform.openai.com](https://platform.openai.com/) |
| Gemini | `gemini/gemini-3.1-pro-preview` | [aistudio.google.com](https://aistudio.google.com/) |
| Claude | `anthropic/claude-sonnet-4-6` | [console.anthropic.com](https://console.anthropic.com/) |

Full provider list: [LiteLLM Providers](https://docs.litellm.ai/docs/providers)


### Notification Channels

Configure one or more notification channels in `config.toml`:

```toml
# Discord
[notification.discord]
webhook_url = "https://discord.com/api/webhooks/..."

# Telegram
[notification.telegram]
bot_token = "your_bot_token"
chat_id = "your_chat_id"

# Email
[notification.email]
sender = "your_email@gmail.com"
password = "your_app_password"
receivers = ["recipient@example.com"]
```

### Search Engines (for news)

```toml
[search]
# Tavily (recommended)
tavily_api_keys = ["your_tavily_key"]

# Bocha (alternative)
bocha_api_keys = ["key1", "key2"]

# SerpAPI (alternative)
serpapi_api_keys = ["your_serpapi_key"]
```

### News Filter Configuration

The news filter uses AI to filter out low-relevance and stale news results.

```toml
[news_filter]
enabled = true
min_results = 3
model = ""  # Optional, falls back to LLM_MODEL
```

## Development

```bash
# Format code
ruff format

# Lint code
ruff check --fix

# Type check
ty check .

# Run tests
uv run pytest
```

## Docker

```bash
# Build image
docker build -t ashare-analyzer -f docker/Dockerfile .

# Run container
docker run -it --env-file .env ashare-analyzer

# Using docker-compose
cd docker && docker-compose up -d
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ZhuLinsen/daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis) - Original project this was forked from
- [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) - Multi-agent architecture inspiration
- All the open-source libraries that made this possible

---

⭐ Star this repo if you find it helpful!
