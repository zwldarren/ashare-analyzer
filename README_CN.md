# A股分析器

基于大语言模型（LLM）的智能A股分析系统，采用多智能体架构。自动分析您的自选股列表，并将"决策仪表盘"推送到 Discord、Telegram 或 Email。

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**[English](README.md)**

## 功能特点

- **多智能体架构**：专业化的 AI 智能体协同工作，提供全面分析
  - 技术分析智能体 - 图表形态、技术指标、趋势分析
  - 基本面分析智能体 - 财务报表、估值指标
  - 新闻情绪智能体 - 新闻聚合与情绪评分
  - 筹码分布智能体 - 机构/散户资金流向分析
  - 风险管理智能体 - 仓位管理、风险评估
  - 投资组合经理智能体 - 最终决策综合

- **多数据源**：聚合 Akshare、Baostock、Tushare、eFinance 等数据源

- **灵活的通知渠道**：支持 Discord、Telegram、Email 推送

- **多 AI 提供商**：通过 LiteLLM 支持 100+ LLM 提供商（DeepSeek、OpenAI、Gemini、Claude 等）

- **定时执行**：自动每日分析

- **灵活配置**：支持 TOML 配置文件，环境变量可覆盖

## 快速开始

### 环境要求

- Python 3.13+

### 安装

```bash
# 使用 uv 安装（推荐）
uv tool install ashare-analyzer

# 或使用 uvx 直接运行（无需安装）
uvx ashare-analyzer

# 或使用 pip 安装
pip install ashare-analyzer
```

### 配置

支持 TOML 配置文件和环境变量两种配置方式。

**配置优先级**（从低到高）：
1. `~/.ashare-analyzer/config.toml` - 基础配置
2. 环境变量 / `.env` - 覆盖 TOML
3. 命令行参数 - 最高优先级

#### 方式一：TOML 配置文件（推荐）

```bash
# 创建配置目录
mkdir -p ~/.ashare-analyzer

# 复制示例配置
cp config.example.toml ~/.ashare-analyzer/config.toml

# 编辑配置
vim ~/.ashare-analyzer/config.toml
```

最小配置示例 `config.toml`：

```toml
stock_list = ["600519", "300750"]

[ai]
llm_model = "deepseek/deepseek-reasoner"
llm_api_key = "your_api_key_here"
```

#### 方式二：环境变量

```bash
# 自选股列表（逗号分隔）
STOCK_LIST=600519,300750,002594

# AI 模型（LiteLLM 格式：provider/model-name）
LLM_MODEL=deepseek/deepseek-reasoner
LLM_API_KEY=your_api_key_here
```

📖 **[完整配置指南](docs/configuration.md)** - 所有配置项详解与示例

### 使用方法

```bash
# 运行分析
ashare-analyzer

# 调试模式（详细日志）
ashare-analyzer --debug

# 分析指定股票
ashare-analyzer --stocks 600519,300750

# 通过命令行覆盖 AI 模型配置
ashare-analyzer --model openai/gpt-5 --api-key your_key --base-url https://api.openai.com/v1

# 定时模式（每日在配置时间自动执行）
ashare-analyzer --schedule

# 试运行模式（仅获取数据，不进行 AI 分析）
ashare-analyzer --dry-run

# 跳过通知推送
ashare-analyzer --no-notify

# 显示帮助
ashare-analyzer --help
```

## 配置详情

> 📖 **[完整配置指南](docs/configuration.md)** - 所有配置项详解与示例

### AI 模型配置

通过 LiteLLM 格式支持 100+ 提供商：

| 提供商 | 模型示例 | API Key 获取 |
|--------|----------|--------------|
| DeepSeek | `deepseek/deepseek-reasoner` | [platform.deepseek.com](https://platform.deepseek.com/) |
| OpenAI | `openai/gpt-5.2` | [platform.openai.com](https://platform.openai.com/) |
| Gemini | `gemini/gemini-3.1-pro-preview` | [aistudio.google.com](https://aistudio.google.com/) |
| Claude | `anthropic/claude-sonnet-4-6` | [console.anthropic.com](https://console.anthropic.com/) |

完整提供商列表：[LiteLLM Providers](https://docs.litellm.ai/docs/providers)


### 通知渠道

在 `config.toml` 中配置一个或多个通知渠道：

```toml
# Discord
[notification.discord]
webhook_url = "https://discord.com/api/webhooks/..."

# Telegram
[notification.telegram]
bot_token = "your_bot_token"
chat_id = "your_chat_id"

# 邮件
[notification.email]
sender = "your_email@gmail.com"
password = "your_app_password"
receivers = ["recipient@example.com"]
```

### 搜索引擎（用于获取新闻）

```toml
[search]
# Tavily（推荐）
tavily_api_keys = ["your_tavily_key"]

# 博查（备选）
bocha_api_keys = ["key1", "key2"]

# SerpAPI（备选）
serpapi_api_keys = ["your_serpapi_key"]
```

### 新闻过滤器配置

新闻过滤器使用 AI 过滤掉低相关性和过时的新闻结果。

```toml
[news_filter]
enabled = true
min_results = 3
model = ""  # 可选，不配置则使用 LLM_MODEL
```

### 定时任务配置

```toml
[schedule]
enabled = true
time = "18:00"  # 24小时制
```

## 开发

```bash
# 格式化代码
ruff format

# 代码检查
ruff check --fix

# 类型检查
ty check .

# 运行测试
uv run pytest
```

## Docker 部署

```bash
# 构建镜像
docker build -t ashare-analyzer -f docker/Dockerfile .

# 运行容器
docker run -it --env-file .env ashare-analyzer

# 使用 docker-compose
cd docker && docker-compose up -d
```

## 命令行选项

| 选项 | 说明 |
|------|------|
| `--debug` | 启用调试模式，输出详细日志 |
| `--dry-run` | 仅获取数据，不进行 AI 分析 |
| `--stocks <codes>` | 指定股票代码（逗号分隔，覆盖配置文件） |
| `--model <name>` | 指定 AI 模型名称（覆盖配置文件） |
| `--api-key <key>` | 指定 API 密钥（覆盖配置文件） |
| `--base-url <url>` | 指定 API 基础 URL（覆盖配置文件） |
| `--no-notify` | 不发送推送通知 |
| `--single-notify` | 单股推送模式：每分析完一只股票立即推送 |
| `--workers <n>` | 并发线程数 |
| `--schedule` | 启用定时任务模式 |

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- [ZhuLinsen/daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis) - 原始项目来源
- [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) - 多智能体架构灵感
- 所有开源库的贡献者

---

⭐ 如果这个项目对您有帮助，请给个 Star！