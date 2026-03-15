# 配置指南

本文档详细说明 ashare-analyzer 的配置方式，包括 TOML 配置文件和环境变量两种方式。

## 配置优先级

配置加载遵循以下优先级（从低到高）：

1. **config.toml** - 基础配置文件
2. **环境变量 / .env 文件** - 覆盖 TOML 配置
3. **命令行参数** - 最高优先级

这意味着你可以：
- 在 `config.toml` 中设置默认配置
- 通过环境变量临时覆盖特定配置
- 通过命令行参数进行单次运行的配置覆盖

## 配置文件位置

配置文件默认位于 `~/.ashare-analyzer/config.toml`。

可以通过设置 `BASE_DIR` 环境变量来更改配置目录：

```bash
export BASE_DIR=/path/to/custom/config/dir
```

## 快速开始

1. 复制示例配置文件：

```bash
mkdir -p ~/.ashare-analyzer
cp config.example.toml ~/.ashare-analyzer/config.toml
```

2. 编辑配置文件：

```bash
vim ~/.ashare-analyzer/config.toml
```

## 配置项详解

### 股票列表

```toml
# 自选股列表（必填）
stock_list = ["600519", "300750", "000858"]
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 说明 |
|--------|-----------|----------|------|------|
| 自选股列表 | `stock_list` | `STOCK_LIST` | `list[string]` | 股票代码数组 |

---

### AI/LLM 配置

```toml
[ai]
llm_model = "deepseek/deepseek-reasoner"
llm_api_key = "your-api-key-here"
llm_base_url = "https://api.deepseek.com"
llm_temperature = 0.7
llm_max_tokens = 8192
llm_request_delay = 2.0
llm_max_retries = 5
llm_retry_delay = 5.0
llm_timeout = 300
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 默认值 | 说明 |
|--------|-----------|----------|------|--------|------|
| 模型名称 | `ai.llm_model` | `LLM_MODEL` | string | - | LiteLLM 格式的模型名称 |
| API 密钥 | `ai.llm_api_key` | `LLM_API_KEY` | string | - | LLM API 密钥 |
| API 地址 | `ai.llm_base_url` | `LLM_BASE_URL` | string | - | API 基础 URL（可选） |
| 温度参数 | `ai.llm_temperature` | `LLM_TEMPERATURE` | float | 0.7 | 生成温度 (0.0-2.0) |
| 最大令牌数 | `ai.llm_max_tokens` | `LLM_MAX_TOKENS` | int | 8192 | 最大生成令牌数 |
| 请求延迟 | `ai.llm_request_delay` | `LLM_REQUEST_DELAY` | float | 2.0 | 请求间隔秒数 |
| 最大重试次数 | `ai.llm_max_retries` | `LLM_MAX_RETRIES` | int | 5 | 失败重试次数 |
| 重试延迟 | `ai.llm_retry_delay` | `LLM_RETRY_DELAY` | float | 5.0 | 重试间隔秒数 |
| 超时时间 | `ai.llm_timeout` | `LLM_TIMEOUT` | int | 300 | 请求超时秒数 |

#### 备用 LLM 配置

当主模型不可用时，自动切换到备用模型：

```toml
[ai.fallback]
model = "gemini/gemini-pro"
api_key = "your-fallback-api-key"
base_url = "https://generativelanguage.googleapis.com"
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 说明 |
|--------|-----------|----------|------|------|
| 备用模型 | `ai.fallback.model` | `LLM_FALLBACK_MODEL` | string | 备用模型名称 |
| 备用密钥 | `ai.fallback.api_key` | `LLM_FALLBACK_API_KEY` | string | 备用 API 密钥 |
| 备用地址 | `ai.fallback.base_url` | `LLM_FALLBACK_BASE_URL` | string | 备用 API 地址 |

---

### 搜索引擎配置

用于获取股票相关新闻：

```toml
[search]
bocha_api_keys = ["key1", "key2"]
tavily_api_keys = ["key3"]
brave_api_keys = []
serpapi_api_keys = []
searxng_base_url = ""
searxng_username = ""
searxng_password = ""
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 说明 |
|--------|-----------|----------|------|------|
| 博查 API Keys | `search.bocha_api_keys` | `BOCHA_API_KEYS` | list[string] | 博查搜索 API 密钥列表 |
| Tavily API Keys | `search.tavily_api_keys` | `TAVILY_API_KEYS` | list[string] | Tavily 搜索 API 密钥列表 |
| Brave API Keys | `search.brave_api_keys` | `BRAVE_API_KEYS` | list[string] | Brave 搜索 API 密钥列表 |
| SerpAPI Keys | `search.serpapi_api_keys` | `SERPAPI_API_KEYS` | list[string] | SerpAPI 密钥列表 |
| SearXNG 地址 | `search.searxng_base_url` | `SEARXNG_BASE_URL` | string | 自建 SearXNG 服务地址 |
| SearXNG 用户名 | `search.searxng_username` | `SEARXNG_USERNAME` | string | SearXNG basic auth用户名 |
| SearXNG 密码 | `search.searxng_password` | `SEARXNG_PASSWORD` | string | SearXNG basic auth密码 |

> **注意**：支持配置多个 API Key，系统会自动轮换使用。列表类型在环境变量中使用逗号分隔，如 `BOCHA_API_KEYS=key1,key2,key3`

---

### 通知配置

#### 邮件通知

```toml
[notification.email]
sender = "your-email@gmail.com"
sender_name = "ashare_analyzer股票分析助手"
password = "your-app-password"
receivers = ["receiver1@example.com", "receiver2@example.com"]
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 说明 |
|--------|-----------|----------|------|------|
| 发件人邮箱 | `notification.email.sender` | `EMAIL_SENDER` | string | 发件人邮箱地址 |
| 发件人名称 | `notification.email.sender_name` | `EMAIL_SENDER_NAME` | string | 发件人显示名称 |
| 邮箱密码 | `notification.email.password` | `EMAIL_PASSWORD` | string | 邮箱应用密码 |
| 收件人列表 | `notification.email.receivers` | `EMAIL_RECEIVERS` | list[string] | 收件人邮箱列表 |

> **Gmail 用户**：需要使用[应用专用密码](https://support.google.com/accounts/answer/185833)而非账户密码。

#### Telegram 通知

```toml
[notification.telegram]
bot_token = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
chat_id = "-1001234567890"
message_thread_id = "123"  # 可选，用于话题群组
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 说明 |
|--------|-----------|----------|------|------|
| Bot Token | `notification.telegram.bot_token` | `TELEGRAM_BOT_TOKEN` | string | Telegram Bot Token |
| Chat ID | `notification.telegram.chat_id` | `TELEGRAM_CHAT_ID` | string | 群组/频道 ID |
| 话题 ID | `notification.telegram.message_thread_id` | `TELEGRAM_MESSAGE_THREAD_ID` | string | 话题 ID（可选） |

#### Discord 通知

```toml
[notification.discord]
webhook_url = "https://discord.com/api/webhooks/123456789/abcdefg"
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 说明 |
|--------|-----------|----------|------|------|
| Webhook URL | `notification.discord.webhook_url` | `DISCORD_WEBHOOK_URL` | string | Discord Webhook 地址 |

#### 自定义 Webhook

```toml
[notification.webhook]
urls = ["https://your-server.com/webhook1", "https://your-server.com/webhook2"]
bearer_token = "your-bearer-token"  # 可选
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 说明 |
|--------|-----------|----------|------|------|
| Webhook URLs | `notification.webhook.urls` | `CUSTOM_WEBHOOK_URLS` | list[string] | 自定义 Webhook 地址列表 |
| Bearer Token | `notification.webhook.bearer_token` | `CUSTOM_WEBHOOK_BEARER_TOKEN` | string | 认证令牌（可选） |

---

### 系统配置

```toml
[system]
log_level = "INFO"
max_workers = 3
debug = false
http_proxy = ""
https_proxy = ""
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 默认值 | 说明 |
|--------|-----------|----------|------|--------|------|
| 日志级别 | `system.log_level` | `LOG_LEVEL` | string | "INFO" | 日志级别：DEBUG/INFO/WARNING/ERROR |
| 最大并发数 | `system.max_workers` | `MAX_WORKERS` | int | 3 | 并发分析线程数 |
| 调试模式 | `system.debug` | `DEBUG` | bool | false | 是否启用调试模式 |
| HTTP 代理 | `system.http_proxy` | `HTTP_PROXY` | string | - | HTTP 代理地址 |
| HTTPS 代理 | `system.https_proxy` | `HTTPS_PROXY` | string | - | HTTPS 代理地址 |

---

### 定时任务配置

```toml
[schedule]
enabled = false
time = "18:00"
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 默认值 | 说明 |
|--------|-----------|----------|------|--------|------|
| 启用定时 | `schedule.enabled` | `SCHEDULE_ENABLED` | bool | false | 是否启用定时任务 |
| 执行时间 | `schedule.time` | `SCHEDULE_TIME` | string | "18:00" | 每日执行时间（24小时制） |

---

### 数据源配置

```toml
[data_source]
tushare_token = ""
efinance_priority = 0
akshare_priority = 1
tushare_priority = 2
baostock_priority = 3
yfinance_priority = 4
realtime_source_priority = "tencent,akshare_sina,efinance,akshare_em"
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 默认值 | 说明 |
|--------|-----------|----------|------|--------|------|
| Tushare Token | `data_source.tushare_token` | `TUSHARE_TOKEN` | string | - | Tushare API Token |
| eFinance 优先级 | `data_source.efinance_priority` | `EFINANCE_PRIORITY` | int | 0 | 数值越小优先级越高 |
| AKShare 优先级 | `data_source.akshare_priority` | `AKSHARE_PRIORITY` | int | 1 | 数值越小优先级越高 |
| Tushare 优先级 | `data_source.tushare_priority` | `TUSHARE_PRIORITY` | int | 2 | 数值越小优先级越高 |
| Baostock 优先级 | `data_source.baostock_priority` | `BAOSTOCK_PRIORITY` | int | 3 | 数值越小优先级越高 |
| YFinance 优先级 | `data_source.yfinance_priority` | `YFINANCE_PRIORITY` | int | 4 | 数值越小优先级越高 |
| 实时数据源 | `data_source.realtime_source_priority` | `REALTIME_SOURCE_PRIORITY` | string | "tencent,akshare_sina,..." | 实时行情数据源优先级 |

---

### 新闻过滤器配置

```toml
[news_filter]
enabled = true
min_results = 3
model = ""  # 可选，留空则使用主 LLM 模型
```

| 配置项 | TOML 路径 | 环境变量 | 类型 | 默认值 | 说明 |
|--------|-----------|----------|------|--------|------|
| 启用过滤器 | `news_filter.enabled` | `NEWS_FILTER_ENABLED` | bool | true | 是否启用新闻过滤 |
| 最小结果数 | `news_filter.min_results` | `NEWS_FILTER_MIN_RESULTS` | int | 3 | 过滤后保留的最小新闻数 |
| 过滤模型 | `news_filter.model` | `NEWS_FILTER_MODEL` | string | - | 专用过滤模型（可选） |

---

## 配置示例

### 最小配置

仅需配置 AI 模型和股票列表即可运行：

```toml
stock_list = ["600519", "300750"]

[ai]
llm_model = "deepseek/deepseek-reasoner"
llm_api_key = "your-api-key"
```

### 完整配置

```toml
# 自选股列表
stock_list = ["600519", "300750", "000858", "002594"]

# AI 配置
[ai]
llm_model = "deepseek/deepseek-reasoner"
llm_api_key = "your-deepseek-api-key"
llm_base_url = "https://api.deepseek.com"
llm_temperature = 0.7
llm_max_tokens = 8192
llm_request_delay = 2.0
llm_max_retries = 5
llm_retry_delay = 5.0
llm_timeout = 300

# 备用模型
[ai.fallback]
model = "gemini/gemini-pro"
api_key = "your-gemini-api-key"
base_url = "https://generativelanguage.googleapis.com"

# 搜索引擎
[search]
tavily_api_keys = ["your-tavily-key"]
bocha_api_keys = ["key1", "key2"]

# 邮件通知
[notification.email]
sender = "your-email@gmail.com"
sender_name = "股票分析助手"
password = "your-app-password"
receivers = ["receiver@example.com"]

# Telegram 通知
[notification.telegram]
bot_token = "your-bot-token"
chat_id = "your-chat-id"

# 系统设置
[system]
log_level = "INFO"
max_workers = 3
debug = false

# 定时任务
[schedule]
enabled = true
time = "18:00"

# 数据源
[data_source]
tushare_token = "your-tushare-token"
efinance_priority = 0
akshare_priority = 1

# 新闻过滤
[news_filter]
enabled = true
min_results = 3
```

---

## 环境变量对照表

所有 TOML 配置项都可以通过环境变量设置。以下是完整对照表：

| TOML 路径 | 环境变量 |
|-----------|----------|
| `stock_list` | `STOCK_LIST` |
| `ai.llm_model` | `LLM_MODEL` |
| `ai.llm_api_key` | `LLM_API_KEY` |
| `ai.llm_base_url` | `LLM_BASE_URL` |
| `ai.llm_temperature` | `LLM_TEMPERATURE` |
| `ai.llm_max_tokens` | `LLM_MAX_TOKENS` |
| `ai.llm_request_delay` | `LLM_REQUEST_DELAY` |
| `ai.llm_max_retries` | `LLM_MAX_RETRIES` |
| `ai.llm_retry_delay` | `LLM_RETRY_DELAY` |
| `ai.llm_timeout` | `LLM_TIMEOUT` |
| `ai.fallback.model` | `LLM_FALLBACK_MODEL` |
| `ai.fallback.api_key` | `LLM_FALLBACK_API_KEY` |
| `ai.fallback.base_url` | `LLM_FALLBACK_BASE_URL` |
| `search.bocha_api_keys` | `BOCHA_API_KEYS` |
| `search.tavily_api_keys` | `TAVILY_API_KEYS` |
| `search.brave_api_keys` | `BRAVE_API_KEYS` |
| `search.serpapi_api_keys` | `SERPAPI_API_KEYS` |
| `search.searxng_base_url` | `SEARXNG_BASE_URL` |
| `search.searxng_username` | `SEARXNG_USERNAME` |
| `search.searxng_password` | `SEARXNG_PASSWORD` |
| `notification.email.sender` | `EMAIL_SENDER` |
| `notification.email.sender_name` | `EMAIL_SENDER_NAME` |
| `notification.email.password` | `EMAIL_PASSWORD` |
| `notification.email.receivers` | `EMAIL_RECEIVERS` |
| `notification.telegram.bot_token` | `TELEGRAM_BOT_TOKEN` |
| `notification.telegram.chat_id` | `TELEGRAM_CHAT_ID` |
| `notification.telegram.message_thread_id` | `TELEGRAM_MESSAGE_THREAD_ID` |
| `notification.discord.webhook_url` | `DISCORD_WEBHOOK_URL` |
| `notification.webhook.urls` | `CUSTOM_WEBHOOK_URLS` |
| `notification.webhook.bearer_token` | `CUSTOM_WEBHOOK_BEARER_TOKEN` |
| `system.log_level` | `LOG_LEVEL` |
| `system.max_workers` | `MAX_WORKERS` |
| `system.debug` | `DEBUG` |
| `system.http_proxy` | `HTTP_PROXY` |
| `system.https_proxy` | `HTTPS_PROXY` |
| `schedule.enabled` | `SCHEDULE_ENABLED` |
| `schedule.time` | `SCHEDULE_TIME` |
| `data_source.tushare_token` | `TUSHARE_TOKEN` |
| `data_source.efinance_priority` | `EFINANCE_PRIORITY` |
| `data_source.akshare_priority` | `AKSHARE_PRIORITY` |
| `data_source.tushare_priority` | `TUSHARE_PRIORITY` |
| `data_source.baostock_priority` | `BAOSTOCK_PRIORITY` |
| `data_source.yfinance_priority` | `YFINANCE_PRIORITY` |
| `data_source.realtime_source_priority` | `REALTIME_SOURCE_PRIORITY` |
| `news_filter.enabled` | `NEWS_FILTER_ENABLED` |
| `news_filter.min_results` | `NEWS_FILTER_MIN_RESULTS` |
| `news_filter.model` | `NEWS_FILTER_MODEL` |
