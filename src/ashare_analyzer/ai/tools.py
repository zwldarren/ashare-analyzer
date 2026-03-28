"""
AI Tools Module

Defines Function Call schemas for LLM structured output.
"""

from typing import Any

# Unified function schema for agent analysis signals
ANALYZE_SIGNAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "analyze_signal",
        "description": "Analyze stock data and generate trading signal with reasoning",
        "parameters": {
            "type": "object",
            "properties": {
                "signal": {
                    "type": "string",
                    "enum": ["buy", "sell", "hold"],
                    "description": "Trading signal type",
                },
                "confidence": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Confidence level (0-100)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Analysis reasoning (max 200 characters)",
                },
                "metadata": {
                    "type": "object",
                    "description": "Agent-specific analysis metadata",
                    "properties": {
                        # Technical Agent fields
                        "trend_assessment": {"type": "string"},
                        "trend_strength": {"type": "integer"},
                        "key_levels": {
                            "type": "object",
                            "properties": {
                                "support": {"type": "number"},
                                "resistance": {"type": "number"},
                                "ideal_entry": {"type": "number"},
                                "stop_loss": {"type": "number"},
                            },
                        },
                        "technical_factors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "risk_factors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "recommendation": {"type": "string"},
                        # Chip Agent fields
                        "control_assessment": {"type": "string"},
                        "phase": {"type": "string"},
                        "risk_level": {"type": "string"},
                        "key_factors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        # News Sentiment fields
                        "sentiment": {"type": "string"},
                        "sentiment_score": {"type": "integer"},
                        "bullish_articles": {"type": "integer"},
                        "bearish_articles": {"type": "integer"},
                        "neutral_articles": {"type": "integer"},
                        "positive_catalysts": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "key_headlines": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "irrelevant_results": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        # Portfolio Manager fields
                        "action": {"type": "string"},
                        "position_ratio": {"type": "number"},
                        "trade_quantity": {
                            "type": "integer",
                            "description": (
                                "Number of shares to trade. For BUY: shares to purchase. "
                                "For SELL: shares to sell (positive number). 0 if HOLD or no action needed."
                            ),
                        },
                        "position_action": {
                            "type": "string",
                            "enum": [
                                "open_position",
                                "add_position",
                                "reduce_position",
                                "close_position",
                                "keep_position",
                                "no_action",
                            ],
                            "description": (
                                "Type of position action: open_position (new), add_position (increase), "
                                "reduce_position (partial exit), close_position (full exit), "
                                "keep_position (maintain), no_action (do nothing)"
                            ),
                        },
                    },
                },
            },
            "required": ["signal", "confidence", "reasoning"],
        },
    },
}


# Decision tool for LLM-driven autonomous decisions
DECISION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "make_trading_decision",
        "description": "Make an autonomous trading decision based on analysis reports and market context",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["buy", "sell", "hold"],
                    "description": "Your final trading decision",
                },
                "confidence": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Your confidence level in this decision (0-100)",
                },
                "position_ratio": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Target position ratio as fraction of portfolio (0-1), must not exceed risk limit",
                },
                "trade_quantity": {
                    "type": "integer",
                    "description": "Number of shares to trade. Positive for buy, positive for sell (absolute value)",
                },
                "position_action": {
                    "type": "string",
                    "enum": [
                        "open_position",
                        "add_position",
                        "reduce_position",
                        "close_position",
                        "keep_position",
                        "no_action",
                    ],
                    "description": "Position action type",
                },
                "reasoning": {
                    "type": "string",
                    "maxLength": 500,
                    "description": (
                        "Your reasoning for this decision - explain YOUR thought process, not just summarize analysts"
                    ),
                },
                "key_considerations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key factors that most influenced your decision",
                },
                "risks_identified": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Risks or concerns you identified that affected your decision",
                },
            },
            "required": ["decision", "confidence", "position_ratio", "reasoning"],
        },
    },
}


__all__ = ["ANALYZE_SIGNAL_TOOL", "DECISION_TOOL"]
