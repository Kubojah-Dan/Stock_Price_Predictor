import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY", "dummy-key-for-now")
)

agent_prompt = PromptTemplate(
    input_variables=["ticker", "price", "probability", "signal", "volatility", "forecasts", "trend_context"],
    template="""You are an expert AI Stock and Forex Trading Advisor.
You have analyzed the latest market data and technical indicators for {ticker}.

Recent Price Trend (Last 10 Days):
{trend_context}

Current Price: ${price}
ML Model Probability for Uptrend: {probability}
ML Suggested Signal: {signal}
Recent Volatility: {volatility}
Expected Forecasts: {forecasts}

Based on this information and your inherent knowledge of the market, provide a concise but insightful analysis on whether the user should BUY, SELL, or HOLD.
Explain the rationale clearly, factoring in both the price trend (candlestick patterns) and the ML model's prediction.
If the signal is BUY, specify a brief note on position sizing or stop loss strategy.

Decision and Rationale:
"""
)

chain = agent_prompt | llm

def get_agent_decision(prediction_data: dict) -> str:
    """Run the LLM to get a trading decision and rationale."""
    try:
        response = chain.invoke(prediction_data)
        return response.content
    except Exception as e:
        return f"Agent Error: Could not generate analysis. Please ensure API keys are set. Details: {e}"
