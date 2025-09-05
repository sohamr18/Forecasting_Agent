import os
import json
import yaml
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from langgraph.graph import StateGraph
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from utils.utils import create_agent, forecast_series

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "TestProjectApp"


def main():

    st.title("📈 Time Series Forecasting Agent")

    uploaded_file = st.file_uploader(
        "Upload a CSV with a time series column", type="csv"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        col = st.selectbox("Select the time series column", df.columns)
        user_prompt = st.text_input(
            "Enter your forecasting request (e.g., \
                'forecast 12 months using additive trend with monthly seasonality')"
        )
        print(user_prompt)
        if st.button("Run Forecast"):
            agent = create_agent()
            result = agent.invoke({"user_prompt": user_prompt})
            # Extract horizon & method

            horizon = result.get("horizon", 6)
            method = result.get("method", "additive")
            seasonal_periods = result.get("seasonal_periods", "monthly")

            series = df[col].dropna()
            forecast = forecast_series(series, horizon, method, seasonal_periods)
            # Plot
            fig, ax = plt.subplots()
            series.plot(ax=ax, label="History")
            forecast.plot(ax=ax, label="Forecast")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("Please upload a CSV file to get started.")


if __name__ == "__main__":
    main()
