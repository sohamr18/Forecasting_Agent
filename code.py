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
import os
import json
import yaml
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "TestProjectApp"

from langchain.chat_models import init_chat_model


def create_agent():

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Define models
    model_id = config["model_id"]
    print(model_id)
    llm = init_chat_model(model_id)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a forecasting assistant.
        User will provide a prompt like: 'forecast 12 months using additive trend'.
        Extract:
        - horizon (integer)
        - method (additive, multiplicative, simple)
        Answer in JSON as {{"horizon": int, "method": str}}
        Prompt: {user_prompt}
        """
    )

    def agent_call(state):
        inp = state["user_prompt"]
        response = llm.invoke(prompt.format_messages(user_prompt=inp))
        jsn = response.content
        if "{" in jsn:
            op = jsn.split("{")[1]
            op = "{" + op
            if "}" in op:
                op = op.split("}")[0]
                op = op + "}"
        else:
            op = "{'horizon': 6, 'method': 'additive'}"
        param = json.loads(op)
        return param

    graph = StateGraph(dict)
    graph.add_node("agent", agent_call)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    print("Agent created")
    return graph.compile()


def forecast_series(df, horizon, method):
    model = ExponentialSmoothing(
        df,
        trend=(
            "add"
            if method == "additive"
            else ("mul" if method == "multiplicative" else None)
        ),
        seasonal="additive",  # or "multiplicative"
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit()
    forecast = model.forecast(horizon)
    forecast = forecast.round()
    return forecast


def main():

    st.title("📈 Time Series Forecasting Agent")

    uploaded_file = st.file_uploader(
        "Upload a CSV with a time series column", type="csv"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # st.write("Preview of data:", df.head())

        col = st.selectbox("Select the time series column", df.columns)
        user_prompt = st.text_input(
            "Enter your forecasting request (e.g., 'forecast 12 months using additive trend')"
        )
        print(user_prompt)
        if st.button("Run Forecast"):
            agent = create_agent()
            result = agent.invoke({"user_prompt": user_prompt})
            # Extract horizon & method

            horizon = result.get("horizon", 6)
            method = result.get("method", "additive")

            series = df[col].dropna()
            forecast = forecast_series(series, horizon, method)

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
