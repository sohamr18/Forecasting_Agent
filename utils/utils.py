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
        User will provide a prompt like: 'forecast 12 months using additive trend with monthly seasonality'.
        Extract:
        - horizon (integer)
        - method (additive, multiplicative, simple)
        - seasonal_period (monthly, weekly, daily)
        Answer in JSON as {{"horizon": int, "method": str, "seasonal_period: : str}}
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
            op = "{'horizon': 6, 'method': 'additive', 'seasonal_period' : 'monthly'}"
        param = json.loads(op)
        return param

    graph = StateGraph(dict)
    graph.add_node("agent", agent_call)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    print("Agent created")
    return graph.compile()


def forecast_series(df, horizon, method, seasonal_period):
    periods = (
        12 if "month" in seasonal_period else (52 if "week" in seasonal_period else 7)
    )
    model = ExponentialSmoothing(
        df,
        trend=(
            "add"
            if method == "additive"
            else ("mul" if method == "multiplicative" else None)
        ),
        seasonal="additive",  # or "multiplicative"
        seasonal_periods=periods,
        initialization_method="estimated",
    ).fit()
    forecast = model.forecast(horizon)
    return forecast
