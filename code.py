import streamlit as st import pandas as pd import matplotlib.pyplot as plt from statsmodels.tsa.holtwinters import ExponentialSmoothing

from langgraph.graph import StateGraph from langchain_groq import ChatGroq from langchain_core.prompts import ChatPromptTemplate

-----------------------

Define LLM agent via LangGraph

-----------------------

def create_agent(): llm = ChatGroq(model="llama3-8b-8192")

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
    return {"parsed": response.content}

graph = StateGraph(dict)
graph.add_node("agent", agent_call)
graph.set_entry_point("agent")
graph.set_finish_point("agent")
return graph.compile()

-----------------------

Forecast function

-----------------------

def forecast_series(df, horizon, method): model = ExponentialSmoothing( df, trend="add" if method == "additive" else ("mul" if method == "multiplicative" else None), seasonal=None ).fit() forecast = model.forecast(horizon) return forecast

-----------------------

Streamlit UI

-----------------------

st.title("📈 Time Series Forecasting Agent")

uploaded_file = st.file_uploader("Upload a CSV with a time series column", type="csv")

if uploaded_file: df = pd.read_csv(uploaded_file) st.write("Preview of data:", df.head())

col = st.selectbox("Select the time series column", df.columns)
user_prompt = st.text_input("Enter your forecasting request (e.g., 'forecast 12 months using additive trend')")

if st.button("Run Forecast") and user_prompt:
    agent = create_agent()
    result = agent.invoke({"user_prompt": user_prompt})

    import json
    parsed = json.loads(result["parsed"])  # Extract horizon & method

    horizon = parsed.get("horizon", 5)
    method = parsed.get("method", "additive")

    series = df[col].dropna()
    forecast = forecast_series(series, horizon, method)

    # Plot
    fig, ax = plt.subplots()
    series.plot(ax=ax, label="History")
    forecast.plot(ax=ax, label="Forecast")
    ax.legend()
    st.pyplot(fig)

