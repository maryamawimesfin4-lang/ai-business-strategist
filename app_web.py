import os
import warnings
from typing import TypedDict
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="AI Business Strategist", page_icon="🚀", layout="centered"
)

st.title("🚀 AI Business Strategist")
st.subheader("Generate Marketing, Sales, and Production Blueprints with LangGraph")


# 1. Define State
class MiniMarketState(TypedDict):
  market_concept: str
  marketing_plan: str
  sales_strategy: str
  supply_operations: str


# 2. Setup Gemini Model safely via Secrets or Environment Variables
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
  try:
    api_key = st.secrets.get("GOOGLE_API_KEY")
  except Exception:
    api_key = None

if not api_key:
  st.error(
      "API Key missing! Please set GOOGLE_API_KEY in your Streamlit Cloud"
      " Secrets."
  )
  st.stop()

# Using standard Gemini 2.5 Flash endpoint
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)


def clean_text(response) -> str:
  if isinstance(response.content, list):
    return response.content[0].get("text", "")
  return str(response.content)


# 3. Define Graph Nodes
def marketing_node(state: MiniMarketState) -> dict:
  prompt = (
      "Create a 3-bullet marketing strategy for this business concept:"
      f" {state['market_concept']}"
  )
  res = llm.invoke(prompt)
  return {"marketing_plan": clean_text(res)}


def sales_node(state: MiniMarketState) -> dict:
  prompt = (
      "Based on this marketing strategy:\n"
      f"{state['marketing_plan']}\n\nList 3 key sales tactics and channels."
  )
  res = llm.invoke(prompt)
  return {"sales_strategy": clean_text(res)}


def operations_node(state: MiniMarketState) -> dict:
  prompt = (
      f"For the business concept '{state['market_concept']}', outline a 3-step"
      " production and supply operations plan."
  )
  res = llm.invoke(prompt)
  return {"supply_operations": clean_text(res)}


# 4. Build Graph Workflow
@st.cache_resource
def build_workflow():
  workflow = StateGraph(MiniMarketState)
  workflow.add_node("marketer", marketing_node)
  workflow.add_node("sales_agent", sales_node)
  workflow.add_node("operations_agent", operations_node)

  workflow.add_edge(START, "marketer")
  workflow.add_edge("marketer", "sales_agent")
  workflow.add_edge("sales_agent", "operations_agent")
  workflow.add_edge("operations_agent", END)
  return workflow.compile()


app = build_workflow()

# 5. User Input Form
market_concept = st.text_input(
    "Enter your Business Idea or Product Concept:",
    placeholder="e.g., Organic Grocery Mini Market",
)

if st.button("Generate Complete Blueprint", type="primary"):
  if market_concept.strip():
    with st.spinner("Processing through AI Graph Nodes..."):
      result = app.invoke({"market_concept": market_concept})

    st.success("Blueprint Generated Successfully!")

    # Display Results in Visual Tabs
    tab1, tab2, tab3 = st.tabs(["📢 Marketing", "💼 Sales", "⚙️ Operations"])

    with tab1:
      st.markdown("### Marketing Strategy")
      st.write(result["marketing_plan"])

    with tab2:
      st.markdown("### Sales Tactics & Funnel")
      st.write(result["sales_strategy"])

    with tab3:
      st.markdown("### Production & Supply Operations")
      st.write(result["supply_operations"])

    # File Download
    report_text = f"""BUSINESS BLUEPRINT: {market_concept}

--- MARKETING ---
{result['marketing_plan']}

--- SALES ---
{result['sales_strategy']}

--- OPERATIONS ---
{result['supply_operations']}
"""
    st.download_button(
        label="📥 Download Blueprint (.txt)",
        data=report_text,
        file_name="business_blueprint.txt",
        mime="text/plain",
    )
  else:
    st.warning("Please enter a concept first!")