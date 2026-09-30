import os 
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from tools.search_tool import duckduckgo_search
from tools.flight_tool import search_flights


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your local PostgreSQL connection URL to .env"
        )

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")


# =========================
# LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    bus_results: str
    train_results: str
    ship_results: str
    cab_results: str
    itinerary: str
    llm_calls: int


# =========================
# Flight Agent
# =========================

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = duckduckgo_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Bus Agent
# =========================

def bus_agent(state: TravelState):
    query = f"Best bus routes and options for {state['user_query']}"
    bus_results = duckduckgo_search(query)

    return {
        "bus_results": bus_results,
        "messages": [
            AIMessage(content="Bus route information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Train Agent
# =========================

def train_agent(state: TravelState):
    query = f"Best train routes and options for {state['user_query']}"
    train_results = duckduckgo_search(query)

    return {
        "train_results": train_results,
        "messages": [
            AIMessage(content="Train route information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Ship Agent
# =========================

def ship_agent(state: TravelState):
    query = f"Best ferry or cruise options for {state['user_query']}"
    ship_results = duckduckgo_search(query)

    return {
        "ship_results": ship_results,
        "messages": [
            AIMessage(content="Ferry/cruise information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Cab/Taxi Agent
# =========================

def cab_agent(state: TravelState):
    query = f"Local cab and taxi options with estimated fares for {state['user_query']}"
    cab_results = duckduckgo_search(query)

    return {
        "cab_results": cab_results,
        "messages": [
            AIMessage(content="Cab/taxi information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Bus Results:
{state['bus_results']}

Train Results:
{state['train_results']}

Ship Results:
{state['ship_results']}

Cab/Taxi Results:
{state['cab_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights (live/status data):
{state['flight_results']}

Hotels:
{state['hotel_results']}

Buses:
{state['bus_results']}

Trains:
{state['train_results']}

Ships/Ferries:
{state['ship_results']}

Cabs/Taxis:
{state['cab_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Bus Options
5. Train Options
6. Ship/Ferry Options
7. Cab/Taxi Options
8. Day-by-Day Itinerary
9. Estimated Budget
10. Final Recommendations

Important:
- Be clear and practical.
- The flight data provided above is LIVE/STATUS data only (real-time flight tracking of planes currently in the air), NOT booking or pricing data. It does not necessarily match the user's requested route or travel dates.
- NEVER say a flight is "already booked", "no extra money is required", or that a ticket has been paid for. The user has NOT booked anything. This is strictly forbidden.
- If the flight data does not clearly match the user's requested route, say so honestly, e.g.: "Live flight tracking data for this exact route was not available. Here is a rough estimate instead, and I recommend checking a booking site directly."
- Always provide a rough ESTIMATED price range for flights based on general knowledge of typical fares for that route and season, clearly labeled as an estimate (e.g. "Estimated fare: ₹X,000–₹Y,000, based on typical pricing — check Skyscanner, Google Flights, or the airline's site for exact current prices").
- Note that bus, train, ship, and cab information comes from web search and should be verified before booking.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("bus_agent", bus_agent)
graph.add_node("train_agent", train_agent)
graph.add_node("ship_agent", ship_agent)
graph.add_node("cab_agent", cab_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "bus_agent")
graph.add_edge("bus_agent", "train_agent")
graph.add_edge("train_agent", "ship_agent")
graph.add_edge("ship_agent", "cab_agent")
graph.add_edge("cab_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "bus_results": "",
            "train_results": "",
            "ship_results": "",
            "cab_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "bus_results": result.get("bus_results", ""),
        "train_results": result.get("train_results", ""),
        "ship_results": result.get("ship_results", ""),
        "cab_results": result.get("cab_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }