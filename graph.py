import os
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv

load_dotenv()


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    is_travel_query: bool


def build_app(db_url: str, groq_api_key: str = "", tavily_api_key: str = "", aviationstack_api_key: str = ""):
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
    if aviationstack_api_key:
        os.environ["AVIATIONSTACK_API_KEY"] = aviationstack_api_key

    llm = ChatGroq(model="llama-3.3-70b-versatile")

    def router_node(state: TravelState):
        """Classify whether the query is travel-related or general conversation."""
        query = state["user_query"]
        response = llm.invoke([
            SystemMessage(content=(
                "You are an intent classifier. Does the user's message relate to travel planning, "
                "flights, hotels, destinations, trips, itineraries, or booking? "
                "Reply with ONLY the word 'travel' or 'general'. Nothing else."
            )),
            HumanMessage(content=query),
        ])
        is_travel = "travel" in response.content.strip().lower()
        return {
            "is_travel_query": is_travel,
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def route_query(state: TravelState) -> str:
        if state.get("is_travel_query", True):
            return "flight_agent"
        return "conversational_agent"

    def conversational_agent(state: TravelState):
        """Handle non-travel queries with a direct conversational response."""
        response = llm.invoke([
            SystemMessage(content=(
                "You are a friendly AI travel assistant. The user has asked something that isn't "
                "a travel planning request. Answer naturally and helpfully. If it seems like they "
                "might want travel help, gently mention you can plan trips for them."
            )),
            HumanMessage(content=state["user_query"]),
        ])
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def flight_agent(state: TravelState):
        query = state["user_query"]
        flight_data = search_flights(query)
        return {
            "flight_results": flight_data,
            "messages": [AIMessage(content="Flight results fetched")],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def hotel_agent(state: TravelState):
        query = f"Best hotels for {state['user_query']}"
        hotel_results = tavily_search(query)
        return {
            "hotel_results": hotel_results,
            "messages": [AIMessage(content="Hotel information fetched")],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def itinerary_agent(state: TravelState):
        prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
    """
        response = llm.invoke([
            SystemMessage(content="You are an expert travel planner"),
            HumanMessage(content=prompt),
        ])
        return {
            "itinerary": response.content,
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def final_agent(state: TravelState):
        final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """
        response = llm.invoke([HumanMessage(content=final_prompt)])
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    graph = StateGraph(TravelState)
    graph.add_node("router_node", router_node)
    graph.add_node("conversational_agent", conversational_agent)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("final_agent", final_agent)

    graph.add_edge(START, "router_node")
    graph.add_conditional_edges("router_node", route_query, {
        "flight_agent": "flight_agent",
        "conversational_agent": "conversational_agent",
    })
    graph.add_edge("conversational_agent", END)
    graph.add_edge("flight_agent", "hotel_agent")
    graph.add_edge("hotel_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "final_agent")
    graph.add_edge("final_agent", END)

    # min_size=0: don't pre-create connections (avoids Neon cold-start failures)
    # max_idle=120: recycle connections after 2 min idle (Neon kills after ~5 min)
    # check=check_connection: validate each connection before use
    # prepare_threshold=0: disable prepared statements for Neon's PgBouncer
    pool = ConnectionPool(
        db_url,
        min_size=0,
        max_size=5,
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        check=ConnectionPool.check_connection,
        max_idle=120.0,
        max_lifetime=300.0,
    )
    checkpointer = PostgresSaver(pool)
    try:
        checkpointer.setup()
    except Exception:
        pass  # Tables already exist from a previous run
    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    db_url = os.getenv("DATABASE_URL")
    app = build_app(db_url)
    config = {"configurable": {"thread_id": "user_sakhawat"}}

    user_input = input("Enter travel request: ")
    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
            "is_travel_query": True,
        },
        config=config,
    )

    print("\nFINAL RESPONSE:\n")
    for msg in result["messages"]:
        print(msg.content)
