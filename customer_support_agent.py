import streamlit as st
import os
import json
from datetime import datetime, timedelta
from openai import OpenAI
from mem0 import Memory

# --- LangGraph & LangChain Imports ---
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Set up the Streamlit App
st.title("AI Customer Support Agent with Memory 🛒")
st.caption("Chat with a highly capable LangGraph agent that autonomously searches your past interactions.")

# Set the OpenAI API key
openai_api_key = st.text_input("Enter OpenAI API Key", type="password")

if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key

    class CustomerSupportAIAgent:
        def __init__(self, api_key):
            # 1. Initialize Mem0 with Qdrant as the vector store
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": "localhost",
                        "port": 6333,
                    }
                },
            }
            try:
                self.memory = Memory.from_config(config)
            except Exception as e:
                st.error(f"Failed to initialize memory: {e}")
                st.stop() 

            # 2. Initialize standard OpenAI client for data generation
            self.client = OpenAI(api_key=api_key)
            self.app_id = "customer-support"

            # 3. Build the LangGraph Agent workflow
            self._build_agent_graph(api_key)

        def _build_agent_graph(self, api_key):
            """Compiles the LangGraph workflow defining the Agent's brain and tools."""
            
            # A. Define the Tool the agent can use
            def search_customer_history(query: str, user_id: str) -> str:
                """Search the customer's past orders and interactions."""
                try:
                    relevant_memories = self.memory.search(query=query, user_id=user_id)
                    context = ""
                    if relevant_memories and "results" in relevant_memories:
                        for mem in relevant_memories["results"]:
                            if "memory" in mem:
                                context += f"- {mem['memory']}\n"
                    return context if context else "No relevant history found in database."
                except Exception as e:
                    return f"Error accessing memory: {str(e)}"

            search_tool = StructuredTool.from_function(
                func=search_customer_history,
                name="search_customer_history",
                description="Search the customer's past orders, preferences, and interactions. Always provide the query and user_id."
            )
            tools = [search_tool]

            # B. Define the Graph State
            class AgentState(TypedDict):
                messages: Annotated[list[BaseMessage], add_messages]
                user_id: str

            # C. Initialize the LLM and bind the tools to it
            llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
            llm_with_tools = llm.bind_tools(tools)

            # D. Define the Agent Node
            def agent_node(state: AgentState):
                # We inject a system message dynamically with the specific customer ID
                sys_msg = SystemMessage(
                    content=f"You are a helpful customer support AI agent for TechGadgets.com. "
                            f"The current customer's ID is {state['user_id']}. "
                            f"If you need context about their past orders or issues, use the search_customer_history tool."
                )
                messages_to_process = [sys_msg] + state["messages"]
                response = llm_with_tools.invoke(messages_to_process)
                return {"messages": [response]}

            # E. Define the Routing Logic
            def should_continue(state: AgentState):
                last_message = state["messages"][-1]
                # If the LLM decided to call a tool, route to the tools node
                if last_message.tool_calls:
                    return "tools"
                # Otherwise, end the execution
                return END

            # F. Construct and compile the Graph
            workflow = StateGraph(AgentState)
            workflow.add_node("agent", agent_node)
            workflow.add_node("tools", ToolNode(tools))

            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges("agent", should_continue)
            workflow.add_edge("tools", "agent")

            self.graph = workflow.compile()

        def handle_query(self, query, user_id=None):
            try:
                # 1. Prepare inputs for LangGraph
                inputs = {
                    "messages": [HumanMessage(content=query)],
                    "user_id": user_id
                }

                # 2. Invoke the autonomous agent graph
                final_state = self.graph.invoke(inputs)
                
                # 3. Extract final response
                answer = final_state["messages"][-1].content

                # 4. Save the new interaction to Mem0 for future reference
                self.memory.add(query, user_id=user_id, metadata={"app_id": self.app_id, "role": "user"})
                self.memory.add(answer, user_id=user_id, metadata={"app_id": self.app_id, "role": "assistant"})

                return answer
            except Exception as e:
                st.error(f"An error occurred while handling the query: {e}")
                return "Sorry, I encountered an error. Please try again later."

        def get_memories(self, user_id=None):
            try:
                return self.memory.get_all(user_id=user_id)
            except Exception as e:
                st.error(f"Failed to retrieve memories: {e}")
                return None

        def generate_synthetic_data(self, user_id: str) -> dict | None:
            try:
                today = datetime.now()
                order_date = (today - timedelta(days=10)).strftime("%B %d, %Y")
                expected_delivery = (today + timedelta(days=2)).strftime("%B %d, %Y")

                prompt = f"""Generate a detailed customer profile and order history for a TechGadgets.com customer with ID {user_id}. Include:
                1. Customer name and basic info
                2. A recent order of a high-end electronic device (placed on {order_date}, to be delivered by {expected_delivery})
                3. Order details (product, price, order number)
                4. Customer's shipping address
                5. 2-3 previous orders from the past year
                6. 2-3 customer service interactions related to these orders
                7. Any preferences or patterns in their shopping behavior

                Format the output as a JSON object."""

                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a data generation AI that creates realistic customer profiles and order histories. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ]
                )

                customer_data = json.loads(response.choices[0].message.content)

                for key, value in customer_data.items():
                    if isinstance(value, list):
                        for item in value:
                            self.memory.add(
                                json.dumps(item), 
                                user_id=user_id, 
                                metadata={"app_id": self.app_id, "role": "system"}
                            )
                    else:
                        self.memory.add(
                            f"{key}: {json.dumps(value)}", 
                            user_id=user_id, 
                            metadata={"app_id": self.app_id, "role": "system"}
                        )

                return customer_data
            except Exception as e:
                st.error(f"Failed to generate synthetic data: {e}")
                return None

    # Initialize the Agent
    support_agent = CustomerSupportAIAgent(openai_api_key)

    # --- Sidebar UI ---
    st.sidebar.title("Customer Context Setup")
    previous_customer_id = st.session_state.get("previous_customer_id", None)
    customer_id = st.sidebar.text_input("Enter your Customer ID")

    if customer_id != previous_customer_id:
        st.session_state.messages = []
        st.session_state.previous_customer_id = customer_id
        st.session_state.customer_data = None

    if st.sidebar.button("Generate Synthetic Data"):
        if customer_id:
            with st.spinner("Generating customer data into Mem0..."):
                st.session_state.customer_data = support_agent.generate_synthetic_data(customer_id)
            if st.session_state.customer_data:
                st.sidebar.success("Synthetic data generated successfully!")
            else:
                st.sidebar.error("Failed to generate synthetic data.")
        else:
            st.sidebar.error("Please enter a customer ID first.")

    if st.sidebar.button("View Customer Profile"):
        if st.session_state.customer_data:
            st.sidebar.json(st.session_state.customer_data)
        else:
            st.sidebar.info("No customer data generated yet. Click 'Generate Synthetic Data' first.")

    if st.sidebar.button("View Memory Info"):
        if customer_id:
            memories = support_agent.get_memories(user_id=customer_id)
            if memories:
                st.sidebar.write(f"Memory for customer **{customer_id}**:")
                if memories and "results" in memories:
                    for memory in memories["results"]:
                        if "memory" in memory:
                            st.sidebar.write(f"- {memory['memory']}")
            else:
                st.sidebar.info("No memory found for this customer ID.")
        else:
            st.sidebar.error("Please enter a customer ID to view memory info.")

    # --- Main Chat UI ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("How can I assist you today?")

    if query and customer_id:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Agent is thinking (and searching memory if needed)..."):
            answer = support_agent.handle_query(query, user_id=customer_id)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    elif not customer_id:
        st.info("Please enter a Customer ID in the sidebar to start chatting.")

else:
    st.warning("Please enter your OpenAI API key to start the application.")
