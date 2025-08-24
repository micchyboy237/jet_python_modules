import os
import shutil
import asyncio
import uuid
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

from jet.file.utils import save_file
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Define agent tools for customer support


async def check_order_status(order_id: str) -> str:
    return f"Order {order_id} is currently in transit and expected to arrive in 2 days."


async def process_refund(order_id: str) -> str:
    return f"Refund for order {order_id} has been initiated."


async def provide_product_info(product_id: str) -> str:
    return f"Product {product_id} is a premium wireless headset with noise cancellation."


async def main() -> None:
    conversation_id = str(uuid.uuid4())

    # Create agents with specific roles
    support_agent = AssistantAgent(
        name="Support_Agent",
        model_client=MLXAutogenChatLLMAdapter(
            model="llama-3.2-3b-instruct-4bit", name="Support_Agent", conversation_id=conversation_id, log_dir=f"{OUTPUT_DIR}/support_chats"),
        description="Handles general customer inquiries and escalates complex issues.",
    )
    order_agent = AssistantAgent(
        name="Order_Agent",
        model_client=MLXAutogenChatLLMAdapter(
            model="llama-3.2-3b-instruct-4bit", name="Order_Agent", conversation_id=conversation_id, log_dir=f"{OUTPUT_DIR}/order_chats"),
        tools=[check_order_status],
        description="Specializes in checking order statuses.",
    )
    refund_agent = AssistantAgent(
        name="Refund_Agent",
        model_client=MLXAutogenChatLLMAdapter(
            model="llama-3.2-3b-instruct-4bit", name="Refund_Agent", conversation_id=conversation_id, log_dir=f"{OUTPUT_DIR}/refund_chats"),
        tools=[process_refund],
        description="Handles refund requests.",
    )
    product_agent = AssistantAgent(
        name="Product_Agent",
        model_client=MLXAutogenChatLLMAdapter(
            model="llama-3.2-3b-instruct-4bit", name="Product_Agent", conversation_id=conversation_id, log_dir=f"{OUTPUT_DIR}/product_chats"),
        tools=[provide_product_info],
        description="Provides detailed product information.",
    )

    # Define termination condition
    termination = TextMentionTermination("RESOLVED")

    # Create the SelectorGroupChat team
    team = SelectorGroupChat(
        participants=[support_agent, order_agent, refund_agent, product_agent],
        name="GroupChatManager",
        model_client=MLXAutogenChatLLMAdapter(
            model="llama-3.2-3b-instruct-4bit", name="GroupChatManager", conversation_id=conversation_id, log_dir=f"{OUTPUT_DIR}/group_chat_manager_chats"),
        termination_condition=termination,
        max_turns=5,
        selector_prompt="""You are managing a customer support team. The following roles are available:
{roles}.
Read the conversation history and select the next role from {participants} to respond. Only return the role name.
{history}""",
        allow_repeated_speaker=False,
    )

    # Run the team with a customer query
    task = "I have a question about my order #12345 and want to know about the product details."
    await Console(team.run_stream(task=task))

    state = team.save_state()
    save_file(state, f"{OUTPUT_DIR}/team_state.json")

if __name__ == "__main__":
    asyncio.run(main())
