import json
from typing import List

import uuid
from langgraph.types import Command

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain.agents import create_agent

from cassandra.cluster import Cluster

from dotenv import load_dotenv

load_dotenv()


# ============ 1. CASSANDRA SETUP - MATCH YOUR JSON ============
print("🟢 Connecting to Cassandra...")
cluster = Cluster(["localhost"], port=9042)
session = cluster.connect()
session.execute("CREATE KEYSPACE IF NOT EXISTS datagen WITH REPLICATION = {'class': 'SimpleStrategy', 'replication_factor': 1}")
session.set_keyspace("datagen")

# Create table with EXACT fields from your JSON
session.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY,
        user_id INT,
        first_name TEXT,
        last_name TEXT,
        email TEXT,
        username TEXT,
        age INT,
        registered_at TIMESTAMP,
        full_text TEXT,
        vector VECTOR<FLOAT, 1536>
    )
""")


# Create vector index
session.execute("""
    CREATE CUSTOM INDEX IF NOT EXISTS users_vector_idx 
    ON users (vector) USING 'StorageAttachedIndex'
""")

embeddings = OpenAIEmbeddings()
print("✅ Cassandra ready with JSON structure!")


@tool
def write_json(filepath: str, data: dict) -> str:
    """Write a Python dictionary as JSON to a file with pretty formatting."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"Successfully wrote JSON data to '{filepath}' ({len(json.dumps(data))} characters)."
    except Exception as e:
        return f"Error writing JSON: {str(e)}"


@tool
def read_json(filepath: str) -> str:
    """Read and return the contents of a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file - {str(e)}"
    except Exception as e:
        return f"Error reading JSON: {str(e)}"


@tool
def save_user_from_json(user_data: dict) -> str:
    """Save a user that matches the JSON format.
    Example: {"id": 1, "firstName": "Carol", "lastName": "Williams", "email": "carol@email.com",
              "username": "carol236", "age": 25, "registeredAt": "2025-07-28T08:26:01.167890"}"""
    try:
        # Extract data from JSON
        user_id = user_data.get("id")
        first_name = user_data.get("firstName")
        last_name = user_data.get("lastName")
        email = user_data.get("email")
        username = user_data.get("username")
        age = user_data.get("age")
        registered_at = user_data.get("registeredAt")

        # Create searchable text
        full_text = f"{first_name} {last_name}, {age} years old, {email}, username: {username}"

        # Generate embedding
        vector = embeddings.embed_query(full_text)

        # Save to Cassandra - match ALL fields
        session.execute("""
                        INSERT INTO users (id, user_id, first_name, last_name, email, username,
                                           age, registered_at, full_text, vector)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            uuid.uuid4(), user_id, first_name, last_name, email, username,
                            age, registered_at, full_text, vector
                        ))

        return f"✅ Saved user: {first_name} {last_name}, {age} years old, {email}"
    except Exception as e:
        return f"❌ Error saving user: {str(e)}"


@tool
def save_multiple_users_from_json(users_data: dict) -> str:
    """Save multiple users from generate_sample_users output.
    Input should be {"users": [...]} with your JSON format."""
    try:
        users = users_data.get("users", [])
        if not users:
            return "❌ No users to save"

        count = 0
        for user in users:
            # Extract fields - match your JSON exactly
            user_id = user.get("id")
            first_name = user.get("firstName")
            last_name = user.get("lastName")
            email = user.get("email")
            username = user.get("username")
            age = user.get("age")
            registered_at = user.get("registeredAt")
            # Create searchable text
            full_text = f"{first_name} {last_name}, {age} years old, {email}, username: {username}"

            # Generate embedding
            vector = embeddings.embed_query(full_text)

            # Save to Cassandra
            session.execute("""
                            INSERT INTO users (id, user_id, first_name, last_name, email, username,
                                               age, registered_at, full_text, vector)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                uuid.uuid4(), user_id, first_name, last_name, email, username,
                                age, registered_at, full_text, vector
                            ))
            count += 1

        return f"✅ Saved {count} users to Cassandra"
    except Exception as e:
        return f"❌ Error saving users: {str(e)}"


@tool
def find_users(query: str) -> str:
    """Find similar users using vector search."""
    try:
        vector = embeddings.embed_query(query)
        rows = session.execute("""
                               SELECT first_name, last_name, age, email, username
                               FROM users
                               ORDER BY vector ANN OF %s
                                   LIMIT 3
                               """, [vector])

        results = list(rows)
        if not results:
            return f"🔍 No users found matching '{query}'"

        response = f"🔍 Found users similar to '{query}':\n"
        for r in results:
            response += f"  • {r.first_name} {r.last_name}, {r.age} yrs, {r.email}, username: {r.username}\n"
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============ 4. AGENT SETUP ============
TOOLS = [write_json, read_json,save_user_from_json, save_multiple_users_from_json, find_users]

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


SYSTEM_MESSAGE = """
You are a helpful assistant that generates and saves users to a vector database.

RULES:
1. To GENERATE users → use generate_users
2. To SAVE users → use save_users IMMEDIATELY after generating
3. To FIND users → use find_users directly - DO NOT ask to generate first

The database is PERSISTENT - users stay saved even after restart.
"""

agent = create_agent(llm, TOOLS, system_prompt=SYSTEM_MESSAGE)


def run_agent(user_input: str, history: List[BaseMessage]) -> AIMessage:
    """Single-turn agent runner with automatic tool execution via LangGraph."""
    try:
        inputs = Command(update={"messages": history + [HumanMessage(content=user_input)]})
        config: RunnableConfig = {"recursion_limit": 50}

        result = agent.invoke(inputs, config=config)
        # Return the last AI message
        return result["messages"][-1]
    except Exception as e:
        # Return error as an AI message so the conversation can continue
        return AIMessage(content=f"Error: {str(e)}\n\nPlease try rephrasing your request or provide more specific details.")


if __name__ == "__main__":
    print("=" * 60)
    print("DataGen Agent - Sample Data Generator")
    print("=" * 60)
    print("Generate sample user data and save to JSON files.")
    print()
    print("Examples:")
    print("  - Generate users named John, Jane, Mike and save to users.json")
    print("  - Create users with last names Smith, Jones")
    print("  - Make users aged 25-35 with company.com emails")
    print()
    print("Commands: 'quit' or 'exit' to end")
    print("=" * 60)

    history: List[BaseMessage] = []

    while True:
        user_input = input("You: ").strip()

        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q', ""]:
            print("Goodbye!")
            break

        print("Agent: ", end="", flush=True)
        response = run_agent(user_input, history)
        print(response.content)
        print()

        # Update conversation history
        history += [HumanMessage(content=user_input), response]