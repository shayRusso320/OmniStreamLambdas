import json
import os
import uuid
from datetime import datetime, timezone

import boto3
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

# ── Clients ──────────────────────────────────────────────────────────────────

dynamodb = boto3.resource("dynamodb")
table    = dynamodb.Table(os.environ["DYNAMODB_TABLE"])

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]
EMBEDDING_MODEL   = "text-embedding-3-small"

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_user_id(event: dict) -> str:
    """Extract the Cognito sub from the JWT claims injected by API Gateway."""
    return event["requestContext"]["authorizer"]["jwt"]["claims"]["sub"]


def get_topic_name(event: dict) -> str:
    body = json.loads(event.get("body") or "{}")
    topic = body.get("topic", "").strip()
    if not topic:
        raise ValueError("Missing 'topic' in request body.")
    return topic


def find_existing_topic(topic_name: str) -> dict | None:
    """
    Search Qdrant for an existing topic vector whose payload matches the name.
    Returns the Qdrant point (with vector_id and payload) or None.
    """
    results = qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="topic_name",
                    match=MatchValue(value=topic_name.lower()),
                )
            ]
        ),
        limit=1,
        with_payload=True,
    )
    points, _ = results
    return points[0] if points else None


def embed_topic(topic_name: str) -> list[float]:
    """Call OpenAI to get an embedding vector for the topic name."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=topic_name,
    )
    return response.data[0].embedding


def store_vector(topic_name: str, vector: list[float]) -> str:
    """Store the vector in Qdrant and return the generated vector ID."""
    vector_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=vector_id,
                vector=vector,
                payload={
                    "type": "topic",
                    "topic_name": topic_name.lower(),
                },
            )
        ],
    )
    return vector_id


def create_topic_metadata(vector_id: str, topic_name: str) -> None:
    """Write the Topic metadata row to DynamoDB."""
    table.put_item(
        Item={
            "PK":           f"Topic#{vector_id}",
            "SK":           "Metadata",
            "topic_name":   topic_name,
            "vector_id":    vector_id,
            "live_summary": "",
            "snippets":     [],
            "need_summary": False,
        },
        # Safety guard — don't overwrite if it somehow already exists.
        ConditionExpression="attribute_not_exists(PK)",
    )


def create_user_topic_link(user_id: str, vector_id: str) -> None:
    """Write the user-topic link row to DynamoDB."""
    table.put_item(
        Item={
            "PK":       f"User#{user_id}",
            "SK":       f"Topic#{vector_id}",
            "added_on": datetime.now(timezone.utc).isoformat(),
        }
    )

# ── Handler ───────────────────────────────────────────────────────────────────

def handler(event, context):
    try:
        user_id = get_user_id(event)
        topic_name = get_topic_name(event)
    except (KeyError, ValueError) as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}

    try:
        # 1. Check if the topic already exists in Qdrant.
        existing = find_existing_topic(topic_name)

        if existing is None:
            # 2.1 Embed the topic name.
            vector = embed_topic(topic_name)

            # 2.2 Store in Qdrant, get back the vector ID.
            vector_id = store_vector(topic_name, vector)

            # 2.3 Create the topic metadata row in DynamoDB.
            create_topic_metadata(vector_id, topic_name)
        else:
            vector_id = str(existing.id)

        # 3. Always create the user↔topic link.
        create_user_topic_link(user_id, vector_id)

    except Exception as e:
        print(f"ERROR: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": "Internal server error."})}

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Topic attached successfully.",
            "topic": topic_name,
            "vectorId": vector_id,
        }),
    }
