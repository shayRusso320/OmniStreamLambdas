import boto3
import os
from datetime import datetime, timezone

dynamodb = boto3.resource("dynamodb")
table    = dynamodb.Table(os.environ["DYNAMODB_TABLE"])

def handler(event, context):
    attrs   = event["request"]["userAttributes"]
    now     = datetime.now(timezone.utc).isoformat()
    user_id = attrs["sub"]

    # Detect whether the user signed in via Google or native Cognito.
    identities = attrs.get("identities", "")
    provider   = "Google" if "Google" in identities else "Cognito"

    item = {
        "PK":             f"User#{user_id}",
        "SK":             "Metadata",
        "user_id":        user_id,
        "username":       event["userName"],
        "email":          attrs.get("email", ""),
        "email_verified": attrs.get("email_verified", "false") == "true",
        "name":           " ".join(filter(None, [
                              attrs.get("given_name"),
                              attrs.get("family_name")
                          ])),
        "phone":          attrs.get("phone_number", ""),
        "picture":        attrs.get("picture", ""),
        "provider":       provider,
        "status":         "ACTIVE",
        "created_at":     now,
        "updated_at":     now,
    }

    # ConditionExpression prevents overwriting an existing record
    # on Google re-authentication or accidental double trigger.
    try:
        table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(PK)"
        )
    except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
        # User already exists — skip silently.
        pass

    # Must return the event back to Cognito.
    return event
