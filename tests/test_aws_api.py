
import json
import subprocess

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


REGION = "ap-southeast-1"
PROFILE = "guest-shield"
STACK_NAME = "guest-shield-api"


def get_api_url():
    """Retrieve the deployed API endpoint from CloudFormation."""
    client = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    ).client("cloudformation")

    response = client.describe_stacks(StackName=STACK_NAME)

    outputs = response["Stacks"][0]["Outputs"]

    for output in outputs:
        if output["OutputKey"] == "GuestShieldApiUrl":
            return output["OutputValue"]

    raise RuntimeError("GuestShieldApiUrl not found")


def main():
    session = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    )

    credentials = session.get_credentials()

    if credentials is None:
        raise RuntimeError("AWS credentials not found")

    frozen_credentials = credentials.get_frozen_credentials()

    api_url = get_api_url()

    reservation = {
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
        "type_of_meal_plan": "Meal Plan 1",
        "required_car_parking_space": 0,
        "room_type_reserved": "Room_Type 1",
        "lead_time": 45,
        "arrival_year": 2018,
        "arrival_month": 10,
        "arrival_date": 17,
        "market_segment_type": "Online",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 120,
        "no_of_special_requests": 1,
    }

    body = json.dumps(reservation)

    aws_request = AWSRequest(
        method="POST",
        url=api_url,
        data=body,
        headers={
            "Content-Type": "application/json",
        },
    )

    SigV4Auth(
        frozen_credentials,
        "execute-api",
        REGION,
    ).add_auth(aws_request)

    signed_headers = dict(aws_request.headers.items())

    response = requests.post(
        api_url,
        data=body,
        headers=signed_headers,
        timeout=60,
    )

    print("HTTP status:", response.status_code)
    print("Response:", response.text)

    response.raise_for_status()


if __name__ == "__main__":
    main()
