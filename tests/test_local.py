
import json

from app.handler import lambda_handler


# Example reservation using categories from the hotel dataset.
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
    "avg_price_per_room": 120.0,
    "no_of_special_requests": 1,
}


event = {
    "body": json.dumps(reservation)
}

response = lambda_handler(event, None)

print("\n========== GUEST SHIELD API TEST ==========")
print(f"HTTP Status: {response['statusCode']}")

print(
    json.dumps(
        json.loads(response["body"]),
        indent=4,
    )
)

assert response["statusCode"] == 200, response["body"]
