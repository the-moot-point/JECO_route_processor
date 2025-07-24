import json
from pathlib import Path

from jeco_route_processor.route_processor import RouteDataLoader, RouteConfig
from jeco_route_processor.utils import DateFormatter


def test_load_trips_filters_by_day():
    data_dir = Path("tests/fixtures")
    date_str = "January 20, 2025"
    config = RouteConfig(data_directory=data_dir, date_str=date_str, route_number=604)
    loader = RouteDataLoader(config)

    result = loader.load_trips(604, date_str)

    with open(data_dir / "Trips" / "Trips RT 604 Jan 25.json", "r") as f:
        full_trips = json.load(f)["trips"]

    start_ms, end_ms = DateFormatter.get_day_bounds_ms(date_str)

    assert len(result["trips"]) < len(full_trips)
    assert all(start_ms <= t["startMs"] < end_ms for t in result["trips"])

