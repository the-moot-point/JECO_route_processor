import math
import json
from pathlib import Path

import pandas as pd

from jeco_route_processor import Location
from jeco_route_processor.utils import (
    nearest_points,
    DateFormatter,
    WarehouseLocations,
    DataValidator,
    RouteStatistics,
    run_batch_processing,
    analyze_metadata_file,
)
import jeco_route_processor.route_processor as route_processor


def test_nearest_points_within_radius():
    locs = [
        Location(0.0, 0.0),
        Location(0.0, 0.001),
        Location(0.001, 0.0),
        Location(0.0005, 0.0005),
        Location(1.0, 1.0),
    ]
    target = Location(0.0, 0.0)
    idxs, dists = nearest_points(locs, target, 150)
    assert 4 not in idxs
    assert len(idxs) == 4
    assert math.isclose(dists[0], 0.0, abs_tol=1e-6)
    assert all(d <= 150 for d in dists)


def test_nearest_points_small_radius():
    locs = [
        Location(0.0, 0.0),
        Location(0.0, 0.001),
    ]
    target = Location(0.0, 0.0)
    idxs, _ = nearest_points(locs, target, 50)
    assert idxs == [0]


def test_date_formatter_parse_and_format():
    date_str = "January 15, 2025"
    parts = DateFormatter.parse_date_string(date_str)
    assert parts["month"] == "January"
    assert parts["day"] == "15"
    assert parts["year_short"] == "25"
    assert DateFormatter.normalize_date(date_str) == "1/15/2025"
    assert (
        DateFormatter.format_for_trips_filename(date_str, 604)
        == "Trips RT 604 Jan 25.json"
    )
    assert (
        DateFormatter.format_for_orders_filename(date_str)
        == "Orders January 2025.csv"
    )


def test_warehouse_locations_get_and_add(monkeypatch):
    monkeypatch.setattr(
        WarehouseLocations,
        "WAREHOUSES",
        WarehouseLocations.WAREHOUSES.copy(),
    )
    assert WarehouseLocations.get_warehouse("Austin")["name"].startswith("Austin")
    WarehouseLocations.add_warehouse(
        "TestLoc", "Test Name", 1.0, 2.0, "Addr"
    )
    assert WarehouseLocations.get_warehouse("TestLoc")["latitude"] == 1.0


def test_data_validator_validate_stops_and_trips():
    df = pd.DataFrame(
        {
            "Stop ID": [1, 2],
            "Sequence": [1, 1],
            "Customer Name": ["A", "B"],
            "Latitude": [10.0, None],
            "Longitude": [20.0, 30.0],
            "Route Num": [101, 101],
            "Phase": [1, 1],
            "Day Of Week": ["Mon", "Mon"],
            "Active": [True, True],
        }
    )
    warnings = DataValidator.validate_stops_data(df)
    assert any("missing coordinates" in w for w in warnings)
    assert any("duplicate sequences" in w for w in warnings)

    warnings = DataValidator.validate_trips_data({"trips": []})
    assert "No trips found in data" in warnings

    warnings = DataValidator.validate_trips_data(
        {
            "trips": [
                {
                    "startAddress": {"name": "Austin Warehouse"},
                    "endAddress": {"name": "Austin Warehouse"},
                    "distanceMeters": 2001,
                }
            ]
        }
    )
    assert any("missing coordinate data" in w for w in warnings)
    assert any("Long warehouse-to-warehouse" in w for w in warnings)


def test_route_statistics_calculate_route_efficiency():
    meta = {
        "summary": {
            "stops_visited": 2,
            "total_scheduled_stops": 4,
            "stops_with_orders": 1,
        },
        "gps_trips": [
            {
                "distanceMeters": 1000,
                "fuelConsumedMl": 500,
                "startMs": 0,
                "endMs": 3_600_000,
            }
        ],
    }
    stats = RouteStatistics.calculate_route_efficiency(meta)
    assert stats["stop_completion_rate"] == 50.0
    assert stats["order_completion_rate"] == 25.0
    assert stats["total_distance_km"] == 1.0
    assert stats["total_fuel_liters"] == 0.5
    assert stats["fuel_efficiency_km_per_liter"] == 2.0
    assert stats["avg_stop_time_minutes"] == 30.0


def test_run_batch_processing(monkeypatch, tmp_path):
    called = []

    class DummyProcessor:
        def __init__(self, config):
            self.config = config

        def process(self):
            called.append((self.config.route_number, self.config.date_str))
            return tmp_path / f"{self.config.route_number}_{self.config.date_str}.json"

    monkeypatch.setattr(route_processor, "RouteProcessor", DummyProcessor)
    results = run_batch_processing(tmp_path, [1], ["January 1, 2025"])
    assert results == [
        {
            "route": 1,
            "date": "January 1, 2025",
            "success": True,
            "output_path": tmp_path / "1_January 1, 2025.json",
        }
    ]
    assert called == [(1, "January 1, 2025")]


def test_analyze_metadata_file(tmp_path, capsys):
    meta = {
        "route_info": {"route_number": 1, "date": "Jan 1, 2025"},
        "summary": {
            "total_scheduled_stops": 2,
            "stops_visited": 2,
            "stops_with_orders": 1,
            "total_alerts": 0,
            "unscheduled_stops": 0,
        },
        "gps_trips": [],
        "alerts": [],
    }
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta))
    analyze_metadata_file(path)
    captured = capsys.readouterr()
    assert "Route Analysis" in captured.out
    assert "Scheduled Stops: 2" in captured.out
