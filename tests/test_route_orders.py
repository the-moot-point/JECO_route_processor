from pathlib import Path

from jeco_route_processor.route_processor import RouteDataLoader, RouteConfig


def test_get_route_orders_for_route_600():
    data_dir = Path("tests/fixtures")
    date_str = "January 20, 2025"
    config = RouteConfig(data_directory=data_dir, date_str=date_str, route_number=600)
    loader = RouteDataLoader(config)

    df = loader.get_route_orders(date_str, 600)

    assert len(df) == 18
    assert (df["Date"] == "1/20/2025").all()
    assert (df["Route Num"] == 600).all()
