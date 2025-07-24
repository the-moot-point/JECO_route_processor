import math
from jeco_route_processor import Location
from jeco_route_processor.utils import nearest_points

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
