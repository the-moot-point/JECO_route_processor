"""
Route Processing System - Utility Functions
Helper functions for date handling and data processing
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
from scipy.spatial import cKDTree
if TYPE_CHECKING:
    from .route_processor import Location
import logging

logger = logging.getLogger(__name__)


class DateFormatter:
    """Handles various date format conversions used in the system"""

    MONTH_ABBR = {
        'January': 'Jan', 'February': 'Feb', 'March': 'Mar',
        'April': 'Apr', 'May': 'May', 'June': 'Jun',
        'July': 'Jul', 'August': 'Aug', 'September': 'Sep',
        'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
    }

    @staticmethod
    def parse_date_string(date_str: str) -> Dict[str, str]:
        """
        Parse date string like "January 15, 2025" into components
        Returns dict with 'month', 'day', 'year', 'month_abbr', 'year_short'
        """
        try:
            parts = date_str.replace(',', '').split()
            month = parts[0]
            day = parts[1]
            year = parts[2]

            return {
                'month': month,
                'day': day,
                'year': year,
                'month_abbr': DateFormatter.MONTH_ABBR.get(month, month[:3]),
                'year_short': year[-2:],
                'year_month': f"{month} {year}"
            }
        except (IndexError, AttributeError) as e:
            logger.error(f"Error parsing date string '{date_str}': {e}")
            return {}

    @staticmethod
    def normalize_date(date_str: str) -> str:
        """Convert a user-supplied date to M/D/YYYY format used in data files."""
        try:
            dt = datetime.strptime(date_str, "%B %d, %Y")
            return f"{dt.month}/{dt.day}/{dt.year}"
        except ValueError as e:
            logger.error(f"Error normalizing date '{date_str}': {e}")
            return date_str

    @staticmethod
    def format_for_trips_filename(date_str: str, route_num: int) -> str:
        """
        Convert "January 15, 2025" to trips filename format
        Returns: "Trips RT 604 Jan 25.json"
        """
        date_parts = DateFormatter.parse_date_string(date_str)
        return f"Trips RT {route_num} {date_parts['month_abbr']} {date_parts['year_short']}.json"

    @staticmethod
    def format_for_orders_filename(date_str: str) -> str:
        """
        Convert "January 15, 2025" to orders filename format
        Returns: "Orders January 2025.csv"
        """
        date_parts = DateFormatter.parse_date_string(date_str)
        return f"Orders {date_parts['year_month']}.csv"


class WarehouseLocations:
    """Manages warehouse location data"""

    # Default warehouse coordinates
    WAREHOUSES = {
        'Austin': {
            'name': 'Austin Warehouse (AUS)',
            'latitude': 30.20037956,
            'longitude': -97.71581548,
            'address': '6269 E Stassney Ln, Austin, TX 78744, USA'
        },
        # Add more warehouses as needed
    }

    @classmethod
    def get_warehouse(cls, location_name: str) -> Optional[Dict]:
        """Get warehouse information by location name"""
        return cls.WAREHOUSES.get(location_name)

    @classmethod
    def add_warehouse(
        cls, location_name: str, name: str, lat: float, lon: float, address: str
    ) -> None:
        """Add a new warehouse location"""
        cls.WAREHOUSES[location_name] = {
            'name': name,
            'latitude': lat,
            'longitude': lon,
            'address': address
        }


class DataValidator:
    """Validates data integrity and provides warnings"""

    @staticmethod
    def validate_stops_data(stops_df: pd.DataFrame) -> List[str]:
        """Validate stops dataframe and return list of warnings"""
        warnings = []

        # Check for required columns
        required_cols = ['Stop ID', 'Sequence', 'Customer Name', 'Latitude', 'Longitude',
                         'Route Num', 'Phase', 'Day Of Week', 'Active']
        missing_cols = [col for col in required_cols if col not in stops_df.columns]
        if missing_cols:
            warnings.append(f"Missing required columns: {missing_cols}")

        # Check for null coordinates
        if 'Latitude' in stops_df.columns and 'Longitude' in stops_df.columns:
            null_coords = stops_df[stops_df['Latitude'].isna() | stops_df['Longitude'].isna()]
            if not null_coords.empty:
                warnings.append(f"Found {len(null_coords)} stops with missing coordinates")

        # Check for duplicate sequences
        if 'Sequence' in stops_df.columns:
            dup_sequences = stops_df[stops_df.duplicated(subset=['Sequence'], keep=False)]
            if not dup_sequences.empty:
                warnings.append(f"Found {len(dup_sequences)} stops with duplicate sequences")

        return warnings

    @staticmethod
    def validate_trips_data(trips_data: Dict) -> List[str]:
        """Validate trips data and return list of warnings"""
        warnings = []

        if 'trips' not in trips_data:
            warnings.append("No 'trips' key found in trips data")
            return warnings

        trips = trips_data['trips']
        if not trips:
            warnings.append("No trips found in data")
            return warnings

        # Check for trips with missing coordinates
        for i, trip in enumerate(trips):
            if 'startCoordinates' not in trip or 'endCoordinates' not in trip:
                warnings.append(f"Trip {i} missing coordinate data")

            # Check for warehouse-to-warehouse trips (often indicates data issues)
            if ('startAddress' in trip and 'endAddress' in trip and
                    'warehouse' in trip['startAddress']['name'].lower() and
                    'warehouse' in trip['endAddress']['name'].lower()):
                if trip.get('distanceMeters', 0) > 1000:  # More than 1km
                    warnings.append(f"Trip {i}: Long warehouse-to-warehouse trip detected")

        return warnings


def nearest_points(locations: Sequence[Location], target: Location, radius: float) -> Tuple[List[int], List[float]]:
    """Return indices and distances of locations within ``radius`` meters of ``target``.

    Distances are calculated using ``geopy`` for accuracy, while ``scipy``'s
    ``cKDTree`` is used to efficiently pre-filter points in a projected space.
    The results are sorted by increasing distance.
    """

    if not locations:
        return [], []

    # Build KD-tree using an approximate meter projection (1 deg ~= 111 km)
    coords = [(loc.latitude, loc.longitude) for loc in locations]
    tree = cKDTree(coords)
    radius_deg = radius / 111_000

    candidate_idx = tree.query_ball_point([target.latitude, target.longitude], radius_deg)

    distances = [
        target.distance_to(locations[i])
        for i in candidate_idx
    ]

    sorted_pairs = sorted(zip(candidate_idx, distances), key=lambda x: x[1])
    if not sorted_pairs:
        return [], []
    idx_sorted, dist_sorted = zip(*sorted_pairs)
    return list(idx_sorted), list(dist_sorted)


class RouteStatistics:
    """Calculate route performance statistics"""

    @staticmethod
    def calculate_route_efficiency(metadata: Dict) -> Dict:
        """Calculate various efficiency metrics from metadata"""
        summary = metadata['summary']

        # Basic efficiency metrics
        stop_completion_rate = (summary['stops_visited'] / summary['total_scheduled_stops'] * 100
                                if summary['total_scheduled_stops'] > 0 else 0)

        order_completion_rate = (summary['stops_with_orders'] / summary['total_scheduled_stops'] * 100
                                 if summary['total_scheduled_stops'] > 0 else 0)

        # Calculate total distance from GPS trips
        total_distance_meters = sum(trip.get('distanceMeters', 0) for trip in metadata['gps_trips'])
        total_distance_km = total_distance_meters / 1000

        # Calculate total fuel consumption
        total_fuel_ml = sum(trip.get('fuelConsumedMl', 0) for trip in metadata['gps_trips'])
        total_fuel_liters = total_fuel_ml / 1000

        # Calculate time metrics
        trips = metadata['gps_trips']
        if trips:
            start_time = min(trip['startMs'] for trip in trips)
            end_time = max(trip['endMs'] for trip in trips)
            total_time_ms = end_time - start_time
            total_time_hours = total_time_ms / (1000 * 60 * 60)
        else:
            total_time_hours = 0

        return {
            'stop_completion_rate': round(stop_completion_rate, 1),
            'order_completion_rate': round(order_completion_rate, 1),
            'total_distance_km': round(total_distance_km, 1),
            'total_fuel_liters': round(total_fuel_liters, 1),
            'total_time_hours': round(total_time_hours, 1),
            'fuel_efficiency_km_per_liter': round(total_distance_km / total_fuel_liters,
                                                  1) if total_fuel_liters > 0 else 0,
            'avg_stop_time_minutes': round(total_time_hours * 60 / summary['stops_visited'], 1) if summary[
                                                                                                       'stops_visited'] > 0 else 0
        }


def run_batch_processing(
    data_dir: Path, routes: List[int], dates: List[str]
) -> List[Dict]:
    """
    Process multiple routes and dates in batch

    Args:
        data_dir: Path to data directory
        routes: List of route numbers to process
        dates: List of date strings to process
    """
    from .route_processor import RouteProcessor, RouteConfig

    results = []

    for route_num in routes:
        for date_str in dates:
            logger.info(f"Processing RT{route_num} for {date_str}")

            config = RouteConfig(
                data_directory=data_dir,
                date_str=date_str,
                route_number=route_num
            )

            processor = RouteProcessor(config)
            output_path = processor.process()

            results.append({
                'route': route_num,
                'date': date_str,
                'success': output_path is not None,
                'output_path': output_path
            })

    # Print summary
    successful = sum(1 for r in results if r['success'])
    logger.info(f"\nBatch processing complete: {successful}/{len(results)} successful")

    return results


def analyze_metadata_file(metadata_path: Path) -> None:
    """Load and analyze a metadata file, printing key statistics"""
    import json

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\n📊 Route Analysis for RT{metadata['route_info']['route_number']} - {metadata['route_info']['date']}")
    print("=" * 60)

    # Summary stats
    summary = metadata['summary']
    print("\n📈 Summary:")
    total_scheduled = summary['total_scheduled_stops']
    print(f"  • Scheduled Stops: {total_scheduled}")
    if total_scheduled > 0:
        visited_pct = summary['stops_visited'] / total_scheduled * 100
        orders_pct = summary['stops_with_orders'] / total_scheduled * 100
        print(f"  • Stops Visited: {summary['stops_visited']} ({visited_pct:.1f}%)")
        print(f"  • Orders Completed: {summary['stops_with_orders']} ({orders_pct:.1f}%)")
    else:
        print(f"  • Stops Visited: {summary['stops_visited']} (0%)")
        print(f"  • Orders Completed: {summary['stops_with_orders']} (0%)")
    print(f"  • Total Alerts: {summary['total_alerts']}")
    print(f"  • Unscheduled Stops: {summary['unscheduled_stops']}")

    # Performance metrics
    stats = RouteStatistics.calculate_route_efficiency(metadata)
    print("\n🚚 Performance Metrics:")
    print(f"  • Total Distance: {stats['total_distance_km']} km")
    print(f"  • Total Fuel: {stats['total_fuel_liters']} L")
    print(f"  • Fuel Efficiency: {stats['fuel_efficiency_km_per_liter']} km/L")
    print(f"  • Total Time: {stats['total_time_hours']} hours")
    print(f"  • Avg Time per Stop: {stats['avg_stop_time_minutes']} minutes")

    # Alert breakdown
    alerts_by_type = {}
    for alert in metadata['alerts']:
        alert_type = alert['type']
        if alert_type not in alerts_by_type:
            alerts_by_type[alert_type] = []
        alerts_by_type[alert_type].append(alert)

    print("\n⚠️  Alert Breakdown:")
    for alert_type, alerts in alerts_by_type.items():
        print(f"  • {alert_type}: {len(alerts)}")
        for alert in alerts[:3]:  # Show first 3 of each type
            print(f"    - {alert['description']}")
        if len(alerts) > 3:
            print(f"    ... and {len(alerts) - 3} more")


if __name__ == "__main__":
    # Example usage
    data_dir = Path("./data")

    # Test date formatting
    test_date = "January 15, 2025"
    print(f"Date: {test_date}")
    print(f"Trips filename: {DateFormatter.format_for_trips_filename(test_date, 604)}")
    print(f"Orders filename: {DateFormatter.format_for_orders_filename(test_date)}")

    # Example batch processing
    # routes = [600, 601, 602, 603, 604]
    # dates = ["January 15, 2025", "January 16, 2025"]
    # run_batch_processing(data_dir, routes, dates)
