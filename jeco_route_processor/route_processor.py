"""
Route Processing System - Metadata Generator
Processes route data and generates metadata JSON for interactive map visualization
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from geopy.distance import distance
import logging
from .utils import DateFormatter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RouteConfig:
    """Configuration for route processing"""
    data_directory: Path
    date_str: str  # e.g., "January 15, 2025"
    route_number: int
    alert_radius_meters: float = 100.0
    unscheduled_stop_threshold_meters: float = 150.0  # Distance from any scheduled stop to be considered unscheduled


@dataclass
class Location:
    """Represents a geographic location"""
    latitude: float
    longitude: float

    def distance_to(self, other: 'Location') -> float:
        """Calculate distance in meters to another location"""
        return distance((self.latitude, self.longitude),
                       (other.latitude, other.longitude)).meters


@dataclass
class Alert:
    """Represents an alert/issue found during analysis"""
    type: str  # 'missed_stop', 'no_order', 'unscheduled_stop'
    severity: str  # 'high', 'medium', 'low'
    location: Location
    description: str
    details: Dict[str, Any]


class RouteDataLoader:
    """Handles loading all data files"""

    def __init__(self, config: RouteConfig):
        self.config = config
        self.data_dir = config.data_directory

    def load_date_conversions(self) -> pd.DataFrame:
        """Load date conversions table"""
        file_path = self.data_dir / 'Dates' / 'DateConversionsTable.csv'
        logger.info(f"Loading date conversions from {file_path}")
        return pd.read_csv(file_path)

    def load_routes(self) -> pd.DataFrame:
        """Load routes information"""
        file_path = self.data_dir / 'Routes' / 'Routes.csv'
        logger.info(f"Loading routes from {file_path}")
        return pd.read_csv(file_path)

    def load_all_stops(self) -> pd.DataFrame:
        """Load all stops data"""
        file_path = self.data_dir / 'Stops' / 'Stops.csv'
        logger.info(f"Loading all stops from {file_path}")
        return pd.read_csv(file_path)

    def load_orders(self, year_month: str) -> pd.DataFrame:
        """Load orders for a specific month"""
        file_path = self.data_dir / 'Orders' / f'Orders {year_month}.csv'
        logger.info(f"Loading orders from {file_path}")
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            logger.warning(f"Orders file not found: {file_path}")
            return pd.DataFrame()

    def load_trips(self, route_num: int, date_str: str) -> Dict:
        """Load GPS trip data"""
        # Convert date format for filename (e.g., "January 15, 2025" -> "Jan 25")
        date_parts = date_str.split()
        month_abbr = date_parts[0][:3]
        year_short = date_parts[2][-2:]
        filename = f'Trips RT {route_num} {month_abbr} {year_short}.json'

        file_path = self.data_dir / 'Trips' / filename
        logger.info(f"Loading trips from {file_path}")

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Trips file not found: {file_path}")
            return {"trips": []}

    @staticmethod
    def _normalize_phase(phase: str) -> str:
        """Normalize phase names to a numeric form for comparison"""
        if phase is None:
            return ""
        p = str(phase).strip().lower()
        mapping = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "1 only": "1",
            "2 only": "2",
            "3 only": "3",
            "4 only": "4",
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "all": "all",
        }
        return mapping.get(p, p)

    def get_scheduled_stops(self, date_str: str, route_num: int) -> pd.DataFrame:
        """Get scheduled stops for a specific date and route"""
        # Load date conversions to get Phase and Day of Week
        date_conv_df = self.load_date_conversions()
        normalized_date = DateFormatter.normalize_date(date_str)
        date_info = date_conv_df[date_conv_df['Date'] == normalized_date]

        if date_info.empty:
            logger.error(f"Date {date_str} not found in date conversions table")
            return pd.DataFrame()

        phase = date_info.iloc[0]['Phase']
        phase_norm = self._normalize_phase(phase)
        day_of_week = date_info.iloc[0]['Day Of Week']

        # Load all stops and filter
        all_stops_df = self.load_all_stops()
        active_col = all_stops_df['Active'].astype(str).str.lower()
        stop_phase_norm = all_stops_df['Phase'].astype(str).apply(self._normalize_phase)
        scheduled_stops = all_stops_df[
            (all_stops_df['Route Num'] == route_num) &
            ((stop_phase_norm == phase_norm) | (stop_phase_norm == 'all')) &
            (all_stops_df['Day Of Week'] == day_of_week) &
            (active_col.isin(['y', 'active']))
        ].copy()

        # Sort by sequence
        scheduled_stops = scheduled_stops.sort_values('Sequence').reset_index(drop=True)

        logger.info(
            f"Found {len(scheduled_stops)} scheduled stops for RT{route_num} on {date_str}"
        )
        return scheduled_stops

    def get_route_orders(self, date_str: str, route_num: int) -> pd.DataFrame:
        """Get orders for a specific date and route"""
        # Extract year and month from date string
        date_parts = date_str.split()
        year_month = f"{date_parts[0]} {date_parts[2]}"

        orders_df = self.load_orders(year_month)
        if orders_df.empty:
            return pd.DataFrame()

        # Filter for specific date and route
        normalized_date = DateFormatter.normalize_date(date_str)
        route_orders = orders_df[
            (orders_df['Date'] == normalized_date) &
            (orders_df['Route Num'] == route_num)
        ].copy()

        logger.info(f"Found {len(route_orders)} orders for RT{route_num} on {date_str}")
        return route_orders


class RouteAnalyzer:
    """Performs route analysis and generates alerts"""

    def __init__(self, config: RouteConfig):
        self.config = config
        self.alerts: List[Alert] = []

    def analyze_stop_visits(self, stops_df: pd.DataFrame, trips_data: Dict,
                           orders_df: pd.DataFrame) -> List[Dict]:
        """Analyze each scheduled stop for GPS visits and orders"""
        stop_analyses = []

        for idx, stop in stops_df.iterrows():
            stop_loc = Location(stop['Latitude'], stop['Longitude'])

            # Check for GPS visits
            gps_visits = self._find_gps_visits(stop_loc, trips_data['trips'])

            # Check for orders
            nearby_orders = self._find_nearby_orders(stop_loc, orders_df)

            # Determine alerts
            if not gps_visits:
                self.alerts.append(Alert(
                    type='missed_stop',
                    severity='high',
                    location=stop_loc,
                    description=f"No GPS visit found for stop #{stop['Sequence']}",
                    details={'stop_id': stop['Stop ID'], 'customer': stop['Customer Name']}
                ))

            if not nearby_orders:
                self.alerts.append(Alert(
                    type='no_order',
                    severity='medium',
                    location=stop_loc,
                    description=f"No order entry found for stop #{stop['Sequence']}",
                    details={'stop_id': stop['Stop ID'], 'customer': stop['Customer Name']}
                ))

            stop_analyses.append({
                'stop_id': stop['Stop ID'],
                'sequence': int(stop['Sequence']),
                'customer_name': stop['Customer Name'],
                'location': asdict(stop_loc),
                'gps_visits': gps_visits,
                'orders': nearby_orders,
                'has_gps_visit': len(gps_visits) > 0,
                'has_order': len(nearby_orders) > 0
            })

        return stop_analyses

    def find_unscheduled_stops(self, trips_data: Dict, stops_df: pd.DataFrame,
                               warehouse_locations: List[Location]) -> List[Dict]:
        """Find GPS stops that don't correspond to scheduled stops"""
        unscheduled_stops = []

        # Get all scheduled stop locations
        scheduled_locations = [
            Location(row['Latitude'], row['Longitude'])
            for _, row in stops_df.iterrows()
        ]

        for trip in trips_data['trips']:
            # Check both start and end locations
            for location_type, coords in [
                ('arrival', trip['endCoordinates']),
                ('departure', trip['startCoordinates'])
            ]:
                trip_loc = Location(coords['latitude'], coords['longitude'])

                # Skip if it's a warehouse location
                if any(trip_loc.distance_to(wh) < 50 for wh in warehouse_locations):
                    continue

                # Check if it's near any scheduled stop
                min_distance = float('inf')
                for scheduled_loc in scheduled_locations:
                    dist = trip_loc.distance_to(scheduled_loc)
                    min_distance = min(min_distance, dist)

                # If not near any scheduled stop, it's unscheduled
                if min_distance > self.config.unscheduled_stop_threshold_meters:
                    # Determine the appropriate key prefix based on location type
                    prefix = 'end' if location_type == 'arrival' else 'start'

                    unscheduled_stops.append({
                        'type': location_type,
                        'location': asdict(trip_loc),
                        'address': trip[f'{prefix}Address'],
                        'time_ms': trip[f'{prefix}Ms'],
                        'nearest_scheduled_distance': min_distance
                    })

                    self.alerts.append(Alert(
                        type='unscheduled_stop',
                        severity='low',
                        location=trip_loc,
                        description=f"Unscheduled {location_type} at {trip[f'{prefix}Address']['name']}",
                        details={
                            'address': trip[f'{prefix}Address']['name'],
                            'nearest_scheduled_distance': min_distance
                        }
                    ))

        return unscheduled_stops

    def _find_gps_visits(self, stop_loc: Location, trips: List[Dict]) -> List[Dict]:
        """Find GPS visits within radius of a stop"""
        visits = []

        for trip in trips:
            # Check start location
            start_loc = Location(
                trip['startCoordinates']['latitude'],
                trip['startCoordinates']['longitude']
            )
            if stop_loc.distance_to(start_loc) <= self.config.alert_radius_meters:
                visits.append({
                    'type': 'departure',
                    'time_ms': trip['startMs'],
                    'address': trip['startAddress']['name'],
                    'distance_meters': stop_loc.distance_to(start_loc)
                })

            # Check end location
            end_loc = Location(
                trip['endCoordinates']['latitude'],
                trip['endCoordinates']['longitude']
            )
            if stop_loc.distance_to(end_loc) <= self.config.alert_radius_meters:
                visits.append({
                    'type': 'arrival',
                    'time_ms': trip['endMs'],
                    'address': trip['endAddress']['name'],
                    'distance_meters': stop_loc.distance_to(end_loc)
                })

        return visits

    def _find_nearby_orders(self, stop_loc: Location, orders_df: pd.DataFrame) -> List[Dict]:
        """Find orders within radius of a stop"""
        nearby_orders = []

        for _, order in orders_df.iterrows():
            order_loc = Location(order['Latitude'], order['Longitude'])
            dist = stop_loc.distance_to(order_loc)

            if dist <= self.config.alert_radius_meters:
                nearby_orders.append({
                    'customer_name': order['Customer Name'],
                    'start_time': order['Start Time'],
                    'end_time': order['End Time'],
                    'duration_minutes': order['Minutes'],
                    'user': order.get('User', 'Unknown'),
                    'distance_meters': dist
                })

        return nearby_orders


class MetadataGenerator:
    """Generates the metadata JSON output"""

    def __init__(self, config: RouteConfig):
        self.config = config

    def generate(self, route_info: pd.Series, stops_analysis: List[Dict],
                 trips_data: Dict, orders_df: pd.DataFrame,
                 unscheduled_stops: List[Dict], alerts: List[Alert],
                 warehouse_info: Dict) -> Dict:
        """Generate complete metadata JSON"""

        metadata = {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'route_info': {
                'route_number': self.config.route_number,
                'date': self.config.date_str,
                'vehicle_name': route_info['Vehicle Name'],
                'warehouse': warehouse_info
            },
            'summary': {
                'total_scheduled_stops': len(stops_analysis),
                'total_gps_trips': len(trips_data['trips']),
                'total_orders': len(orders_df),
                'total_alerts': len(alerts),
                'stops_visited': sum(1 for s in stops_analysis if s['has_gps_visit']),
                'stops_with_orders': sum(1 for s in stops_analysis if s['has_order']),
                'unscheduled_stops': len(unscheduled_stops)
            },
            'scheduled_stops': stops_analysis,
            'gps_trips': trips_data['trips'],
            'orders': orders_df.to_dict('records') if not orders_df.empty else [],
            'unscheduled_stops': unscheduled_stops,
            'alerts': [
                {
                    'type': alert.type,
                    'severity': alert.severity,
                    'location': asdict(alert.location),
                    'description': alert.description,
                    'details': alert.details
                }
                for alert in alerts
            ],
            'config': {
                'alert_radius_meters': self.config.alert_radius_meters,
                'unscheduled_stop_threshold_meters': self.config.unscheduled_stop_threshold_meters
            }
        }

        return metadata


class RouteProcessor:
    """Main orchestrator for route processing"""

    def __init__(self, config: RouteConfig):
        self.config = config
        self.loader = RouteDataLoader(config)
        self.analyzer = RouteAnalyzer(config)
        self.generator = MetadataGenerator(config)

    def process(self) -> Optional[Path]:
        """Process route data and generate metadata JSON"""
        try:
            logger.info(f"Starting processing for RT{self.config.route_number} on {self.config.date_str}")

            # Load route information
            routes_df = self.loader.load_routes()
            route_info = routes_df[routes_df['Route Number'] == self.config.route_number]

            if route_info.empty:
                logger.error(f"Route {self.config.route_number} not found")
                return None

            route_info = route_info.iloc[0]

            # Get warehouse location
            warehouse_loc = self._parse_location(route_info['Location'])
            warehouse_info = {
                'name': f"{route_info['Location']} Warehouse",
                'location': asdict(warehouse_loc)
            }

            # Load data
            stops_df = self.loader.get_scheduled_stops(self.config.date_str, self.config.route_number)
            trips_data = self.loader.load_trips(self.config.route_number, self.config.date_str)
            orders_df = self.loader.get_route_orders(self.config.date_str, self.config.route_number)

            # Analyze
            stops_analysis = self.analyzer.analyze_stop_visits(stops_df, trips_data, orders_df)
            unscheduled_stops = self.analyzer.find_unscheduled_stops(
                trips_data, stops_df, [warehouse_loc]
            )

            # Generate metadata
            metadata = self.generator.generate(
                route_info, stops_analysis, trips_data, orders_df,
                unscheduled_stops, self.analyzer.alerts, warehouse_info
            )

            # Save metadata
            output_filename = f"route_metadata_RT{self.config.route_number}_{self.config.date_str.replace(' ', '_')}.json"
            output_path = self.config.data_directory / 'metadata' / output_filename
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Metadata saved to {output_path}")
            logger.info(f"Summary: {metadata['summary']}")

            return output_path

        except Exception as e:
            logger.error(f"Error processing route: {e}", exc_info=True)
            return None

    def _parse_location(self, location_str: str) -> Location:
        """Parse location string (placeholder - would need actual warehouse coordinates)"""
        # This would need to be implemented based on your warehouse location data
        # For now, returning Austin warehouse coordinates from the sample
        warehouse_coords = {
            'Austin': Location(30.20037956, -97.71581548),
            # Add other warehouses as needed
        }
        return warehouse_coords.get(location_str, Location(0, 0))


def main():
    """Main entry point"""
    # Example usage
    config = RouteConfig(
        data_directory=Path("./data"),
        date_str="January 15, 2025",
        route_number=604
    )

    processor = RouteProcessor(config)
    output_path = processor.process()

    if output_path:
        print(f"\n✅ Successfully generated metadata: {output_path}")
    else:
        print("\n❌ Failed to generate metadata")


if __name__ == "__main__":
    main()
