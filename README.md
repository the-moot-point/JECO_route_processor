# Route Processing System

A modular Python system for processing delivery route data and generating metadata for interactive map visualization.

## Overview

This system processes route data from multiple sources (scheduled stops, GPS trips, order entries) and generates comprehensive metadata JSON files that can be consumed by web-based visualization tools.

## Directory Structure

```
data/
├── Dates/
│   └── DateConversionsTable.csv    # Maps dates to phases and days of week
├── Orders/
│   └── Orders [Month Year].csv     # Monthly order entry data
├── Routes/
│   └── Routes.csv                  # Route definitions and warehouse locations
├── Stops/
│   └── Stops.csv             # All scheduled stops for all routes
├── Trips/
│   └── Trips RT [num] [Mon YY].json # GPS tracking data
└── metadata/
    └── route_metadata_*.json       # Generated output files
```

## Installation

Required Python packages:
```bash
pip install pandas geopy
```

## Usage

### Basic Usage

```python
from pathlib import Path
from jeco_route_processor import RouteProcessor, RouteConfig

# Configure processing
config = RouteConfig(
    data_directory=Path("./data"),
    date_str="January 15, 2025",
    route_number=604,
    alert_radius_meters=100.0,
    unscheduled_stop_threshold_meters=150.0
)

# Process route
processor = RouteProcessor(config)
output_path = processor.process()
```

### Custom Alert Thresholds

For rural routes or areas with less precise GPS:
```python
config = RouteConfig(
    data_directory=Path("./data"),
    date_str="January 15, 2025",
    route_number=600,
    alert_radius_meters=200.0,  # More lenient 200m radius
    unscheduled_stop_threshold_meters=300.0
)
```

### Batch Processing

Process multiple routes and dates:
```python
from jeco_route_processor import run_batch_processing

routes = [600, 601, 602, 603, 604]
dates = ["January 13, 2025", "January 14, 2025", "January 15, 2025"]

results = run_batch_processing(Path("./data"), routes, dates)
```

## Alert Types

The system generates three types of alerts:

1. **Missed Stop** (High Severity)
   - A scheduled stop with no GPS visit within the alert radius
   - Indicates driver may have skipped the stop

2. **No Order** (Medium Severity)
   - A scheduled stop with no order entry within the alert radius
   - Indicates possible data entry issue

3. **Unscheduled Stop** (Low Severity)
   - GPS stop at location not near any scheduled stop
   - Could indicate additional delivery or break

## Output Format

The generated metadata JSON includes:

```json
{
  "version": "1.0",
  "generated_at": "ISO timestamp",
  "route_info": {
    "route_number": 604,
    "date": "January 15, 2025",
    "vehicle_name": "Truck 604",
    "warehouse": { /* location data */ }
  },
  "summary": {
    "total_scheduled_stops": 12,
    "stops_visited": 10,
    "stops_with_orders": 10,
    "total_alerts": 3,
    "unscheduled_stops": 1
  },
  "scheduled_stops": [ /* detailed stop analysis */ ],
  "gps_trips": [ /* raw GPS data */ ],
  "orders": [ /* order entries */ ],
  "unscheduled_stops": [ /* unexpected stops */ ],
  "alerts": [ /* all alerts with details */ ]
}
```

## Performance Metrics

Use the utilities module to calculate route efficiency:

```python
from jeco_route_processor import RouteStatistics, analyze_metadata_file

# Analyze a single route
analyze_metadata_file(Path("./data/metadata/route_metadata_RT604_January_15_2025.json"))

# Calculate detailed statistics
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
stats = RouteStatistics.calculate_route_efficiency(metadata)
```

## Data Validation

The system includes validation checks:

```python
from jeco_route_processor import DataValidator

# Validate stops data
warnings = DataValidator.validate_stops_data(stops_df)

# Validate trips data
warnings = DataValidator.validate_trips_data(trips_data)
```

## Adding New Warehouses

```python
from jeco_route_processor import WarehouseLocations

WarehouseLocations.add_warehouse(
    location_name="Dallas",
    name="Dallas Warehouse (DAL)",
    lat=32.7767,
    lon=-96.7970,
    address="123 Main St, Dallas, TX 75201"
)
```

## Error Handling

The system includes comprehensive logging:
- INFO: Normal processing steps
- WARNING: Data quality issues that don't stop processing
- ERROR: Critical issues that prevent processing

## Next Steps

The generated metadata JSON files are designed to be consumed by:
- Interactive web-based map visualizations
- Reporting and analytics tools
- Route optimization systems
- Performance dashboards

## Troubleshooting

Common issues:

1. **Missing date in DateConversionsTable**
   - Ensure the date exists in the conversion table
   - Check date format matches exactly

2. **No trips file found**
   - Verify filename format: "Trips RT 604 Jan 25.json"
   - Check month abbreviation and year format

3. **Empty scheduled stops**
   - Verify Phase and Day of Week combination exists
   - Check route number is correct
   - Ensure stops are marked as active ('Y' or 'Active')
   - Confirm phase names ("One", "Two", etc.) match the numeric notation
     ("1 Only", "2 Only", ...) used in `Stops.csv`
