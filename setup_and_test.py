"""
Setup and Test Script for Route Processing System
Run this to verify your installation and test with sample data
"""

import sys
from pathlib import Path
import json


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")

    required_packages = {
        'pandas': 'pandas',
        'geopy': 'geopy'
    }

    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"✗ {package_name} NOT installed")
            missing.append(package_name)

    if missing:
        print(f"\nPlease install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True


def check_data_structure(data_dir: Path):
    """Verify the data directory structure"""
    print(f"\nChecking data directory structure at: {data_dir}")

    required_dirs = ['Dates', 'Orders', 'Routes', 'Stops', 'Trips']
    required_files = {
        'Dates/DateConversionsTable.csv': 'Date conversions table',
        'Routes/Routes.csv': 'Route definitions',
        'Stops/Stops  Copy.csv': 'Scheduled stops data'
    }

    all_good = True

    # Check directories
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory MISSING")
            all_good = False

    # Check files
    for file_path, description in required_files.items():
        full_path = data_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} MISSING ({description})")
            all_good = False

    # Create metadata directory if it doesn't exist
    metadata_dir = data_dir / 'metadata'
    if not metadata_dir.exists():
        metadata_dir.mkdir(exist_ok=True)
        print(f"✓ Created metadata/ directory")

    return all_good


def test_basic_processing():
    """Run a basic test of the processing system"""
    print("\nTesting basic processing...")

    try:
        from route_processor import RouteProcessor, RouteConfig
        from utils import analyze_metadata_file

        # Test configuration
        config = RouteConfig(
            data_directory=Path("./data"),
            date_str="January 15, 2025",  # Adjust to a date in your data
            route_number=604  # Adjust to a route in your data
        )

        print(f"Processing RT{config.route_number} for {config.date_str}...")

        processor = RouteProcessor(config)
        output_path = processor.process()

        if output_path and output_path.exists():
            print(f"✓ Successfully generated metadata: {output_path}")

            # Load and verify the output
            with open(output_path, 'r') as f:
                metadata = json.load(f)

            print(f"\nMetadata summary:")
            print(f"  - Route: {metadata['route_info']['route_number']}")
            print(f"  - Date: {metadata['route_info']['date']}")
            print(f"  - Scheduled stops: {metadata['summary']['total_scheduled_stops']}")
            print(f"  - GPS trips: {metadata['summary']['total_gps_trips']}")
            print(f"  - Alerts: {metadata['summary']['total_alerts']}")

            return True
        else:
            print("✗ Processing failed")
            return False

    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_sample_data():
    """Validate that sample data files have correct format"""
    print("\nValidating data file formats...")

    try:
        import pandas as pd

        # Check DateConversionsTable
        date_conv_path = Path("./data/Dates/DateConversionsTable.csv")
        if date_conv_path.exists():
            df = pd.read_csv(date_conv_path)
            required_cols = ['Date', 'Phase', 'Day Of Week']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"✗ DateConversionsTable missing columns: {missing}")
            else:
                print(f"✓ DateConversionsTable format OK ({len(df)} dates)")

        # Check Routes
        routes_path = Path("./data/Routes/Routes.csv")
        if routes_path.exists():
            df = pd.read_csv(routes_path)
            required_cols = ['Route Number', 'Vehicle Name', 'Location']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"✗ Routes.csv missing columns: {missing}")
            else:
                print(f"✓ Routes.csv format OK ({len(df)} routes)")

        # Check Stops
        stops_path = Path("./data/Stops/Stops  Copy.csv")
        if stops_path.exists():
            df = pd.read_csv(stops_path)
            required_cols = ['Stop ID', 'Route Num', 'Phase', 'Day Of Week',
                             'Sequence', 'Customer Name', 'Latitude', 'Longitude', 'Active']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"✗ Stops file missing columns: {missing}")
            else:
                print(f"✓ Stops file format OK ({len(df)} total stops)")
                active_stops = len(df[df['Active'] == 'Y'])
                print(f"  - Active stops: {active_stops}")
                routes_with_stops = df['Route Num'].nunique()
                print(f"  - Routes with stops: {routes_with_stops}")

        return True

    except Exception as e:
        print(f"✗ Error validating data: {e}")
        return False


def main():
    """Run all setup and test steps"""
    print("=" * 60)
    print("Route Processing System - Setup and Test")
    print("=" * 60)

    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended")

    # Run checks
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return

    data_dir = Path("./data")
    if not check_data_structure(data_dir):
        print("\n❌ Please fix data directory structure issues")
        return

    if not validate_sample_data():
        print("\n⚠️  Data validation issues found")

    # Test processing
    print("\n" + "=" * 60)
    if test_basic_processing():
        print("\n✅ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Review the generated metadata file in ./data/metadata/")
        print("2. Adjust route numbers and dates in the test to match your data")
        print("3. Use example_usage.py for more advanced processing examples")
    else:
        print("\n❌ Processing test failed. Please check:")
        print("1. The date exists in DateConversionsTable.csv")
        print("2. The route number exists in Routes.csv")
        print("3. There are active stops for the route/date combination")
        print("4. GPS trip files exist for the route/date")


if __name__ == "__main__":
    main()
