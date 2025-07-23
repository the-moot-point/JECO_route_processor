"""
Route Processing System - Example Usage
Shows different ways to use the route processor
"""

from pathlib import Path
from jeco_route_processor import (
    RouteProcessor,
    RouteConfig,
    DateFormatter,
    analyze_metadata_file,
    run_batch_processing,
)
import json


def example_single_route():
    """Process a single route for a single date"""
    print("=" * 60)
    print("Example 1: Processing single route")
    print("=" * 60)

    config = RouteConfig(
        data_directory=Path("./data"),
        date_str="January 15, 2025",
        route_number=604,
        alert_radius_meters=100.0,  # 100m radius for matching
        unscheduled_stop_threshold_meters=150.0  # 150m to be considered unscheduled
    )

    processor = RouteProcessor(config)
    output_path = processor.process()

    if output_path:
        print(f"\n✅ Successfully generated: {output_path}")
        # Analyze the results
        analyze_metadata_file(output_path)
    else:
        print("\n❌ Processing failed")


def example_custom_thresholds():
    """Process with custom alert thresholds"""
    print("\n" + "=" * 60)
    print("Example 2: Custom alert thresholds")
    print("=" * 60)

    # More lenient thresholds for rural routes
    config = RouteConfig(
        data_directory=Path("./data"),
        date_str="January 15, 2025",
        route_number=600,
        alert_radius_meters=200.0,  # 200m radius (more lenient)
        unscheduled_stop_threshold_meters=300.0  # 300m for unscheduled stops
    )

    processor = RouteProcessor(config)
    output_path = processor.process()

    if output_path:
        print(f"\n✅ Generated with custom thresholds: {output_path}")


def example_batch_processing():
    """Process multiple routes and dates"""
    print("\n" + "=" * 60)
    print("Example 3: Batch processing")
    print("=" * 60)

    data_dir = Path("./data")

    # Process a week of data for multiple routes
    routes = [600, 601, 602, 603, 604]
    dates = [
        "January 13, 2025",  # Monday
        "January 14, 2025",  # Tuesday
        "January 15, 2025",  # Wednesday
        "January 16, 2025",  # Thursday
        "January 17, 2025",  # Friday
    ]

    results = run_batch_processing(data_dir, routes, dates)

    # Summary report
    print("\n📊 Batch Processing Summary:")
    successful = sum(1 for r in results if r['success'])
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")


def example_analyze_existing():
    """Analyze an existing metadata file"""
    print("\n" + "=" * 60)
    print("Example 4: Analyze existing metadata")
    print("=" * 60)

    metadata_path = Path("./data/metadata/route_metadata_RT604_January_15_2025.json")

    if metadata_path.exists():
        analyze_metadata_file(metadata_path)
    else:
        print(f"Metadata file not found: {metadata_path}")


def example_generate_summary_report():
    """Generate a summary report for multiple routes"""
    print("\n" + "=" * 60)
    print("Example 5: Generate summary report")
    print("=" * 60)

    metadata_dir = Path("./data/metadata")

    if not metadata_dir.exists():
        print("No metadata directory found")
        return

    # Collect all metadata files
    all_stats = []

    for metadata_file in metadata_dir.glob("*.json"):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        stats = {
            'route': metadata['route_info']['route_number'],
            'date': metadata['route_info']['date'],
            'stops_scheduled': metadata['summary']['total_scheduled_stops'],
            'stops_visited': metadata['summary']['stops_visited'],
            'completion_rate': metadata['summary']['stops_visited'] / metadata['summary'][
                'total_scheduled_stops'] * 100,
            'alerts': metadata['summary']['total_alerts'],
            'unscheduled_stops': metadata['summary']['unscheduled_stops']
        }
        all_stats.append(stats)

    # Sort by route and date
    all_stats.sort(key=lambda x: (x['route'], x['date']))

    # Print summary table
    print("\n📊 Route Performance Summary")
    print("-" * 80)
    print(
        f"{'Route':<8} {'Date':<20} {'Scheduled':<10} {'Visited':<10} {'Completion':<12} {'Alerts':<8} {'Unscheduled':<12}")
    print("-" * 80)

    for stats in all_stats:
        print(f"{stats['route']:<8} {stats['date']:<20} {stats['stops_scheduled']:<10} "
              f"{stats['stops_visited']:<10} {stats['completion_rate']:<11.1f}% "
              f"{stats['alerts']:<8} {stats['unscheduled_stops']:<12}")

    # Calculate averages
    avg_completion = sum(s['completion_rate'] for s in all_stats) / len(all_stats)
    total_alerts = sum(s['alerts'] for s in all_stats)

    print("-" * 80)
    print(f"\nAverage completion rate: {avg_completion:.1f}%")
    print(f"Total alerts across all routes: {total_alerts}")


def main():
    """Run all examples"""
    print("\n🚚 Route Processing System - Examples")
    print("=====================================\n")

    # Run examples
    example_single_route()
    # example_custom_thresholds()
    # example_batch_processing()
    # example_analyze_existing()
    # example_generate_summary_report()

    print("\n✅ Examples complete!")


if __name__ == "__main__":
    main()
