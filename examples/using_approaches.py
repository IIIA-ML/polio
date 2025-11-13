#!/usr/bin/env python3
"""
Examples of using the new Approach-based interface.

This demonstrates the object-oriented refactoring of the detection methods.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from approaches import (
    ApproachFactory,
    CoRetweetsApproach,
    IgnoringTweetApproach,
    SharedTweetsApproach,
    SameTweetSameTimeApproach
)
from synchronous_repeated_detection import import_data


def example_1_using_factory():
    """Example 1: Create approaches using the factory."""
    print("="*60)
    print("Example 1: Using ApproachFactory")
    print("="*60)

    # Get all available approach keys
    print("\nAvailable approaches:", ApproachFactory.get_all_keys())

    # Get approach names mapping
    print("\nApproach names:", ApproachFactory.get_approach_names())

    # Create a specific approach
    approach = ApproachFactory.create('coretweets', window_sec=60, min_coactions=1)
    print(f"\nCreated approach: {approach}")
    print(f"  Name: {approach.get_approach_name()}")
    print(f"  Key: {approach.get_approach_key()}")

    # Create all approaches
    all_approaches = ApproachFactory.get_all_approaches(window_sec=60)
    print(f"\nCreated {len(all_approaches)} approaches:")
    for app in all_approaches:
        print(f"  - {app.get_approach_name()}")


def example_2_using_classes_directly():
    """Example 2: Instantiate approach classes directly."""
    print("\n" + "="*60)
    print("Example 2: Using classes directly")
    print("="*60)

    # Create instances directly
    coretweets = CoRetweetsApproach(window_sec=60, min_coactions=1)
    ignoring = IgnoringTweetApproach(window_sec=120, min_coactions=2)

    print(f"\nCreated: {coretweets}")
    print(f"  Metadata: {coretweets.get_metadata()}")

    print(f"\nCreated: {ignoring}")
    print(f"  Needs filtered data: {ignoring.needs_filtered_data()}")


def example_3_processing_data(data_path):
    """Example 3: Process actual data with approaches."""
    print("\n" + "="*60)
    print("Example 3: Processing data")
    print("="*60)

    try:
        # Load data
        RTs, io_users = import_data(data_path)
        print(f"\nLoaded {len(RTs)} retweets")

        # Create and use an approach
        approach = CoRetweetsApproach(window_sec=60, min_coactions=1)
        print(f"\nUsing approach: {approach.get_approach_name()}")

        pairs_scores = approach.compute_pairs_scores(RTs)
        print(f"Found {len(pairs_scores)} user pairs")

        # Show top 5 pairs
        if pairs_scores:
            sorted_pairs = sorted(pairs_scores.items(),
                                 key=lambda x: x[1], reverse=True)
            print("\nTop 5 pairs:")
            for pair, score in sorted_pairs[:5]:
                print(f"  {pair}: {score}")

    except Exception as e:
        print(f"\nCould not load data: {e}")
        print("(This is expected if data path doesn't exist)")


def example_4_polymorphism():
    """Example 4: Using approaches polymorphically."""
    print("\n" + "="*60)
    print("Example 4: Polymorphism")
    print("="*60)

    # Create different approaches
    approaches = [
        CoRetweetsApproach(window_sec=60),
        IgnoringTweetApproach(window_sec=60),
        SharedTweetsApproach(window_sec=60),
        SameTweetSameTimeApproach(window_sec=60),
    ]

    # Treat them all the same way
    print("\nProcessing with multiple approaches:")
    for approach in approaches:
        print(f"\n  Approach: {approach.get_approach_name()}")
        print(f"    Key: {approach.get_approach_key()}")
        print(f"    Needs filtering: {approach.needs_filtered_data()}")

        # Could call compute_pairs_scores(RTs) on any of them
        # The interface is uniform!


def example_5_extending():
    """Example 5: How to extend with a custom approach."""
    print("\n" + "="*60)
    print("Example 5: Extending with custom approach")
    print("="*60)

    from approaches import Approach

    class CustomApproach(Approach):
        """Example custom approach."""

        def get_approach_name(self):
            return "My Custom Approach"

        def get_approach_key(self):
            return "custom"

        def compute_pairs_scores(self, RTs, **kwargs):
            # Your custom logic here
            return {}

    # Register with factory
    ApproachFactory.register_approach('custom', CustomApproach)

    print("\nRegistered custom approach")
    print(f"Available approaches: {ApproachFactory.get_all_keys()}")

    # Now can create it via factory
    custom = ApproachFactory.create('custom', window_sec=30)
    print(f"Created: {custom}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("APPROACH-BASED INTERFACE EXAMPLES")
    print("="*60)

    example_1_using_factory()
    example_2_using_classes_directly()
    example_3_processing_data("../data/Armenia/Processed/")
    example_4_polymorphism()
    example_5_extending()

    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60 + "\n")
