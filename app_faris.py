#!/usr/bin/env python3
"""
Ultra-Low Latency Dashcam - Modular Entry Point
Road damage detection with GPS and API integration
"""

from dashcam import DashcamApp, parse_arguments


def main():
    """Main entry point"""
    args = parse_arguments()
    app = DashcamApp(args)
    app.run()


if __name__ == "__main__":
    main()