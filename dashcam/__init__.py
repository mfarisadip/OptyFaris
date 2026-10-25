"""
Dashcam Package - Road Damage Detection System
Ultra-low latency detection with GPS and API integration
"""

__version__ = "2.0.0"
__author__ = "Road AI Team"

# Package-level imports for convenience
from dashcam.core.application import DashcamApp
from dashcam.config.settings import parse_arguments

__all__ = ['DashcamApp', 'parse_arguments']
