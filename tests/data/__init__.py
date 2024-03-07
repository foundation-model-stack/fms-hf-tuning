"""Helpful datasets for configuring individual unit tests.
"""
# Standard
import os

### Constants used for data
DATA_DIR = os.path.join(os.path.dirname(__file__))
TWITTER_COMPLAINTS_DATA = os.path.join(DATA_DIR, "twitter_complaints_small.json")
