"""Test: long-run energy drift bound."""
from occupancy_gaussian_framework import scene_two_body_contact
import config as C

def test_conservation():
    # scene_two_body_contact prints drift; re-run logic inline would be cleaner
    # but for Phase 0 we just assert the scene gate passes.
    assert scene_two_body_contact()
