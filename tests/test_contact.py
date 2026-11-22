"""Test: two-body contact penetration and energy drift."""
from occupancy_gaussian_framework import scene_two_body_contact

def test_contact():
    assert scene_two_body_contact(), "two-body contact failed gate G0-2"
