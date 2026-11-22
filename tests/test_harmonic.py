"""Test: harmonic oscillator matches analytic solution."""
from occupancy_gaussian_framework import scene_harmonic_oscillator

def test_harmonic():
    assert scene_harmonic_oscillator(), "harmonic oscillator failed gate G0-1"
