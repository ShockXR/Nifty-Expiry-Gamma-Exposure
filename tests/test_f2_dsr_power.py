"""Tests for the post hoc DSR power calculation (synthetic numbers only)."""

import importlib.util
import math
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "f2_dsr_power", Path(__file__).resolve().parents[1] / "experiments" / "f2_dsr_power.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_cutoff_gives_psr_095():
    T, skew, kurt, sr0 = 389, 1.1, 24.0, 0.34
    c = m.pass_cutoff(T, skew, kurt, sr0)
    assert c > sr0
    assert abs(m.psr(c, T, skew, kurt, sr0) - 0.95) < 1e-9


def test_prob_pass_is_half_at_cutoff_and_monotone():
    T, skew, kurt, sr0 = 389, 0.0, 3.0, 0.2
    c = m.pass_cutoff(T, skew, kurt, sr0)
    assert abs(m.prob_pass(c, T, skew, kurt, c) - 0.5) < 1e-12
    ps = [m.prob_pass(s, T, skew, kurt, c) for s in np.linspace(-0.2, 0.6, 9)]
    assert all(b >= a for a, b in zip(ps, ps[1:]))


def test_gaussian_case_matches_closed_form():
    # skew 0, kurt 3, true SR 0: sd of the estimate is sqrt(1 / (T - 1))
    T, c = 101, 0.25
    p = m.prob_pass(0.0, T, 0.0, 3.0, c)
    assert abs(p - (1 - 0.5 * (1 + math.erf(c * math.sqrt(T - 1) / math.sqrt(2))))) < 1e-12


def test_sr_needed_inverts_prob_pass():
    T, skew, kurt, sr0 = 389, 1.1, 24.0, 0.34
    c = m.pass_cutoff(T, skew, kurt, sr0)
    s = m.sr_needed(0.8, T, skew, kurt, c)
    assert abs(m.prob_pass(s, T, skew, kurt, c) - 0.8) < 1e-9
