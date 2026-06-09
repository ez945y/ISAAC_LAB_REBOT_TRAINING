from __future__ import annotations

import tomllib
from pathlib import Path

from setuptools import Distribution
from setuptools.config.pyprojecttoml import apply_configuration


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_discovers_controll_script_subpackages() -> None:
    dist = Distribution()
    apply_configuration(dist, str(ROOT / "pyproject.toml"))

    packages = set(dist.packages or [])

    assert "controll_scripts" in packages
    assert "controll_scripts.controllers" in packages
    assert "controll_scripts.configs" in packages
    assert "controll_scripts.safety" in packages
    assert "controll_scripts.so_arm_101" in packages


def test_pyproject_includes_safety_stackfile_data() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())

    package_data = data["tool"]["setuptools"]["package-data"]

    assert "*.yaml" in package_data["controll_scripts.safety"]
