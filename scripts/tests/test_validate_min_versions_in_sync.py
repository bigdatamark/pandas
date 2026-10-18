# [MODULE DOC]
# Test for the script that pins minimum dependency versions in the conda environment file to
# match the project configuration file. Uses fixture data files in a data folder two levels up
# from this directory named by path constant.
import pathlib
import tomllib

import pytest
import yaml

from scripts.validate_min_versions_in_sync import (
    get_toml_map_from,
    get_yaml_map_from,
    pin_min_versions_to_yaml_file,
)

DATA_PATH = pathlib.Path(__file__).parents[2] / "scripts/tests/data/"


@pytest.mark.parametrize(
    "src_toml, src_yaml, expected_yaml",
    [
        (
            DATA_PATH / "deps_minimum.toml",
            DATA_PATH / "deps_unmodified_random.yaml",
            DATA_PATH / "deps_expected_random.yaml",
        ),
        (
            DATA_PATH / "deps_minimum.toml",
            DATA_PATH / "deps_unmodified_same_version.yaml",
            DATA_PATH / "deps_expected_same_version.yaml",
        ),
        (
            DATA_PATH / "deps_minimum.toml",
            DATA_PATH / "deps_unmodified_duplicate_package.yaml",
            DATA_PATH / "deps_expected_duplicate_package.yaml",
        ),
        (
            DATA_PATH / "deps_minimum.toml",
            DATA_PATH / "deps_unmodified_no_version.yaml",
            DATA_PATH / "deps_expected_no_version.yaml",
        ),
        (
            DATA_PATH / "deps_minimum.toml",
            DATA_PATH / "deps_unmodified_range.yaml",
            DATA_PATH / "deps_expected_range.yaml",
        ),
    ],
)
def test_pin_min_versions_to_yaml_file(src_toml, src_yaml, expected_yaml) -> None:
    # [FUNCTION DOC]
    # Verifies, for five scenarios (random, same version, duplicate package, no version, version
    # range), that reading a minimum-versions configuration file and an unmodified environment
    # file, mapping both, and pinning the environment text produces text exactly equal to the
    # expected environment file. Parametrized over the three file paths per scenario.
    with open(src_toml, "rb") as toml_f:
        toml_map = tomllib.load(toml_f)
    with open(src_yaml, encoding="utf-8") as yaml_f:
        yaml_file_data = yaml_f.read()
    yaml_file = yaml.safe_load(yaml_file_data)
    yaml_dependencies = yaml_file["dependencies"]
    yaml_map = get_yaml_map_from(yaml_dependencies)
    toml_map = get_toml_map_from(toml_map)
    result_yaml_file = pin_min_versions_to_yaml_file(yaml_map, toml_map, yaml_file_data)
    with open(expected_yaml, encoding="utf-8") as yaml_f:
        dummy_yaml_expected_file_1 = yaml_f.read()
    assert result_yaml_file == dummy_yaml_expected_file_1
