"""Tests for DuckDB-backed MeSH lookup."""

import pytest

from src.shared.mesh_db import MeSHDatabase


@pytest.fixture
def mesh_db():
    db = MeSHDatabase(":memory:")
    db._init_schema()
    db.conn.execute("""
        INSERT INTO mesh_descriptors VALUES
        ('D009369', 'Neoplasms', ['C04']),
        ('D002318', 'Cardiovascular Diseases', ['C14']),
        ('D003324', 'Coronary Artery Disease', ['C14.280.647.250']),
        ('D006333', 'Heart Failure', ['C14.280.434']),
        ('D020370', 'Osteoarthritis, Knee', ['C05.550.114.606'])
    """)
    db.conn.execute("""
        INSERT INTO mesh_synonyms VALUES
        ('Cancer', 'D009369'),
        ('Tumors', 'D009369'),
        ('Malignancy', 'D009369'),
        ('Heart Disease', 'D002318'),
        ('Knee Osteoarthritis', 'D020370'),
        ('Degenerative Arthritis of Knee', 'D020370')
    """)
    return db


def test_lookup_by_name(mesh_db):
    result = mesh_db.lookup("Neoplasms")
    assert result is not None
    assert result["descriptor_ui"] == "D009369"
    assert result["name"] == "Neoplasms"


def test_lookup_by_synonym(mesh_db):
    result = mesh_db.lookup("Cancer")
    assert result is not None
    assert result["name"] == "Neoplasms"


def test_lookup_case_insensitive(mesh_db):
    result = mesh_db.lookup("cancer")
    assert result is not None
    assert result["name"] == "Neoplasms"


def test_lookup_not_found(mesh_db):
    result = mesh_db.lookup("nonexistent_term")
    assert result is None


def test_get_children(mesh_db):
    children = mesh_db.get_children("C14")
    names = [c["name"] for c in children]
    assert "Coronary Artery Disease" in names
    assert "Heart Failure" in names
    assert "Cardiovascular Diseases" not in names


def test_get_synonyms(mesh_db):
    synonyms = mesh_db.get_synonyms("D009369")
    assert "Cancer" in synonyms
    assert "Tumors" in synonyms
    assert "Malignancy" in synonyms


def test_validate_term_exists(mesh_db):
    assert mesh_db.validate_term("Neoplasms") is True
    assert mesh_db.validate_term("Cancer") is True


def test_validate_term_not_exists(mesh_db):
    assert mesh_db.validate_term("FakeDrug123") is False
