"""Shared test fixtures."""

import os

import pytest

from src.shared.mesh_db import MeSHDatabase


@pytest.fixture
def mesh_db():
    db = MeSHDatabase(":memory:")
    db._init_schema()
    db.conn.execute("""
        INSERT INTO mesh_descriptors VALUES
        ('D009369', 'Neoplasms', ['C04']),
        ('D001943', 'Breast Neoplasms', ['C04.588.180']),
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
        ('Breast Cancer', 'D001943'),
        ('Heart Disease', 'D002318'),
        ('Knee Osteoarthritis', 'D020370'),
        ('Knee Pain', 'D020370'),
        ('Degenerative Arthritis of Knee', 'D020370')
    """)
    yield db
    db.close()


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
