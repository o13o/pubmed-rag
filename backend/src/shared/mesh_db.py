"""DuckDB-backed MeSH lookup for query expansion and terminology validation."""

import logging

import duckdb

logger = logging.getLogger(__name__)


class MeSHDatabase:
    def __init__(self, db_path: str = "data/mesh.duckdb"):
        self.conn = duckdb.connect(db_path)

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mesh_descriptors (
                descriptor_ui VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                tree_numbers VARCHAR[] NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mesh_synonyms (
                synonym VARCHAR NOT NULL,
                descriptor_ui VARCHAR NOT NULL REFERENCES mesh_descriptors(descriptor_ui)
            )
        """)

    def lookup(self, term: str) -> dict | None:
        row = self.conn.execute(
            "SELECT descriptor_ui, name, tree_numbers FROM mesh_descriptors WHERE name ILIKE ?",
            [term],
        ).fetchone()

        if row:
            return {"descriptor_ui": row[0], "name": row[1], "tree_numbers": row[2]}

        row = self.conn.execute(
            """SELECT d.descriptor_ui, d.name, d.tree_numbers
               FROM mesh_synonyms s
               JOIN mesh_descriptors d ON s.descriptor_ui = d.descriptor_ui
               WHERE s.synonym ILIKE ?""",
            [term],
        ).fetchone()

        if row:
            return {"descriptor_ui": row[0], "name": row[1], "tree_numbers": row[2]}

        return None

    def get_children(self, tree_number: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT DISTINCT d.descriptor_ui, d.name
               FROM mesh_descriptors d, unnest(d.tree_numbers) AS t(tn)
               WHERE tn LIKE ? AND tn != ?""",
            [f"{tree_number}.%", tree_number],
        ).fetchall()
        return [{"descriptor_ui": r[0], "name": r[1]} for r in rows]

    def get_synonyms(self, descriptor_ui: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT synonym FROM mesh_synonyms WHERE descriptor_ui = ?",
            [descriptor_ui],
        ).fetchall()
        return [r[0] for r in rows]

    def validate_term(self, term: str) -> bool:
        row = self.conn.execute(
            """SELECT EXISTS(SELECT 1 FROM mesh_descriptors WHERE name ILIKE ?)
               OR EXISTS(SELECT 1 FROM mesh_synonyms WHERE synonym ILIKE ?)""",
            [term, term],
        ).fetchone()
        return bool(row and row[0])

    def close(self) -> None:
        self.conn.close()
