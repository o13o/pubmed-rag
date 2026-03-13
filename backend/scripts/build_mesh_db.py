"""Parse NLM MeSH XML (desc2025.xml) and build DuckDB database.

Usage:
    uv run python scripts/build_mesh_db.py --input data/desc2025.xml --output data/mesh.duckdb
"""

import argparse
import logging
from pathlib import Path
from xml.etree import ElementTree as ET

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_mesh_xml(xml_path: Path) -> tuple[list[tuple], list[tuple]]:
    logger.info("Parsing %s ...", xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    descriptors = []
    synonyms = []

    for record in root.findall("DescriptorRecord"):
        ui = record.findtext("DescriptorUI", default="")
        name = record.findtext("DescriptorName/String", default="")
        tree_numbers = [tn.text for tn in record.findall("TreeNumberList/TreeNumber") if tn.text]

        if ui and name:
            descriptors.append((ui, name, tree_numbers))

        for concept in record.findall("ConceptList/Concept"):
            for term in concept.findall("TermList/Term"):
                term_text = term.findtext("String", default="")
                if term_text and term_text != name:
                    synonyms.append((term_text, ui))

    logger.info("Parsed %d descriptors, %d synonyms", len(descriptors), len(synonyms))
    return descriptors, synonyms


def build_duckdb(descriptors: list[tuple], synonyms: list[tuple], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    conn = duckdb.connect(str(output_path))

    conn.execute("""
        CREATE TABLE mesh_descriptors (
            descriptor_ui VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            tree_numbers VARCHAR[] NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE mesh_synonyms (
            synonym VARCHAR NOT NULL,
            descriptor_ui VARCHAR NOT NULL REFERENCES mesh_descriptors(descriptor_ui)
        )
    """)

    conn.executemany("INSERT INTO mesh_descriptors VALUES (?, ?, ?)", descriptors)
    conn.executemany("INSERT INTO mesh_synonyms VALUES (?, ?)", synonyms)

    conn.execute("CREATE INDEX idx_desc_name ON mesh_descriptors(name)")
    conn.execute("CREATE INDEX idx_syn_text ON mesh_synonyms(synonym)")
    conn.execute("CREATE INDEX idx_syn_ui ON mesh_synonyms(descriptor_ui)")

    count_d = conn.execute("SELECT COUNT(*) FROM mesh_descriptors").fetchone()[0]
    count_s = conn.execute("SELECT COUNT(*) FROM mesh_synonyms").fetchone()[0]
    logger.info("Built DuckDB: %d descriptors, %d synonyms -> %s", count_d, count_s, output_path)

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Build MeSH DuckDB from XML")
    parser.add_argument("--input", required=True, help="Path to MeSH XML")
    parser.add_argument("--output", default="data/mesh.duckdb", help="Output DuckDB path")
    args = parser.parse_args()

    descriptors, synonyms = parse_mesh_xml(Path(args.input))
    build_duckdb(descriptors, synonyms, Path(args.output))


if __name__ == "__main__":
    main()
