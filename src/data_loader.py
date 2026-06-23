"""
Data loading utilities for the ISCX IDS 2012 XML dataset.
"""
import os
import glob
import re
import xml.etree.ElementTree as ET

import pandas as pd

_ILLEGAL_XML_CHARS = re.compile(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]')


def parse_iscx_xml(filepath: str) -> pd.DataFrame:
    """
    Parse a single ISCX IDS 2012 XML file.

    Handles:
      - Records as direct children of root (not wrapped in <flow>)
      - Malformed tokens / invalid XML characters
    """

    def _iter_records(root):
        """Yield dicts for every record element found."""
        children = list(root)
        if children and all(len(list(c)) > 0 for c in children[:5]):
            for record in children:
                yield {child.tag: child.text for child in record}
        else:
            for flow in root.iter("flow"):
                yield {child.tag: child.text for child in flow}

    # --- Attempt 1: standard parse ---
    try:
        tree = ET.parse(filepath)
        records = list(_iter_records(tree.getroot()))
        if records:
            return pd.DataFrame(records)
    except ET.ParseError:
        pass

    # --- Attempt 2: strip illegal XML characters line-by-line ---
    clean_lines = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            clean_lines.append(_ILLEGAL_XML_CHARS.sub("", line))
    content = "".join(clean_lines)

    # Fix bare & that aren't entity references
    content = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#)', '&amp;', content)

    root = ET.fromstring(content)
    records = list(_iter_records(root))
    return pd.DataFrame(records)


def load_all_xml(dataset_path: str) -> pd.DataFrame:
    """Find and parse every XML file under dataset_path, concatenating results."""
    xml_files = (
        glob.glob(os.path.join(dataset_path, "*.xml"))
        + glob.glob(os.path.join(dataset_path, "**", "*.xml"), recursive=True)
    )

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found in {dataset_path}.\n"
            "Check DATASET_PATH and that the archive is extracted."
        )

    print(f"Found {len(xml_files)} XML file(s):")
    dfs = []
    for fp in xml_files:
        fname = os.path.basename(fp)
        try:
            df = parse_iscx_xml(fp)
            print(f"  ✓  {fname:50s}  →  {len(df):>7,} flows")
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"  ✗  {fname:50s}  →  ERROR: {e}")

    if not dfs:
        raise ValueError("All files parsed to 0 rows. Check the XML structure.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal flows loaded: {len(combined):,}")
    print(f"Columns: {combined.columns.tolist()}")
    return combined
