#!/usr/bin/env python3
"""
scan_hinge_v3.py
Download P450 sequences from UniProt (PAGINATED, complete download) for multiple
plant species, plus an optional local FASTA (e.g., Sophora tonkinensis P450s),
then scan for two hinge motifs within configurable N-terminal windows:

  STRICT: [P]-x(2)-[G]-x(1)-[L]  (= P..G.L)
  CORE:   [P]-x(2)-[G]           (= P..G)

v3 changes vs v2:
  1. Cursor-based pagination -> no more size=500 truncation (fixes incomplete
     datasets for Glycine/Populus/Zea/Medicago/Vitis).
  2. Species list aligned with the manuscript Table 3
     (adds Selaginella moellendorffii, Amborella trichopoda; Sophora via local FASTA).
  3. Two N-terminal windows reported together (default: first <=100, then <=60).
  4. ONE unified confidence function used in BOTH the summary table and the
     interpretation section (fixes the v2 inconsistency between sections).
  5. Download date recorded in the report header and in every CSV file.

Usage:
    python scan_hinge_v3.py [--windows 100 60] [--fasta Sophora_P450s.fasta]
                            [--output-dir hinge_scan_v3] [--search-range]
NOTE: match positions are reported 0-based (as in v2). Add +1 for 1-based
      coordinates in the manuscript.
"""

import sys
import re
import time
import csv
import argparse
from pathlib import Path
import urllib.request
import urllib.parse
import numpy as np
from Bio import SeqIO
from io import StringIO

# ── Motifs ─────────────────────────────────────────────────────
MOTIF_STRICT = re.compile(r'P..G.L')   # [P]-x(2)-[G]-x(1)-[L]
MOTIF_CORE   = re.compile(r'P..G')     # [P]-x(2)-[G]

# ── Species (taxids from NCBI; Sophora loaded from local FASTA) ─
SPECIES = {
    "Physcomitrium patens":      3218,
    "Selaginella moellendorffii": 88036,
    "Amborella trichopoda":       13333,
    "Oryza sativa (rice)":        39947,
    "Zea mays (maize)":           4577,
    "Arabidopsis thaliana":       3702,
    "Populus trichocarpa":        3694,
    "Solanum lycopersicum":       4081,
}


# ── Paginated UniProt download ────────────────────────────────
def _get_next_cursor(headers):
    """Extract the next-page cursor from the Link response header."""
    link = headers.get("Link", "")
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    if not m:
        return None
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(m.group(1)).query)
    return qs.get("cursor", [None])[0]


def fetch_all_fasta(query, size=500, max_pages=30, timeout=120):
    """Fetch ALL matching sequences via cursor-based pagination.
    Returns a list of SeqRecord objects (empty if the query fails)."""
    records = []
    cursor = None
    for _ in range(max_pages):
        params = {"query": query, "format": "fasta", "size": str(size)}
        if cursor:
            params["cursor"] = cursor
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8")
                next_cursor = _get_next_cursor(resp.headers)
        except Exception as e:
            print(f"      [warn] download page failed: {e}")
            break
        page = list(SeqIO.parse(StringIO(data), "fasta"))
        records.extend(page)
        if not next_cursor or not page:
            break
        cursor = next_cursor
        time.sleep(0.3)   # be polite to the UniProt server
    return records


def download_p450(taxid):
    """Query UniProt with three fallback strategies; return the FIRST
    strategy that yields sequences (kept identical to v2 for interpretability)."""
    queries = [
        f'(taxonomy_id:{taxid}) AND (family:"cytochrome p450")',
        f'(taxonomy_id:{taxid}) AND (ec:1.14.-.-)',
        f'(taxonomy_id:{taxid}) AND (gene:CYP*)',
    ]
    for q in queries:
        records = fetch_all_fasta(q)
        if records:
            return records
    return []


def load_local_fasta(path):
    """Load a local FASTA (e.g., Sophora tonkinensis P450s)."""
    return list(SeqIO.parse(path, "fasta"))


# ── Unified confidence (ONE function for all sections) ────────
def confidence_level(core_pct, median_pos, pct_below_50):
    """Single set of thresholds used in both the summary and the
    interpretation. Criterion: high N-terminal prevalence + median match
    position near the N-terminus + most matches within 50 aa."""
    if (core_pct >= 70 and median_pos is not None
            and median_pos <= 40 and pct_below_50 >= 80):
        return "HIGH confidence N-terminal hinge"
    if (core_pct >= 50 and median_pos is not None
            and median_pos <= 55 and pct_below_50 >= 50):
        return "MODERATE confidence"
    return "LOW confidence - may not be true N-terminal PxxG"


# ── Scan one species over multiple windows ────────────────────
def scan_species(records, label, windows, output_matches):
    """Scan all sequences for STRICT and CORE motifs within each window.
    Returns a dict keyed by window size with per-window statistics."""
    total = len(records)
    results = {}

    for win in windows:
        strict_positions = []
        core_positions = []
        for rec in records:
            seq = str(rec.seq)[:win]

            ms = MOTIF_STRICT.search(seq)
            if ms:
                strict_positions.append(ms.start())
                output_matches["strict"].append({
                    "species": label, "id": rec.id, "pos": ms.start(),
                    "match": ms.group(),
                    "ctx": str(rec.seq)[max(0, ms.start()-3):ms.end()+3],
                    "window": win})
            mc = MOTIF_CORE.search(seq)
            if mc:
                core_positions.append(mc.start())
                output_matches["core"].append({
                    "species": label, "id": rec.id, "pos": mc.start(),
                    "match": mc.group(),
                    "ctx": str(rec.seq)[max(0, mc.start()-3):mc.end()+3],
                    "window": win})

        def stats(positions):
            if not positions:
                return {"n": 0, "min": None, "max": None, "median": None,
                        "pct_below_50": 0.0}
            arr = np.array(positions)
            n_below = int(np.sum(arr <= 50))
            return {"n": len(arr),
                    "min": int(np.min(arr)), "max": int(np.max(arr)),
                    "median": int(np.median(arr)),
                    "pct_below_50": round(100.0 * n_below / len(arr), 1)}

        s_strict = stats(strict_positions)
        s_core = stats(core_positions)
        pct_core = 100.0 * s_core["n"] / total if total else 0.0
        pct_strict = 100.0 * s_strict["n"] / total if total else 0.0

        results[win] = {
            "label": label, "total": total,
            "strict": s_strict, "core": s_core,
            "pct_core": round(pct_core, 1),
            "pct_strict": round(pct_strict, 1),
            "quality": confidence_level(pct_core, s_core["median"],
                                        s_core["pct_below_50"]),
        }
    return results


# ── Report printing ────────────────────────────────────────────
def print_window_table(window, results):
    print(f"\n  WINDOW <= {window} aa")
    header = (f"{'Species':<28} {'N':>5}  {'Core%':>7}  {'Strict%':>8}  "
              f"{'Core med pos':>13}  {'%<=50':>6}  {'Quality':>28}")
    print("  " + header)
    print("  " + "-" * len(header))
    for label in results:
        r = results[label][window]
        med = r["core"]["median"]
        med_str = str(med) if med is not None else "N/A"
        p50 = r["core"]["pct_below_50"]
        print(f"  {r['label']:<28} {r['total']:>5}  {r['pct_core']:>6.0f}%  "
              f"{r['pct_strict']:>7.0f}%  {med_str:>13}  {p50:>5.0f}%  "
              f"{r['quality']:>28}")


def print_interpretation(window, results):
    print(f"\n  INTERPRETATION (window <= {window}):")
    for label in results:
        r = results[label][window]
        med = r["core"]["median"]
        med_str = str(med) if med is not None else "N/A"
        print(f"    {r['label']:<28} Core={r['pct_core']:.0f}%  "
              f"median_pos={med_str}  %<=50={r['core']['pct_below_50']:.0f}%  "
              f"-> {r['quality']}")


# ── CSV export ────────────────────────────────────────────────
def export_csv(out_dir, tag, rows, download_date):
    path = out_dir / f"hinge_scan_{tag}_positions.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["species", "id", "pos",
                                               "match", "ctx", "window"])
        writer.writeheader()
        writer.writerows(rows)
    # append the download date as a trailing comment line
    with open(path, "a", newline="") as f:
        f.write(f"# download_date={download_date}\n")
    print(f"  Saved {path.name} ({len(rows)} matches)")


def main():
    p = argparse.ArgumentParser(
        description="Scan P450 hinge motifs across species (v3, paginated)")
    p.add_argument("--windows", nargs="+", type=int, default=[100, 60],
                   help="N-terminal search windows, largest first "
                        "(default: 100 60)")
    p.add_argument("--fasta", default=None,
                   help="Local FASTA with Sophora tonkinensis P450s "
                        "(loaded as 'Sophora tonkinensis')")
    p.add_argument("--output-dir", default="hinge_scan_v3",
                   help="Output directory for CSV files")
    args = p.parse_args()

    windows = sorted(set(args.windows), reverse=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datetime import date
    download_date = date.today().isoformat()

    results = {}
    output_matches = {"core": [], "strict": []}

    print(f"Hinge scan v3 | download date: {download_date}")
    print(f"Windows: {windows}")

    # 1) UniProt species
    for name, taxid in SPECIES.items():
        print(f"  Downloading {name} (taxid={taxid}) ... ", end="", flush=True)
        records = download_p450(taxid)
        if records:
            print(f"{len(records)} sequences")
            results[name] = scan_species(records, name, windows, output_matches)
        else:
            print("FAILED (skipped)")
        time.sleep(0.5)

    # 2) Local Sophora data (if provided)
    if args.fasta:
        print(f"  Loading local FASTA: {args.fasta} ... ", end="", flush=True)
        try:
            records = load_local_fasta(args.fasta)
            print(f"{len(records)} sequences")
            results["Sophora tonkinensis"] = scan_species(
                records, "Sophora tonkinensis", windows, output_matches)
        except Exception as e:
            print(f"FAILED: {e} (skipped)")
    else:
        print("  [info] no --fasta provided; Sophora tonkinensis skipped. "
              "Add --fasta Sophora_P450s.fasta to include it.")

    if not results:
        print("\nNo data. Exiting.")
        sys.exit(1)

    # 3) Report
    print("\n" + "=" * 110)
    print(f"SUMMARY: Hinge motif prevalence + position verification "
          f"(download date: {download_date})")
    print("=" * 110)
    for win in windows:
        print_window_table(win, results)

    print("\n" + "=" * 110)
    print("INTERPRETATION (same confidence function for all sections)")
    print("=" * 110)
    for win in windows:
        print_interpretation(win, results)

    # 4) CSV export (one file per window)
    for win in windows:
        core_rows = [r for r in output_matches["core"] if r["window"] == win]
        strict_rows = [r for r in output_matches["strict"] if r["window"] == win]
        if core_rows:
            export_csv(out_dir, f"core_W{win}", core_rows, download_date)
        if strict_rows:
            export_csv(out_dir, f"strict_W{win}", strict_rows, download_date)

    print("\nNOTE: 'PxxG' positions are 0-based. Add +1 for 1-based "
          "coordinates in the manuscript.")
    print("NOTE: check the per-species 'N' column - if UniProt annotation is "
          "incomplete for a species, percentages should be interpreted with "
          "caution.")


if __name__ == "__main__":
    main()
