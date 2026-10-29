"""Verified Crossref reference metadata and a conservative method relevance inventory."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

from uclada.common import write_json
from uclada.data import fetch

DOIS = [
    ("10.1002/9781394188536.ch2", "Recent LA foundations; not a spectrum-specific competitor"),
    ("10.1016/j.ins.2023.119487", "Modern hierarchical LA and action ordering; different LA variant"),
    ("10.3390/fi17120555", "LA-based LoRaWAN channel selection; incumbent model differs"),
    ("10.6028/NIST.TN.2240", "RF measurement provenance; not allocation outcomes"),
    ("10.1109/INFOCOM53939.2023.10228867", "Swept PSD classification dataset; timing mismatch"),
    (
        "10.1016/j.phycom.2023.102054",
        "Distributed overlay DSA / recurrent multiagent RL; close conceptual comparator",
    ),
    ("10.1016/j.icte.2024.01.001", "Distributed sensing-threshold control; different action space"),
    ("10.3390/s25072025", "Attention/MADRL opportunistic access; independent DQN is not a reproduction"),
    ("10.1109/ACCESS.2025.3534657", "Uncoordinated DQL; underlay assumptions and guarantee do not transfer"),
    ("10.1109/TCOMM.2025.3534565", "Multicell joint power/channel allocation; public code but task mismatch"),
    (
        "10.1016/j.compeleceng.2025.110147",
        "Joint channel/power control; adaptation would require changed task",
    ),
    (
        "10.1109/JSAC.2026.3659811",
        "STAR heterogeneous decentralized sharing; richer observation/action model",
    ),
    ("10.1016/j.sciaf.2024.e02523", "DRL dynamic spectrum access in television whitespace"),
    (
        "10.1137/24M167651X",
        "Adaptive smooth nonstationary bandits; theory assumptions differ from collisions",
    ),
    ("10.1109/TSMC.1974.5408453", "Learning automata survey / foundational probability updates"),
    ("10.1007/BF00992698", "Tabular Q-learning foundation"),
    ("10.1038/nature14236", "DQN with replay and target network; our implementation is adapted"),
    ("10.1023/A:1013689704352", "Finite-time UCB; stationary assumptions do not transfer"),
    ("10.1137/S0097539701398375", "EXP3 adversarial bandits; bounded feedback mapping"),
    ("10.1093/biomet/25.3-4.285", "Original Thompson sampling"),
    ("10.1109/TIT.2006.872873", "Optimal multichannel opportunistic access under partial observations"),
    ("10.1109/JSAC.2005.843904", "Cognitive radio foundations"),
    ("10.1109/COMST.2012.031212.00097", "Spectrum sensing algorithm survey"),
    ("10.1109/JSTSP.2007.914879", "SNR walls / noise uncertainty in sensing"),
    ("10.1109/FOCS.2013.30", "Bandits with knapsacks; resource constraints vs forecast threshold"),
    ("10.1214/aos/1176344552", "Bootstrap foundations"),
    ("10.2307/4615733", "Holm sequential multiple-testing correction"),
    ("10.1214/09-SS054", "Cross-validation model-selection methodology"),
    ("10.1145/2382577.2382579", "Leakage in data mining; general methodological reference"),
    ("10.1016/j.patter.2023.100804", "Scientific data leakage and reproducibility"),
]


def verify_references(root: Path) -> dict[str, Any]:
    folder = root / "data/raw/literature"
    folder.mkdir(parents=True, exist_ok=True)

    def one(item: tuple[str, str]) -> dict[str, Any]:
        doi, relevance = item
        path = folder / (re.sub(r"[^a-zA-Z0-9]", "_", doi) + ".json")
        if not path.exists():
            receipt = fetch("https://api.crossref.org/works/" + quote(doi, safe=""), path, timeout=40)
            if not receipt["success"]:
                return {"doi": doi, "verified": False, "error": receipt.get("error"), "relevance": relevance}
        try:
            metadata = json.loads(path.read_text())["message"]
            title = metadata.get("title", [""])[0]
            year = metadata.get("published", metadata.get("issued", {})).get("date-parts", [[None]])[0][0]
            authors = [
                " ".join([a.get("given", ""), a.get("family", "")]).strip()
                for a in metadata.get("author", [])
            ]
            return {
                "doi": doi,
                "verified": True,
                "title": title,
                "year": year,
                "authors": "; ".join(authors),
                "venue": metadata.get("container-title", [""])[0],
                "url": metadata.get("URL"),
                "type": metadata.get("type"),
                "relevance": relevance,
                "code_status": "No original-code reproduction claimed",
                "metadata_source": "https://api.crossref.org/works/" + doi,
                "include_bibtex": not relevance.startswith("CANDIDATE"),
            }
        except (ValueError, KeyError, IndexError) as exc:
            return {"doi": doi, "verified": False, "error": str(exc), "relevance": relevance}

    with ThreadPoolExecutor(5) as pool:
        rows = list(pool.map(one, DOIS))
    frame = pd.DataFrame(rows)
    (root / "data/manifests").mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "data/manifests/verified_references.csv", index=False)
    entries = []
    for i, r in enumerate(rows):
        if not r.get("verified") or not r.get("include_bibtex"):
            continue

        def clean(value):
            return str(value).replace("&", r"\&").replace("{", "").replace("}", "")

        entries.append(
            "@article{ref%02d,\n  title={%s},\n  author={%s},\n  year={%s},\n  journal={%s},\n  doi={%s},\n  url={%s}\n}"
            % (
                i + 1,
                clean(r["title"]),
                clean(r["authors"].replace("; ", " and ")),
                r["year"],
                clean(r["venue"]),
                r["doi"],
                r["url"],
            )
        )
    (root / "references.bib").write_text("\n\n".join(entries) + "\n")
    text = [
        "# Related work source audit (2022–2026 and foundations)",
        "",
        "Metadata below was fetched from Crossref's registered DOI records. Metadata verification does not",
        "constitute an equation-level reproduction or a completed full-text systematic review.",
        "",
        "| Year | Verified title | DOI | Relation to this project |",
        "|---|---|---|---|",
    ]
    for r in rows:
        if r.get("verified"):
            text.append(
                f"| {r['year']} | {r['title']} | [{r['doi']}](https://doi.org/{r['doi']}) | {r['relevance']} |"
            )
    text.extend(
        [
            "",
            "No cited paper's published numbers are used in head-to-head tables. The implemented DQN",
            "is an independent local-history common-simulator baseline. Exact reproduction remains unclaimed.",
            "The 30–45 reference target and equation-level novelty search remain open if this inventory is shorter.",
        ]
    )
    (root / "docs/related_work_2022_2026.md").write_text("\n".join(text) + "\n")
    report = {
        "candidates": len(rows),
        "verified": sum(bool(r["verified"]) for r in rows),
        "bibtex_entries": len(entries),
        "full_text_novelty_certification": False,
    }
    write_json(root / "reports/literature_audit.json", report)
    return report
