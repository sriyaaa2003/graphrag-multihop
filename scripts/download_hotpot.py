"""Download the HotpotQA distractor validation split (Hugging Face hub parquet export).

Requires pyarrow only for reading it later (`pip install pyarrow`). Output: data/hotpot_validation.parquet
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    "refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet"
)


def main(out: str = "data/hotpot_validation.parquet") -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, out)  # noqa: S310 (fixed https URL)
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:])
