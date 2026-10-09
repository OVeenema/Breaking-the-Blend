#!/usr/bin/env python
"""
Download the level-3 JWST IFU cubes (``*_s3d.fits``) of a target from MAST.

Example (NGC 5728, the data used in notebook 01):

    python scripts/download_mast_cubes.py --target "NGC 5728" --programs 1670 5017 --out data/NGC5728

Add ``--dry-run`` to list the files (and their total size) without downloading.
Needs ``astroquery`` (included in environment.yml; otherwise ``pip install astroquery``).
The cubes are several GB in total.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help='target name resolvable by MAST, e.g. "NGC 5728"')
    ap.add_argument("--programs", nargs="+", required=True, help="JWST program (proposal) IDs, e.g. 1670 5017")
    ap.add_argument("--out", required=True, help="output folder, e.g. data/NGC5728")
    ap.add_argument("--radius", default="0.01 deg", help="search radius around the target (default 0.01 deg)")
    ap.add_argument("--dry-run", action="store_true", help="only list the files")
    a = ap.parse_args()

    from astroquery.mast import Observations

    obs = Observations.query_criteria(objectname=a.target, radius=a.radius, obs_collection="JWST",
                                      proposal_id=a.programs, dataproduct_type="cube", calib_level=3)
    if len(obs) == 0:
        raise SystemExit("No JWST level-3 cube observations found - check the target name / program IDs, and that "
                         "the data are public (MAST 'Release Date').")
    products = Observations.get_product_list(obs)
    cubes = Observations.filter_products(products, productSubGroupDescription="S3D", calib_level=[3])
    if len(cubes) == 0:
        raise SystemExit("Observations found, but no S3D cube products.")
    size = sum(float(s) for s in cubes["size"]) / 1e9 if "size" in cubes.colnames else float("nan")
    print(f"{len(cubes)} cube(s), {size:.1f} GB:")
    for name in cubes["productFilename"]:
        print("  ", name)
    if a.dry_run:
        return
    os.makedirs(a.out, exist_ok=True)
    try:
        Observations.download_products(cubes, download_dir=a.out, flat=True)
    except TypeError:            # older astroquery without flat=
        Observations.download_products(cubes, download_dir=a.out)
    print(f"done -> {a.out}  (the notebooks search this folder and its sub-folders)")


if __name__ == "__main__":
    main()
