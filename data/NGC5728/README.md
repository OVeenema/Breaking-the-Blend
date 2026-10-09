# NGC 5728 - example data for `notebooks/01_NGC5728.ipynb`

Notebook 01 reads the cubes from this folder (`DATA_DIR = "../data/NGC5728"`). It needs the level-3 (`s3d`)
cubes of the following public JWST programmes, from the Galaxy Activity, Torus and Outflow Survey (GATOS):

| Instrument | Programme ID | PI | Bands used |
|---|---|---|---|
| MIRI/MRS | **1670** | T. Shimizu & R. Davies | all 12 sub-bands (ch1-4 short/medium/long) |
| NIRSpec IFU | **5017** | I. García-Bernete | G395H/F290LP |

## Option 1 - download script

From the repository root, with the `breaking-the-blend` environment active:

```bash
python scripts/download_mast_cubes.py --target "NGC 5728" --programs 1670 5017 --out data/NGC5728 --dry-run
python scripts/download_mast_cubes.py --target "NGC 5728" --programs 1670 5017 --out data/NGC5728
```

The first command only lists the files; the second downloads them (several GB).

## Option 2 - MAST portal

1. Open the [MAST portal](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html).
2. Search for **NGC 5728** (Advanced Search: *Mission* = JWST, *Proposal ID* = 1670 or 5017).
3. Select the observations, open the **Download Basket**, and filter the products to **S3D** (level 3,
   `*_s3d.fits`) - one cube per MRS sub-band and one for NIRSpec G395H/F290LP.
4. Download and unpack the files anywhere inside `data/NGC5728/`.

## Notes

* MAST products are processed with the current JWST pipeline, whereas the published analysis used the GATOS
  reductions (Davies et al. 2024; Donnan et al. 2026). Results will be close to, but not identical with, the paper.
* In notebook 01, `USE_ORIGINAL_AGN_XY = False` (default) takes the AGN position of each cube from its continuum
  peak, which is what you want for MAST cubes. The pixel positions of the original analysis only apply to the
  original cubes.
* If a programme is still within its exclusive-access period, its files cannot be downloaded yet; the MAST
  "Release Date" column shows when it becomes public.
* Please acknowledge the programmes and MAST in any publication using these data.
