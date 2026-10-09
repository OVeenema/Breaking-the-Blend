# Data folder

Put the IFU cubes for each galaxy in its own sub-folder here (e.g. `data/NGC5728/`). The notebooks search a
galaxy's folder **and all its sub-folders** for `*_s3d.fits` / `*_s3d.fits.gz` files, so a MAST download can be
used as it is.

Data cubes are **not** stored in this repository (they are several GB, and individual cubes exceed GitHub's
100 MB file limit); `.gitignore` keeps them out of git. Download them from MAST instead - see
[`NGC5728/README.md`](NGC5728/README.md) for the example dataset.
