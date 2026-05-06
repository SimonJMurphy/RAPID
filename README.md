# RAPID: Rotation-Aware Probabilistic Inference Dashboard

RAPID is an interactive applet for inferring stellar parameters from synthetic populations of rapidly rotating intermediate-mass stars.

The synthetic populations presented here cover stars with masses of roughly 1.4 to 2.5 $M_\odot$ and include the effects of rotation, inclination, unresolved binarity, metallicity variation, and observational uncertainty. Given a target star's effective temperature and luminosity, RAPID estimates posterior distributions for mass, age, and other stellar properties.

## Features

- Place a target star on the HR diagram using $T_{\mathrm{eff}}$ and $L$.
- Enter asymmetric uncertainties in $T_{\mathrm{eff}}$ and $L$.
- Toggle between the full synthetic population and the no-binaries sample.
- Visualise the distance-weighted probability field around the target.
- Compute KDE-based posterior summaries for mass and age.
- Generate weighted corner plots for selected stellar parameters.
- Export the weighted $3\sigma$ selection to CSV for further analysis.

## Quick start

Clone the repository:
```bash
git clone https://github.com/SimonJMurphy/RAPID
```

Install the Python dependencies:

```bash
cd RAPID
pip install -r requirements.txt
```

Run the applet from the repository root:

```bash
python applet.py
```

The applet loads the required data files from the `data/` directory.

## Data files

The repository contains both Feather and CSV versions of the synthetic populations. Feather files are preferred for normal use because they load much faster.

| File | Description |
| --- | --- |
| `data/popsynth_HRD_outputs_extended_apr2026.feather` | Full synthetic population, including unresolved binaries. |
| `data/popsynth_HRD_outputs_no-binaries_extended_apr2026.feather` | Synthetic population with no binary companions. |
| `data/solar_tracks.feather` | Solar-metallicity evolutionary tracks plotted for context. |
| `data/*.csv` | CSV copies of the same data products. |

## Citation

If you use RAPID or the synthetic populations, please cite the associated publication once it is available. Citation details, DOI, and data-release links will be added here after publication.
