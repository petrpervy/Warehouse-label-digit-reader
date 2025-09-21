# Data Directory

This directory contains the synthetic training data for the SKU digit recognition model.

## Structure

- `synth/` - Generated synthetic images and labels
  - `*.png` - Grayscale images of digit strings
  - `labels.csv` - CSV file with filename,text columns

## Generating Data

To generate synthetic training data:

```bash
python -m src.synth_digits --count 50000 --len 3-8 --out data/synth
```

This creates 50,000 images with digit strings of length 3-8 characters.
