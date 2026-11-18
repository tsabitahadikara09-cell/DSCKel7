# DermaSense — ISIC 2019 Streamlit App

Streamlit research prototype styled to match the supplied DermaSense UI reference.

## Included pages
- Dashboard
- Dataset Explorer (Overview, Data Viewer, Metadata, Preprocessing)
- Model CNN — EfficientNet-B0 training configuration and training
- Model Multimodal — architecture concept and missing-metadata scenario selector
- Evaluation — test-set metrics, confusion matrix, classification report
- About & Methods

## Dataset
Use ISIC 2019 training files:
- `ISIC_2019_Training_GroundTruth.csv`
- `ISIC_2019_Training_Metadata.csv` (optional)
- `ISIC_2019_Training_Input/` image folder

The app filters the three requested classes: MEL, NV, BCC.

### Local run
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

On the Dataset page:
1. Upload the Ground Truth CSV.
2. Optionally upload the Metadata CSV.
3. Set the image folder path to your ISIC image directory.
4. Click **Muat Dataset**.
5. Open **Preprocessing** and create the 70/15/15 split.
6. Open **Model CNN** and configure training.

You may upload a ZIP containing the image files from the Dataset page. Extracted files are stored under `data/ISIC_2019_Training_Input/`.

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository.
2. Add `app.py`, `requirements.txt`, and this README.
3. Do not upload the full ISIC image dataset to a public repository unless its license and size make that appropriate.
4. Deploy the repository from https://share.streamlit.io.
5. For large datasets, use a private external storage solution or a controlled dataset download workflow. Streamlit Cloud local filesystem is not durable storage for user uploads.

## Important implementation notes
- The Dataset Explorer works with the official ISIC 2019 one-hot ground-truth columns `MEL`, `NV`, `BCC` and image ID column `image`.
- Metadata fields recognized: `age_approx`, `sex`, `anatom_site_general`, and `lesion_id`.
- The split attempts to group by `lesion_id` when available to reduce leakage; otherwise it uses stratified random splitting.
- CNN training uses EfficientNet-B0 with weighted cross-entropy. Training runs on CPU if CUDA is unavailable and may be slow for the full dataset.
- The current Model Multimodal page is an architecture/experiment planning interface. Actual multimodal training and missing-mask fusion still need to be implemented as a separate model pipeline.
- Evaluation metrics are computed only after training and evaluation on the held-out test split. No fake metrics are displayed.

This is a research prototype, not a clinical diagnostic system. Do not upload identifiable patient information. Confirm the ISIC dataset license and attribution requirements before redistribution.
