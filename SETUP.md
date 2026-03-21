# Setup Guide (Windows)

## The Error You Saw

```
[WARNING] Missing: data\Balloon_Race_Data_Breaches...
ValueError: No objects to concatenate
```

This means the program couldn't find the data files. Fix it by following the steps below.

---

## Step 1 — Check Your Folder Structure

After unzipping, your folder should look like this:

```
crypto_ml_project\
├── main.py
├── data_ingestion.py
├── feature_engineering.py
├── models.py
├── evaluation.py
├── generate_notebook.py
├── requirements.txt
├── README.md
└── data\                      ← YOU MUST CREATE THIS FOLDER
    ├── Balloon_Race_Data_Breaches_-_LATEST_-_breaches.csv
    ├── Cyber_Security_Breaches.csv
    ├── Data_BreachesN_new.csv
    ├── Data_Breaches_EN_V2_2004_2017_20180220.csv
    ├── df_1.csv
    └── 260306_Cyber_Events.pdf
```

**The `data\` folder is NOT included in the zip** because the files are your uploaded datasets.
You need to create it and copy your files in manually.

---

## Step 2 — Create the data\ Folder and Copy Files

Open **Command Prompt** in the `crypto_ml_project` folder and run:

```cmd
mkdir data
```

Then copy all 6 files (the CSVs and PDF you originally uploaded) into that `data\` folder.

---

## Step 3 — Install Dependencies

```cmd
pip install -r requirements.txt
```

For the PDF parser (needed for the CSIS cyber events file), install poppler:

**Windows (easiest):** Download from https://github.com/oschwartz10612/poppler-windows/releases
- Download the latest release zip
- Extract it (e.g. to `C:\poppler`)
- Add `C:\poppler\Library\bin` to your system PATH

Or skip it — the pipeline still works with the 5 CSV files even without the PDF.

---

## Step 4 — Run the Pipeline

```cmd
python main.py
```

If your data files are in a different location, specify the path:

```cmd
python main.py --data_dir "C:\Users\usr1\Downloads\my_data_folder"
```

---

## Step 5 — Open the Notebook (Optional)

```cmd
pip install jupyter
python generate_notebook.py
jupyter notebook Crypto_Breach_ML_Pipeline.ipynb
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Missing: data\...` | Create the `data\` folder and copy your 6 files into it |
| `ModuleNotFoundError: sklearn` | Run `pip install scikit-learn` |
| `ModuleNotFoundError: pandas` | Run `pip install pandas` |
| `pdftotext not found` | Install poppler (see Step 3) — the PDF file will be skipped gracefully |
| `No objects to concatenate` | At least one CSV must be in `data\` — check filenames match exactly |

---

## File Name Checklist

Make sure the file names in your `data\` folder match **exactly** (case-sensitive on some systems):

- [ ] `Balloon_Race_Data_Breaches_-_LATEST_-_breaches.csv`
- [ ] `Cyber_Security_Breaches.csv`
- [ ] `Data_BreachesN_new.csv`
- [ ] `Data_Breaches_EN_V2_2004_2017_20180220.csv`
- [ ] `df_1.csv`
- [ ] `260306_Cyber_Events.pdf`

If any name is slightly different, either rename the file or update the filename in `data_ingestion.py`
(look for the `loaders` list near the bottom of the file).
