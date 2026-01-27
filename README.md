# My Masters Project

## Project Structure

This repository is structured as follows:

```
My_Masters_Project/
├── .git/               <- Git version control
├── data/               <- Symlinks to HPC data folders (to be created manually)
├── src/                <- Custom models and training scripts
│   └── __init__.py     <- Makes src a Python package
├── notebooks/          <- Analysis and visualization
└── README.md           <- This file
```

## Setup

1.  **Data**: The `data/` directory is intended to contain symlinks to your actual data locations (e.g., on HPC storage).
    ```bash
    ln -s /path/to/hpc/data data/my_data_name
    ```
2.  **Environment**: Activate the virtual environment before running code:
    ```bash
    source .venv/bin/activate
    ```
