# BERT-FRIDE

BERT-FRIDE is a transformer-based model developed to analyze user reviews and extract app design insights such as color scheme, navigation, loading speed, and readability. Using a custom labeling method and custom loss function, the model achieves high accuracy (approximately 98%), outperforming other architectures like XLNet, RoBERTa, and GPT-2.

This repository includes implementations for:
- **BERT-FRIDE:** Our main model based on BERT.
- **XLNet-FRIDE:** A variant using XLNet.
- **RoBERTa-FRIDE:** A variant using RoBERTa.
- **GPT2-FRIDE:** A variant using GPT-2.

## Repository Structure

- **README.md:** Project overview and instructions.
- **requirements.txt:** Python dependencies.
- **model.py:** BERT-FRIDE model implementation.
- **models_extra.py:** Implementations for XLNet, RoBERTa, and GPT-2 based models.
- **train.py:** Script to train the chosen model.
- **utils.py:** Utility functions for data loading and preprocessing.
- **data/sample_reviews.csv:** Sample dataset in CSV format.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sohaibcs1/bert-fried.git
   cd BERT-FRIDE
