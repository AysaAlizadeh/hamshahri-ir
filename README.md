# hamshahri-ir
Persian information retrieval on the Hamshahri corpus with TF-IDF and BM25.
cat > README.md << 'ENDFILE'
# Persian Information Retrieval on Hamshahri

A search system for the Hamshahri news corpus using inverted indexing, TF-IDF, and BM25.
Persian text is normalized, tokenized, and stemmed with Parsivar. Retrieval quality is measured with Precision, Recall, F1, and MAP by news category, with a simple GUI to pick a category.

The Hamshahri XML dataset is not included. Put the corpus in a local Dataset folder next to ir_system.py.

## Features

- Parse Hamshahri XML documents
- Stop-word removal and Persian stemming
- Inverted index
- TF-IDF and cosine similarity
- BM25 ranking
- Category-based evaluation (Precision, Recall, F1, MAP)
- Tkinter GUI for category search

## Setup

pip install pandas numpy scikit-learn openpyxl parsivar
python ir_system.py
ENDFILE
