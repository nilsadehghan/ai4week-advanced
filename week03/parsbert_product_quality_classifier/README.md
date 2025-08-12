
📦 ParsBERT Product Quality Classifier
🎯 Fine-tune ParsBERT to classify Persian product reviews into quality levels
(Low, Medium, High) based on review scores.

🛠 Features
📝 Persian text preprocessing with Hazm (tokenization + stopword removal)

🤖 Fine-tuning ParsBERT for sequence classification

📊 Macro Precision / Recall / F1 / Accuracy evaluation metrics

💾 Saves fine-tuned model & tokenizer for future use

📂 Dataset Requirements
Your data.csv file should contain:

Column	Description
Text	The Persian review text
Score	A numerical score (0–100)

📊 Labeling Rules
Score Range	Label
0 < score < 20	Low-quality-product
20 < score < 60	Medium-quality-product
≥ 60	High-quality-product

🚀 Installation
bash
Copy
Edit
pip install pandas numpy hazm scikit-learn torch transformers datasets
▶️ Usage
Place data.csv in the same folder as the script.

Run the script:

bash
Copy
Edit
python parsbert_product_quality_classifier.py
Model & tokenizer will be saved in:

bash
Copy
Edit
./my_finetuned_model
./my_finetuned_tokenizer