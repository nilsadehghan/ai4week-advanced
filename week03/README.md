# 🌟 Week03 - NLP & Computer Vision Projects  

This repository contains **two** machine learning workflows completed during **Week 03**:  

1. 🤖 **Fine-tuning ParsBERT** for Persian product review classification based on quality score.  
2. 🌿 **Banana leaf disease classification** using a simple CNN on image data.  

---

## 📌 Part 1: Fine-Tuning ParsBERT for Product Quality Classification

### 📝 Overview  
We fine-tune the [ParsBERT](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased) model to classify Persian product reviews into three categories:  
- 🟥 **Low-quality-product**  
- 🟨 **Medium-quality-product**  
- 🟩 **High-quality-product**  

Classification is based on a numeric `Score` column from the dataset.

---

### 🔄 Steps
1. **Data Loading & Label Assignment**  
   - Load `data.csv` (must contain `Score` and `Text` columns).  
   - Convert numeric `Score` values into labels using defined thresholds.  

2. **Text Preprocessing**  
   - Tokenize Persian text using **Hazm** (`WordTokenizer`).  
   - Remove Persian stopwords (`stopwords_list`).  

3. **Label Encoding & Dataset Split**  
   - Encode labels with `LabelEncoder`.  
   - Split data into **train (80%)** and **test (20%)** (stratified).  

4. **Tokenization for ParsBERT**  
   - Use HuggingFace's `AutoTokenizer` for ParsBERT.  
   - Tokenize with padding, truncation, and `max_length=128`.  

5. **Model Setup & Training**  
   - Load `AutoModelForSequenceClassification` with `num_labels=3`.  
   - Train using HuggingFace's `Trainer` with:
     - Learning rate `2e-5`
     - Batch size `4`
     - Weight decay `0.01`
     - Evaluation & checkpointing after each epoch.  

6. **Metrics**  
   - Accuracy, Precision, Recall, F1 (macro average).  

7. **Saving the Model**  
   - Save fine-tuned model and tokenizer locally.

---



📌 Part 2: Banana Leaf Disease Classification with CNN
📝 Overview
We train a simple Convolutional Neural Network (CNN) to classify banana leaves into:

🍂 cordana

🌱 healthy

🍄 pestalotiopsis

☀️ sigatoka

Dataset images are organized into folders by class.

🔄 Steps
1.Dataset Preparation


2.Visualization

3.Display first 2 images per class using matplotlib.

4.Transforms

5.Training: Resize (128×128), ColorJitter, RandomHorizontalFlip, Normalize.

6.Validation: Resize, Normalize.

7.Data Loading

8.Use ImageFolder to load dataset.

9.Split into 80% train / 20% validation.

10.CNN Model

11.Training Loop

12.Evaluate on validation set using macro Precision & Recall.

