
🍌 Banana Leaf Disease Classification 🩺🌿
🎯 What this project does
This project trains a Convolutional Neural Network (CNN) to detect and classify banana leaf diseases into four categories:

🍂 Cordana

✅ Healthy

🍄 Pestalotiopsis

🌑 Sigatoka

📂 Dataset
Structure:

Copy
Edit
OriginalSet/
├── cordana/
├── healthy/
├── pestalotiopsis/
└── sigatoka/
Each folder contains images of leaves belonging to that class.

🛠 Features
🖼 Image visualization — see sample images from each class.

🎨 Data augmentation — random flips, color jitter for better generalization.

⚡ GPU-ready training — automatically uses CUDA if available.

📊 Metrics tracking — precision & recall after each epoch.

🎯 SimpleCNN — lightweight model for quick experiments.

🚀 How to run
Install dependencies

bash
Copy
Edit
pip install torch torchvision matplotlib pillow scikit-learn
Update dataset path
Change:

python
Copy
Edit
base_path = r"D:\ai_data\ai_data_advanced\week03\data\ BananaLSD\OriginalSet"
to your local dataset location.

Run the script

bash
Copy
Edit
python banana_leaf_disease_classification_cnn.py
📈 Outputs
🖼 Sample and augmented images before training.

📊 Bar chart of precision & recall after each epoch.

💬 Console logs showing loss, precision, and recall.

🧠 Model Overview
Conv1 → ReLU → MaxPool

Conv2 → ReLU → MaxPool

Fully Connected Layers → Output logits

Loss: CrossEntropyLoss

Optimizer: Adam