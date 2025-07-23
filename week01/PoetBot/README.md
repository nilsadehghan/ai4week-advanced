
# 📝 Poetic Chatbot with Chainlit & Transformers

This project is a **Persian poetic chatbot** that uses the power of [Hugging Face Transformers 🤗](https://huggingface.co/) and [Chainlit](https://www.chainlit.io/) to create an interactive, literary, and aesthetically beautiful conversation experience. The chatbot responds in a **poetic and elegant Persian tone** using the `rahiminia/manshoorai` language model.

---

## ✨ Features

- 🌸 Responds in a poetic and literary **Persian** style  
- 🤖 Powered by [`rahiminia/manshoorai`](https://huggingface.co/rahiminia/manshoorai) language model  
- 🔄 Interactive web-based UI using **Chainlit**  
- 🔐 Loads model safely with `use_safetensors=True`  
- 🎯 Configurable generation with nucleus sampling (`top_p`, `top_k`)  
- 🧠 Automatically combines a poetic persona with user prompts  

---

## 🚀 Getting Started

### ✅ Prerequisites

Make sure you have Python 3.8+ and the required packages:

```bash
pip install torch transformers chainlit
````

### ▶️ Run the Chatbot

```bash
chainlit run chatbot.py
```

This will launch the Chainlit chat UI in your browser.

---

## 🌐 Example

**User:**
چگونه می‌توانم آرامش پیدا کنم؟
(*How can I find peace?*)

**Bot:**
در دل شب‌های ساکت، نجوا کن با ستارگان
آرامش را از دل شعرِ مهتاب بجوی...
(*In the silence of night, whisper to the stars,
Seek peace from the heart of moonlight’s poem...*)

---

## 📁 File Structure

```
poetic/
├── chatbot.py         # Main chatbot application
├── README.md          # Project documentation
```

---

## 💡 How It Works

1. Loads the Persian poetic model: `rahiminia/manshoorai`
2. Sets a poetic persona:

   > تو باید سوالات رو به صورت شاعرانه و ادبی پاسخ بدی
3. Combines the user message with the persona
4. Uses `generate()` with top-k and top-p sampling to create diverse poetic replies
5. Displays the result in a live chat using Chainlit

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

* [Hugging Face 🤗](https://huggingface.co/)
* [Chainlit](https://www.chainlit.io/)
* [Model: rahiminia/manshoorai](https://huggingface.co/rahiminia/manshoorai)

