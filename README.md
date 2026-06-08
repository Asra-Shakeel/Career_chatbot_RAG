# Career_chatbot_RAG
**Career Chatbot RAG**
**Overview**

This project is a Career Guidance Chatbot based on Retrieval-Augmented Generation (RAG). It provides career-related guidance by retrieving relevant information from a dataset and generating responses.

**Features**
Career guidance and recommendations
Dataset-based information retrieval
Interactive chatbot interface
RAG architecture
**Requirements**

Install the required libraries:

pip install -r requirements.txt

If requirements.txt is not available:

pip install pandas numpy scikit-learn sentence-transformers faiss-cpu gradio
Project Structure
Career_chatbot_RAG/
│
├── app.py
├── dataset.csv
├── embeddings/
├── README.md
└── other project files
**How to Run**

Open Command Prompt and navigate to the project folder:

cd Career_chatbot_RAG

Run the application:

python app.py

After running successfully, a local URL will appear:

Running on local URL: http://127.0.0.1:7860

Open the URL in your browser to use the chatbot.

Dataset

The chatbot uses a career guidance dataset for retrieving relevant information and generating responses.

Author

Asra Shakeel
