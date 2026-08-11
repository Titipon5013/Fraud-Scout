# Fraud-Scout

🛡️ Fraud-Scout is an AI-powered web application for retail anomaly detection and fraud prevention. It helps business owners and SMEs identify suspicious sales patterns from transaction logs in CSV format using NLP-based query matching and rule-based analysis.

The system is designed for practical retail use, especially in environments where cash handling and point-of-sale (POS) behavior need closer monitoring. It supports Thai natural-language input and provides fast, explainable results without storing uploaded data.

## ✨ Key Features

- Thai NLP smart matching for suspicious behavior descriptions
- Rule-based anomaly detection for common fraud patterns
- Zero-data retention policy: uploaded data is processed in memory only
- Bilingual interface (Thai and English)
- Interactive dashboard for reviewing suspicious transactions

## 🚨 Detection Cases

Fraud-Scout currently detects three main risk patterns:

1. Midnight Sales
   - Detects transactions occurring outside normal business hours, such as late-night sales.

2. Ghost Transactions
   - Detects missing or incomplete critical information such as item name, price, quantity, or total spent.

3. Price Manipulation
   - Detects cases where the total amount does not match the item price multiplied by quantity.

## 🛠️ Tech Stack

- Frontend and Backend: Streamlit
- Data Processing: Pandas, NumPy
- NLP and Machine Learning: Scikit-learn, PyThaiNLP

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Titipon5013/Fraud-Scout.git
cd Fraud-Scout
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn pythainlp
```

### 3. Run the application

```bash
streamlit run app.py
```

## 📖 How to Use

1. Open the local Streamlit URL in your browser (usually http://localhost:8501).
2. Describe the suspicious behavior in the search box, for example:
   - "cashier deleting bills"
   - "midnight sales"
   - "total amount does not match the item price"
3. Click "Analyze Problem" to match the description to the most relevant fraud case.
4. Upload your transaction log file in CSV format.
5. Review the suspicious transactions displayed by the system.

## 📁 Project Structure

```text
Fraud-Scout/
├── app.py                  # Main Streamlit application
├── data_resources/        # Sample transaction datasets
│   ├── retail_store_sales.csv
│   └── retail_store_sales_engineered.csv
├── data_cleaning.ipynb    # Notebook for data cleaning and testing
└── README.md              # Project documentation
```

## 🔒 Security and Privacy

Fraud-Scout follows a strict zero-data retention approach. Uploaded transaction data is processed only in memory during the active session and is not saved to any database.

## 👨‍💻 Developer

Developed by: Titipon Tawong
