# ⚖️ FairLens – AI Bias Auditor

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Fairlearn](https://img.shields.io/badge/Fairlearn-ML_Fairness-00B894?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/github/license/prajwal-Inovator/fairlens-streamlit?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/prajwal-Inovator/fairlens-streamlit?style=for-the-badge)

</p>

<p align="center">
  <b>Detect. Explain. Mitigate.</b><br>
  An interactive AI Fairness & Bias Auditing platform built with <b>Streamlit</b>, <b>Fairlearn</b>, and <b>Scikit-Learn</b> that helps evaluate, visualize, and reduce bias in Machine Learning models.
</p>

---

# 📖 Overview

**FairLens** is an intelligent AI fairness auditing platform designed to help developers, researchers, and organizations build **responsible AI systems**.

It enables users to upload datasets, identify potential bias against sensitive groups, visualize fairness metrics, understand model behaviour, and apply fairness mitigation techniques—all through a modern interactive Streamlit dashboard.

Instead of treating fairness as an afterthought, FairLens makes fairness evaluation a natural part of the machine learning workflow.

---

# ✨ Key Features

### 🔍 Automated Bias Detection

- Detects demographic bias in datasets
- Computes fairness metrics automatically
- Identifies disparities across sensitive groups

---

### 📊 Interactive Fairness Dashboard

- Beautiful visual analytics
- Bias severity gauge
- Group disparity charts
- Performance comparison graphs

---

### 🤖 Explainable AI

- Feature importance analysis
- Model explanation
- Understand why predictions are made

---

### ⚖️ Fairness Mitigation

- Applies fairness-aware machine learning
- Generates improved models with reduced bias
- Compares original vs mitigated performance

---

### 📂 Dataset Support

Works with custom datasets as well as sample datasets including:

- Adult Income Dataset
- German Credit Dataset
- COMPAS Dataset

---

### 🎯 User Friendly

- Drag & Drop dataset upload
- Interactive controls
- Real-time visualizations
- No coding required

---

# 🚀 Workflow

```text
             Dataset Upload
                    │
                    ▼
          Data Preprocessing
                    │
                    ▼
         Bias Detection Engine
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Fairness Metrics      Model Training
          │                   │
          └─────────┬─────────┘
                    ▼
          Explainable AI (XAI)
                    │
                    ▼
        Fairness Mitigation Engine
                    │
                    ▼
        Before vs After Comparison
                    │
                    ▼
           Interactive Dashboard
```

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend |
| Streamlit | Web Application |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Scikit-Learn | Machine Learning |
| Fairlearn | Fairness Metrics & Mitigation |
| Plotly | Interactive Charts |
| Matplotlib | Visualizations |

---

# 📂 Repository Structure

```text
fairlens-streamlit/
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── ml-engine/
│   ├── core/
│   │   ├── bias_detector.py
│   │   ├── explainer.py
│   │   ├── fair_model.py
│   │   └── preprocessor.py
│   │
│   └── datasets/
│       ├── adult_income.csv
│       ├── german_credit.csv
│       └── compas.csv
│
└── .gitignore
```

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://github.com/prajwal-Inovator/fairlens-streamlit.git

cd fairlens-streamlit
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will launch in your browser.

---

# 📊 Sample Datasets

The repository includes ready-to-use datasets for experimentation.

- 👥 Adult Income
- 💳 German Credit
- ⚖️ COMPAS Recidivism

These datasets allow users to explore fairness metrics without requiring external data.

---

# 🎯 Core Capabilities

- ✅ Bias Detection
- ✅ Fairness Metrics
- ✅ Explainable AI
- ✅ Interactive Visualizations
- ✅ Fairness Mitigation
- ✅ Before vs After Comparison
- ✅ Sample Dataset Support
- ✅ Custom Dataset Upload

---

# 🌍 Applications

FairLens can be used in:

- AI Ethics Research
- Responsible AI Development
- Academic Projects
- ML Fairness Audits
- HR & Recruitment Analytics
- Credit Risk Analysis
- Healthcare AI
- Government AI Systems

---

# 🔮 Future Enhancements

- [ ] Support additional fairness metrics
- [ ] Multiple mitigation algorithms
- [ ] PDF Fairness Report Generation
- [ ] Model Export
- [ ] API Integration
- [ ] Cloud Deployment
- [ ] Dashboard Authentication
- [ ] Real-time Bias Monitoring

---

# 🤝 Contributing

Contributions are welcome!

1. Fork this repository

2. Create a feature branch

```bash
git checkout -b feature/YourFeature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature/YourFeature
```

5. Open a Pull Request

---

# 📸 Screenshots

> Add screenshots of your Streamlit dashboard here.

Example:

```
screenshots/
├── home.png
├── fairness-report.png
├── mitigation.png
└── dashboard.png
```

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Prajwal V Sortur**

GitHub: https://github.com/prajwal-Inovator

---

# ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork it

📢 Share it with others

---

<p align="center">

### ⚖️ Building Fair, Transparent and Responsible AI with FairLens

Made with ❤️ by **Prajwal V Sortur**

</p>
