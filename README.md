# Polypharmacy Drug-Drug Interaction (DDI) Prediction System

---

[English](#english) | [Türkçe](#türkçe)

<a id="ingilizce"></a>
## English

This project is a machine learning system that predicts the risk of adverse interactions that can occur when multiple drugs are used together. Using the TWOSIDES dataset and the molecular structure of drugs (Morgan Fingerprint), a binary classification model was built. Three models (Logistic Regression, Random Forest, XGBoost) were compared, and Logistic Regression, which gave the highest Recall and ROC-AUC scores, was chosen as the final model. The system was extended, through pairwise combination decomposition, to handle polypharmacy scenarios with up to 10 drugs, and is available as an interactive app built with Streamlit.

**Important note:** This project is for academic and research purposes only. Its outputs are not medical advice and should not be used for real clinical decisions.

### Folder Structure

```
├── tr/
│   ├── notebook/     → Notebook in Turkish (EDA, modeling, SHAP)
│   └── report/       → Academic report in Turkish
├── en/
│   ├── notebook/     → Notebook in English
│   └── report/       → Academic report in English
└── app/
    ├── app.py            → Streamlit app
    ├── requirements.txt  → Required libraries
    └── logistic_regression_model.pkl → Trained model
```

### Live App

https://polypharmacy-ddi-prediction-ff3lh34cmkqjhz6vkn2mpk.streamlit.app/

### Tech Stack

Python | Pandas | NumPy | RDKit | Scikit-learn | XGBoost | SHAP | Joblib | Streamlit

---- 


<a id="türkçe"></a>
## Türkçe

Bu proje, birden fazla ilacın birlikte kullanımında ortaya çıkabilecek olumsuz etkileşim riskini tahmin eden bir makine öğrenmesi sistemidir. TWOSIDES veri seti ve ilaçların moleküler yapısından (Morgan Fingerprint) yola çıkılarak bir ikili sınıflandırma modeli geliştirilmiştir. Üç model (Logistic Regression, Random Forest, XGBoost) karşılaştırılmış, en yüksek Recall ve ROC-AUC değerlerini veren Logistic Regression nihai model olarak seçilmiştir. Sistem, ikili kombinasyon ayrıştırması yoluyla 10 ilaca kadar polifarmasi senaryolarını değerlendirebilecek şekilde genişletilmiş ve Streamlit üzerinden interaktif bir uygulama olarak sunulmuştur.

**Önemli not:** Bu proje akademik ve araştırma amaçlıdır. Üretilen sonuçlar tıbbi tavsiye niteliği taşımaz ve gerçek klinik kararlar için kullanılmamalıdır.

### Klasör Yapısı

```
├── tr/
│   ├── notebook/     → Türkçe notebook (EDA, modelleme, SHAP)
│   └── report/       → Türkçe akademik rapor
├── en/
│   ├── notebook/     → İngilizce notebook
│   └── report/       → İngilizce akademik rapor
└── app/
    ├── app.py            → Streamlit uygulaması
    ├── requirements.txt  → Gerekli kütüphaneler
    └── logistic_regression_model.pkl → Eğitilmiş model
```

### Live App

https://polypharmacy-ddi-prediction-ff3lh34cmkqjhz6vkn2mpk.streamlit.app/

### Kullanılan Teknolojiler

Python | Pandas | NumPy | RDKit | Scikit-learn | XGBoost | SHAP | Joblib | Streamlit

---
