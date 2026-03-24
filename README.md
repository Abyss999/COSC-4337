### Authenticate Kaggle Account

1.  Go to your Kaggle account settings (kaggle.com/your_username/account).
2.  Under the 'API' section, click 'Create New API Token'. This will download a `kaggle.json` file.
3.  Upload this `kaggle.json` file to your Colab environment. You can do this by clicking the 'Files' icon on the left sidebar (folder icon), then 'Upload to session storage' icon, and selecting your `kaggle.json` file.


### Training Log on Model 2 (DenseNet121)


#### Attempt 1
- **Label Weight [bacterial, normal, viral]:** 1 : 3.0 : 1  
- **Best Accuracy:** ~0.70  
- **Validation Accuracy Floating Range:** 0.63 ~ 0.70  
- **DenseNet121 Freezing:** DenseBlock 1–3 frozen  
- **FC:** 2‑layer MLP (512 → 128 → 3)  
- **Overfitting:** Yes  

---

#### Attempt 2
- **Label Weight [bacterial, normal, viral]:** 1 : 2.5 : 1.2  
- **Best Accuracy:** ~0.70  
- **Validation Accuracy Floating Range:** 0.63 ~ 0.70  
- **DenseNet121 Freezing:** DenseBlock 1–3 frozen  
- **FC:** 2‑layer MLP (512 → 128 → 3)  
- **Overfitting:** Yes  

---

#### Attempt 3
- **Label Weight [bacterial, normal, viral]:** 1 : 4.0 : 0.5  
- **Best Accuracy:** ~0.68  
- **Validation Accuracy Floating Range:** 0.60 ~ 0.68  
- **DenseNet121 Freezing:** DenseBlock 1–3 frozen  
- **FC:** 2‑layer MLP (512 → 128 → 3)  
- **Overfitting:** Yes (severe)  

---

#### Attempt 4
- **Label Weight [bacterial, normal, viral]:** 1 : 3.0 : 0.7  
- **Best Accuracy:** ~0.75  
- **Validation Accuracy Floating Range:** 0.72 ~ 0.76  
- **DenseNet121 Freezing:** DenseBlock 1–3 frozen + DenseBlock4 partial 
- **FC:** Reduced FC (1024 → 256 → 3)  
- **Overfitting:** Mild but improving  

---

#### Attempt 5 
- **Label Weight [bacterial, normal, viral]:** 1 : 3.0 : 0.7  
- **Best Accuracy:** 0.7605  
- **Validation Accuracy Floating Range:** 0.74 ~ 0.76  
- **DenseNet121 Freezing:** DenseBlock 1–3 frozen + DenseBlock4 partial 
- **FC:** Reduced FC (1024 → 256 → 3)  
- **Overfitting:** No (training finally stabilized)  

## Attempt 6 (Current Best)
- **Label Weight [bacterial, normal, viral]:** 0.8 : 6.0 : 0.6  
- **Best Accuracy:** 0.8285  
- **Validation Accuracy Floating Range:** 0.76 ~ 0.83  
- **DenseNet121 Freezing:** DenseBlock 1–3 frozen + DenseBlock4 partial 
- **FC:** Reduced FC (1024 → 256 → 3)  
- **Early Stopping:** Yes (patience = 5)  
- **Overfitting:** No (training stopped before overfitting)

## Attempt 7 (Current Best)
- **Label Weight [bacterial, normal, viral]:** 0.7 : 7.0 : 0.4  
- **Best Accuracy:** 0.8560  
- **Validation Accuracy Floating Range:** 0.66 ~ 0.86  
- **DenseNet121 Freezing:** DenseBlock 1–3 frozen + DenseBlock4 后半可训练  
- **FC:** Reduced FC (1024 → 256 → 3)  
- **Early Stopping:** Yes (patience = 5)  
- **Overfitting:** No  
- **Confusion Highlights:**  
  - BACTERIAL: 224 / 240（16 → NORMAL）  
  - NORMAL: 205 / 231（9 → BACTERIAL，17 → VIRAL）  
  - VIRAL: 100 / 147（37 → BACTERIAL，10 → NORMAL）
