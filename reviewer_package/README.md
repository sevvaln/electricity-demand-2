# Elektrik Talep Analizi

## Reviewer Shortcut

Bu repoda cok sayida deney ve ara artefakt bulundugu icin, disaridan inceleme yapacak kisiler icin temiz ve kucuk paket `reviewer_package/` altinda hazirlandi.

- Reviewer-facing final kod, cikti ve raporlar: `reviewer_package/`
- Eski deneyler ve buyuk ara artefaktlar icin plan/manifests: `archive_full/`

Bu repo, 2022, 2023 ve 2024 yillari icin saatlik elektrik tuketimi ve sicaklik verilerini kullanarak farkli OLS regresyon modelleri kurar, VIF tablolari uretir ve modellerin `R^2` performanslarini karsilastirir.

Ana calisan dosya:

- `talep_tahmin_tek_dosya.py`
- `xgboost_train.py`

## Veri Yapisi

Ham veriler `data/raw/` altindadir:

- `data/raw/consumption/`
- `data/raw/temperature/`
- `data/raw/economic/`

Kullanilan temel degiskenler:

- `HDD`
- `CDD`
- `night dummy`
- `weekend dummy`
- `month dummies`
- `PMI`
- `IR`
- `CUR`

## Kurulan Modeller

Script, her yil icin su 8 modeli ayri ayri kurar:

1. `consumption ~ HDD + CDD`
2. `consumption ~ HDD + CDD + night dummy`
3. `consumption ~ HDD + CDD + night dummy + weekend dummy`
4. `consumption ~ HDD + CDD + night dummy + weekend dummy + month dummies`
5. `consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR`
6. `consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR + CUR`
7. `log_consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR`
8. `sqrt_consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR`

## Uretilen Ciktilar

Script calistiginda `outputs/` altinda su dosyalar olusur:

- Her yil icin ayri model rehberi
- Her tablo icin ayri OLS regression table
- Her tablo icin ayri VIF table
- Her tablo icin ayri model summary
- Her tablo icin ayri standardizasyon bazlari
- Her tablo icin ayri model diagnostic
- 2022, 2023 ve 2024 icin tuketim figurlari
- `R^2` karsilastirma grafigi
- `R^2` isi haritasi

Ornek ciktilar:

- `outputs/reports/2022_model_rehberi.txt`
- `outputs/reports/2023_table_6_ols_regresyon_tablosu.csv`
- `outputs/reports/2024_table_6_vif_tablosu.txt`
- `outputs/reports/tum_yillar_ols_regresyon_tablolari.csv`
- `outputs/reports/tum_yillar_vif_tablolari.txt`
- `outputs/reports/r_squared_karsilastirma_tablosu.csv`
- `outputs/figures/r_squared_karsilastirma_grafigi.png`
- `outputs/figures/r_squared_karsilastirma_isiharitasi.png`

## XGBoost Egitim Akisi

`xgboost_train.py`, 2022, 2023 ve 2024 verilerini kullanarak iki ayri `XGBoost` modeli egitir:

- `Table 3 Intramonth`: `consumption ~ HDD + CDD + night dummy + weekend dummy`
- `Table 8 Broadscale`: `sqrt_consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR`
- `Table 3 Intramonth SARIMA-Guided`: `Table 3` cekirdegi + SARIMA bulgularina dayali `lag_1` ve 24 saatlik seasonal-difference proxy
- `Table 8 Broadscale SARIMA-Guided`: `Table 8` cekirdegi + SARIMA bulgularina dayali `lag_1` ve 24 saatlik seasonal-difference proxy

`Table 8` modelinde aylik ekonomik seriler leakage riskini azaltmak icin ayni ay yerine bir onceki ay degeri olarak hizalanir. Bu asamada `2025` verilmedigi icin validation/test metriği uretilmez; 2025 dosyalari geldikten sonra her iki model icin ayri test/prediction ve `R^2` hesabi calistirilir.

Bu script su ciktilari uretir:

- `outputs/xgboost/table3_intramonth_train_predictions.csv`
- `outputs/xgboost/table3_intramonth_feature_importance.csv`
- `outputs/xgboost/table3_intramonth_leakage_audit.csv`
- `outputs/xgboost/table3_intramonth_leakage_notlari.txt`
- `outputs/xgboost/table8_broadscale_train_predictions.csv`
- `outputs/xgboost/table8_broadscale_feature_importance.csv`
- `outputs/xgboost/table8_broadscale_leakage_audit.csv`
- `outputs/xgboost/table8_broadscale_leakage_notlari.txt`
- `outputs/xgboost/table3_intramonth_sarima_guided_train_predictions.csv`
- `outputs/xgboost/table3_intramonth_sarima_guided_feature_importance.csv`
- `outputs/xgboost/table3_intramonth_sarima_guided_leakage_audit.csv`
- `outputs/xgboost/table3_intramonth_sarima_guided_leakage_notlari.txt`
- `outputs/xgboost/table8_broadscale_sarima_guided_train_predictions.csv`
- `outputs/xgboost/table8_broadscale_sarima_guided_feature_importance.csv`
- `outputs/xgboost/table8_broadscale_sarima_guided_leakage_audit.csv`
- `outputs/xgboost/table8_broadscale_sarima_guided_leakage_notlari.txt`
- `outputs/xgboost/xgboost_training_summary.csv`
- `outputs/xgboost/xgboost_2025_test_hazirlik.txt`
- `outputs/models/table3_intramonth_xgboost_model.json`
- `outputs/models/table3_intramonth_xgboost_metadata.json`
- `outputs/models/table8_broadscale_xgboost_model.json`
- `outputs/models/table8_broadscale_xgboost_metadata.json`
- `outputs/models/table3_intramonth_sarima_guided_xgboost_model.json`
- `outputs/models/table3_intramonth_sarima_guided_xgboost_metadata.json`
- `outputs/models/table8_broadscale_sarima_guided_xgboost_model.json`
- `outputs/models/table8_broadscale_sarima_guided_xgboost_metadata.json`

## Calistirma

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python talep_tahmin_tek_dosya.py
python xgboost_train.py
```

## GitHub'a Yukleme

Bu repo su an ciktilariyla birlikte GitHub'a yuklenmeye uygun hale getirildi. `outputs/` klasoru ignore edilmiyor; yani olusan tablolar ve grafikler de commit'e dahil olabilir.

Terminalde proje klasorunde su adimlari izleyebilirsin:

```powershell
cd "C:\Users\Monster\OneDrive\Masaüstü\proje2"
git init
git add .
git commit -m "Initial project upload"
git branch -M main
git remote add origin <REPO_URL>
git push -u origin main
```

`<REPO_URL>` yerine GitHub'da olusturdugun bos reponun adresini koy:

```text
https://github.com/kullanici-adi/repo-adi.git
```

## Not

- `.venv`, `__pycache__` ve benzeri gecici dosyalar repo'ya gitmez.
- Ciktilar repo'ya dahildir; yani GitHub'da tablolar ve grafikler dogrudan gorulebilir.
