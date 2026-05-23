# XGBoost Forecasting Pipeline Ozet Notu

## 1. Project Context

Bu projenin makine ogrenmesi kismi, Turkiye'nin saatlik elektrik tuketimini tahmin etmeyi hedefliyor. Problemin zorlugu, yalnizca genel seviyeyi degil, gun icindeki ritmi, hafta ici ve hafta sonu farkini, mevsim etkisini ve sicaklik kaynakli yuk baskisini ayni anda yakalamak zorunda olmamizdan geliyor.

Calisma duzeni net olarak su sekilde kuruldu:

- Egitim donemi: `2022-2024`
- Dogrulama / backtest donemi: `2025`
- Son hedef: `2026` icin guvenilir forecast uretmek

Bu asamada iki temel sicaklik gostergesi kullandik:

- `weighted_HDD`: Nufus agirlikli heating degree day. Hava sogudukca isinma ihtiyacini temsil eder.
- `weighted_CDD`: Nufus agirlikli cooling degree day. Hava isindikca sogutma ihtiyacini temsil eder.

Buradaki "weighted" kisminin mantigi su: tum illerin sicakligini basit ortalama ile birlestirmek yerine, il bazinda HDD/CDD hesapladik ve sonra bunlari nufus paylariyla agirliklandirdik. Boylece Istanbul, Ankara, Izmir gibi nufusu yuksek yerlerin talep uzerindeki sicaklik etkisi modele daha gercekci bicimde yansitilmis oldu.

Takvim degiskenleri de saatlik tahminde kritik rol oynadi:

- `night`
- `weekend`
- `hour`
- `day_of_week`
- `month`
- resmi tatil ve tatil cevresi degiskenleri

Saatlik forecasting neden onemli? Cunku elektrik sistemi gun icinde cok farkli seviyelerde calisiyor. Gece tuketim seviyesi ile is gunu aksam zirvesi ayni degil. Dolayisiyla aylik ya da gunluk degil, saatlik modelleme daha operasyonel ve daha gercek bir talep resmi veriyor.

## 2. Why We Moved to XGBoost

Klasik OLS ve benzeri ekonometrik modeller bize iyi bir baslangic verdi. Hangi degiskenlerin anlamli oldugunu, hangi etkinin mantikli oldugunu ve hangi makro gostergelerin faydali olabilecegini bu tarafta gorduk. Ancak bu modellerin dogal sinirlari vardi.

Elektrik tuketimi ile sicaklik arasindaki iliski tam dogrusal degil. Ornegin hava biraz sogudugunda olusan etki ile cok sogudugunda olusan etki ayni siddette olmayabilir. Benzer sekilde sicaklik etkisi gece saatlerinde, hafta sonlarinda ya da resmi tatillerde farkli davranabilir. OLS ile bu etkileri tek tek denklem icine koymak mumkun ama hizli sekilde karmasiklasiyor.

XGBoost'a gecmemizin nedeni buydu:

- dogrusal olmayan iliskileri yakalayabilmesi
- degiskenler arasi etkilesimleri esnek bicimde ogrenebilmesi
- hava, takvim ve zamansal yapiyi birlikte kullanabilmesi
- forecasting problemlerinde guclu performans verebilmesi

Yani ekonometrik tarafin mantigini birakmadik; onu daha esnek bir tahminleme aracina tasidik.

## 3. Table 3 - Original XGBoost Design

Ilk guclu Table 3 tasarimimiz, SARIMA-esinli one-step-ahead modeldi. Bu model her saatte bir sonraki saati tahmin etmeye calisiyordu. Buradaki ana fikir, tuketimin bir onceki saatten ve gunluk ritimden guclu bicimde etkilenmesiydi.

Kullanilan feature set:

- `weighted_HDD`
- `weighted_CDD`
- `night`
- `weekend`
- `log_consumption_lag_1h`
- `log_consumption_seasonal_diff_24h`

Buradaki `lag` su anlama gelir: "bir saat once tuketim seviyesi neydi?"  
`seasonal difference` ise sunu temsil eder: "bir saat onceki seviye, bir gun onceki ritme gore ne kadar farkliydi?"

Bu nedenle model kisa donem hafiza tasiyor. SARIMA tarafinda gordugumuz otoregresif ve gunluk sezonsal yapi, burada dogrudan zaman serisi denklemi olarak degil, feature olarak modele veriliyor.

2025 validation sonuclari:

- `R2 = 0.9624`
- `MAE = 946.00`
- `RMSE = 1274.37`
- `MAPE = 2.33%`

Bu cok guclu bir sonuc verdi. Fakat burada onemli bir metodolojik soru dogdu: Bu model gercekten orta/uzun ufuk forecasting icin mi guclu, yoksa sadece gecmis gercek tuketimi saat saat kullanabildigi icin mi bu kadar yuksek performans veriyor?

## 4. Leakage and Forecasting Design Discussions

Bu asamada iki ana konuya odaklandik: leakage ve forecasting yorumu.

Leakage'in basit tanimi su: modelin tahmin aninda sahip olmamasi gereken bilgiyi dolayli yoldan kullanmasi.

Biz su kontrolleri yaptik:

- train/test split audit
- feature construction audit
- lag ve rolling feature'larin gelecegi kullanip kullanmadigi
- recursive ve non-recursive forecast farki

Teknik audit'lerde model leakage-safe cikti. Yani lag feature'lar gercekten gecmise dayaniyordu. Ancak bu, modelin yorumunun ayni oldugu anlamina gelmiyor. One-step-ahead bir model teknik olarak leakage-safe olabilir ama yine de yalnizca kisa donem operasyonlar icin anlamli olabilir.

Burada ana ayrim soyle netlesti:

- `one-step benchmark`: gecmis gercek degerleri kullanabilen kisa donem model
- `recursive forecast`: modelin kendi tahminlerini tekrar girdiye cevirdigi yapi
- `non-recursive forecast`: tum hedef ufkun tek seferde, predicted feedback olmadan tahmin edildigi yapi

## 5. Table 3 Rolling Month-Ahead Experiment

Original Table 3 cok gucluydu ama operasyonel olarak daha gercekci bir kullanim sekli test etmek istedik. Cunku gercek hayatta forecast pipeline'lari cogu zaman bir kez baslatilip tum yil recursive kosmaz.

Bu nedenle `rolling month-ahead` tasarimi denedik.

Mantik su:

- her ay basinda model yeniden egitiliyor
- sadece ilgili ay forecast ediliyor
- ay icinde recursive one-step mantigi devam ediyor
- fakat ay bittikten sonra tahminler sonraki aya otomatik tasinmiyor

Bu tasarimi deneme nedenimiz, daha gercekci bir operasyonel kullanim senaryosu kurmakti.

2025 pseudo-backtest sonuclari:

- Full-year recursive:
  - `R2 = 0.4721`
  - `RMSE = 4772.60`
  - `MAE = 3736.37`
  - `MAPE = 9.01%`
- Rolling month-ahead:
  - `R2 = 0.4915`
  - `RMSE = 4684.05`
  - `MAE = 3622.35`
  - `MAPE = 8.73%`

Rolling yapi biraz daha iyi cikti ama yine de guclu degildi. Buradan su sonuca vardik: Table 3'un asli gucu, one-step yapida gercek gecmis saatlik tuketimi kullanabilmesinden geliyor. Bu avantaj azalinca performans sert dusuyor.

Bu nedenle rolling yapi metodolojik olarak yararli bir deney olsa da final aday olmadi.

## 6. Table 8 - Historical Non-Recursive Design

Table 8 tarafinda daha uzun ufuklu, recursive olmayan bir tasarim kurduk. Buradaki ana fikir historical analog ve historical profile feature'lari kullanmakti.

Bu model 2025 tahmini yaparken 2025 icindeki gercek tuketimi saat saat geri beslemiyor. Onun yerine onceki yil benzer saat / benzer takvim penceresi davranislarini feature olarak kullaniyor.

Bu modelde iki kritik historical feature ailesi vardi:

- previous-year analogs
- previous-year grouped historical profiles

Burada cok onemli bir kavramsal ayrim var:

**Training data ile historical feature source ayni sey degildir.**

XGBoost modeli egitim asamasinda zaten `2022`, `2023` ve `2024` verilerinin tamamini goruyor. Yani model pattern ogrenme asamasinda bu yillarin hepsinden faydalaniyor. Dolayisiyla:

> "historical feature'larda previous-year kullaniyoruz"

demek,

> "model sadece 2024 verisini kullaniyor"

anlamina gelmez.

`2022-2023` bilgisi zaten modelin ogrenme surecinin icinde vardir. Buradaki ayri soru, historical analog ve grouped profile feature'larini **uretirken** hangi gecmis davranis kaynagini kullanmamiz gerektigiydi. Yani bu, modelin neyle egitildigi degil; feature engineering asamasinda "hangi gecmisi referans alalim?" sorusuydu.

Burada onemli bir tasarim sorusu ortaya cikti: historical profile'lari sadece onceki yildan mi üretelim, yoksa 2022-2024 gibi cok yilli ortalama mi kullanalim?

Bunu varsayimla secmedik; kontrollu test yaptik. Test edilen iki tasarim sunlardi:

- **A) validated original specification**
  - historical source = previous-year only
  - yani `2025` target satirlari icin grouped profile source'u `2024`
- **B) multi-year grouped profile design**
  - historical source = `2022-2024`
  - yani grouped profile daha genis tarihsel ortalamadan geliyor

Sonuclar:

- Previous-year only grouped profiles:
  - `R2 = 0.928574`
  - `RMSE = 1755.47`
  - `MAPE = 3.2515`
- Multi-year grouped profiles:
  - `R2 = 0.918537`
  - `RMSE = 1874.75`
  - `MAPE = 3.6492`

Delta (multi-year minus validated previous-year):

- `test_R2 = -0.0100`
- `test_RMSE = +119.28`
- `test_MAPE = +0.3977`

Yani multi-year historical profile tasarimi performansi dusurdu. Bunun sezgisel yorumu su olabilir:

- cok yilli ortalama daha "smooth", daha genel ve daha ortalamaya cekilmis bir davranis ogreniyor
- buna karsilik onceki yil, sistemin en guncel calisma rejimini daha iyi temsil ediyor

Kisacasi multi-year grouped design davranisi fazla ortalamaya cekiyor olabilir. Previous-year specification ise daha guncel operational behavior tasiyor.

Bu nedenle final Table 8 spesifikasyonu korundu:

- previous-year analogs
- previous-year grouped profiles
- recursive predicted feedback yok

Final Table 8 2025 sonuclari:

- `R2 = 0.9286`
- `MAE = 1263.12`
- `RMSE = 1755.47`
- `MAPE = 3.25%`

Bu model original one-step Table 3 kadar yuksek skor vermedi ama bilgi seti daha kisitli ve daha gercekci oldugu halde cok guclu performans verdi.

## 7. New Table 3 Medium-Horizon Model (Final Table 3 Version)

Projenin en kritik gelistirmesi bu oldu.

Original Table 3 cok iyiydi ama direkt saatlik actual lag tuketime dayaniyordu. Rolling month-ahead ise daha gercekciydi ama performans ciddi zayifladi. Bu ikisi arasinda yeni bir ara tasarim kurduk.

Amac:

- direct hourly actual lag bagimliligini kaldirmak
- temporal intelligence'i korumak
- operasyonel olarak daha tasinabilir bir yapi kurmak

Bu modelde dogrudan su feature'lari yasakladik:

- `log_consumption_lag_1h`
- `consumption_lag_1h`
- `log_consumption_seasonal_diff_24h`
- recursive predicted feedback

Ama zamani temsil etmeyi birakmadik. Bunun yerine historical profile feature'lari kullandik.

En onemli feature ornekleri:

- `prev_4week_same_hour_day_of_week_mean`
- `prev_7d_same_hour_mean`
- `hist_mean_by_hour_day_of_week`
- `hist_mean_by_hour_month`

Bunlarin sezgisel anlami:

- `prev_4week_same_hour_day_of_week_mean`:
  Son dort haftadaki benzer gun ve benzer saat davranisini temsil eder.
  Ornegin "son dort haftadaki Sali 14:00 ortalamasi" gibi dusunulebilir.

- `prev_7d_same_hour_mean`:
  Son gunlerde ayni saatin genel seviyesini ozetler.
  Ornegin "son yedi adet 14:00 degeri ortalamada neydi?"

- `hist_mean_by_hour_day_of_week`:
  Tarihsel olarak "Pazartesi 09:00", "Cuma 18:00" gibi paternleri ozetler.

- `hist_mean_by_hour_month`:
  Ayni ay ve ayni saat kombinasyonunun tarihsel davranisini temsil eder.

Bu nedenle model artik "bir saat once kacti?" diye ogrenmiyor. Ama "bu takvim penceresi tarihsel olarak nasil davranir?" bilgisini koruyor.

Leakage audit'te kontrol ettiklerimiz:

- forbidden lag feature yok
- recursive feedback yok
- future bilgilerden türetilmis direct hourly signal yok
- farkli cutoff noktalarinda future degerleri bozdugumuzda, cutoff oncesi feature'lar degismedi

Yani audit sonucu temiz cikti.

Bu yeni modelin 2025 performansi:

- `R2 = 0.9482`
- `MAE = 1056.13`
- `RMSE = 1495.57`
- `MAPE = 2.66%`

Karsilastirma:

- Original Table 3 One-Step:
  - `R2 = 0.9624`
  - `RMSE = 1274.37`
  - `MAPE = 2.33%`
- Rolling Table 3:
  - `R2 = 0.4915`
  - `RMSE = 4684.05`
  - `MAPE = 8.73%`
- Table 8 Final:
  - `R2 = 0.9286`
  - `RMSE = 1755.47`
  - `MAPE = 3.25%`
- New Medium-Horizon Table 3:
  - `R2 = 0.9482`
  - `RMSE = 1495.57`
  - `MAPE = 2.66%`

Net yorum:

Bu model original one-step Table 3 kadar yuksek degil; bu normal cunku artik direct hourly lag avantajini kullanmiyor. Ama rolling operational Table 3'ten cok daha iyi. Ayrica final Table 8'den de daha iyi cikti.

Bu nedenle yeni medium-horizon model final Table 3 versiyonu olarak one cikti.

Feature importance tarafinda da en baskin aile `historical_profile` oldu. Bu da modelin gercekten zamani, aliskanliklari ve tekrar eden paternleri ogrendigini gosteriyor.

## 8. Final 2026 Forecast Generation

2026 forecast asamasinda iki resmi model mantigi korundu:

- Final Table 3: medium-horizon historical-profile XGBoost mantigi
- Final Table 8: non-recursive historical analog XGBoost mantigi

2026 icin exogenous girdilerde acik varsayimlar yapildi:

### Weather proxy

Gercek 2026 hava verisi olmadigi icin:

- `weighted_HDD`
- `weighted_CDD`

icin 2025'in ayni ay-gun-saat degerleri seasonal-naive proxy olarak ileri tasindi.

### Macro proxy

Table 8 tarafinda:

- `PMI_prev_month`
- `IR_prev_month`

icin 2025 aylik seviyeleri 2026'ya tasindi ve prev_month mantigi korundu.

2026 ozet rakamlar:

- Table 3 forecast dosyasi:
  - saatlik ortalama: `37999.81`
  - yillik toplam: `332878317.76`

- Table 8 final forecast dosyasi:
  - saatlik ortalama: `41222.16`
  - yillik toplam: `361106118.48`

Onemli not:
Mevcut `table3_forecast_2026.csv` dosyasi rolling operational workflow ile uretilmis son Table 3 forecast dosyasidir. Daha sonra model secimi acisindan final Table 3 versiyonu olarak medium-horizon historical-profile yapi one cikti. Dolayisiyla kavramsal final Table 3 modeli medium-horizon modeldir; fakat 2026 Table 3 forecast dosyasini bu son secime gore refresh etmek istersek ayrica bir final uretim adimi daha kosabiliriz.

## 9. Final Takeaways

Bu XGBoost surecinin sonunda iki ana model mantigi elimizde kaldi:

- Final Table 3:
  historical-profile temelli, non-recursive, medium-horizon model
- Final Table 8:
  historical analog ve previous-year profile temelli, non-recursive long-horizon model

Table 3 neyi temsil ediyor?
Kisa donem lag bagimliligini azaltip, zamansal hafizayi historical profile feature'lari ile koruyan bir yapiyi.

Table 8 neyi temsil ediyor?
Onceki yil benzerlikleri ve gecmis paternler uzerinden calisan, daha uzun ufukta metodolojik olarak cok temiz bir forecast yapisini.

Tradeoff su:

- En yuksek skor original one-step Table 3'teydi.
- En operasyonel gercekci kisa donem kullanim rolling tasarimda test edildi ama zayif kaldi.
- En dengeli ve tasinabilir Table 3 versiyonu medium-horizon historical-profile model oldu.
- En temiz ve guvenilir uzun donem model Table 8 final tasarimi oldu.

Kisacasi projede yalnizca "hangi model en yuksek R2 veriyor?" sorusuna bakmadik. Onun yerine su dengeyi kurmaya calistik:

- dogruluk
- operasyonel gercekcilik
- leakage guvenligi
- uzun ufuk forecasting robustness

Bu nedenle final mimari, yalnizca yuksek skor veren degil, ayni zamanda nasil yorumlanmasi gerektigi net olan iki model etrafinda kuruldu.

## 10. Important Files and Where They Are

Asagidaki dosyalar bu surecin en kritik artefact'leridir:

### Core data and weather construction

- Population-weighted HDD/CDD final series:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\reports\population_weighted_hourly_hdd_cdd.csv`

- Population-weighted methodology note:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\reports\population_weighted_methodology_note.txt`

### Main Table 8 model artifacts

- Final Table 8 metrics:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table8_nonrecursive_historical_population_weighted_metrics.json`

- Final Table 8 predictions for 2025:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table8_nonrecursive_historical_population_weighted_predictions_2025.csv`

- Final Table 8 methodology note:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table8_nonrecursive_historical_population_weighted_methodology_note.txt`

### Controlled historical design check outputs

- Previous-year vs multi-year design check table:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table8_historical_profile_design_check_2025.csv`

- Previous-year vs multi-year design check note:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table8_historical_profile_design_check_2025.txt`

Bu iki dosya, previous-year historical source ile multi-year grouped historical source arasindaki resmi kontrollu test sonuclarini icerir.

### Original and rolling Table 3 artifacts

- Original Table 3 one-step metrics:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table3_sarima_guided_population_weighted_metrics.json`

- Original Table 3 predictions:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table3_sarima_guided_population_weighted_predictions_2025.csv`

- Rolling backtest metrics:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table3_2025_rolling_backtest_metrics.csv`

- Rolling vs recursive comparison:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table3_2025_rolling_vs_recursive_comparison.csv`

### Final new Table 3 medium-horizon artifacts

- New medium-horizon metrics:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table3_medium_horizon_metrics.csv`

- New medium-horizon monthly validation:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table3_medium_horizon_monthly_validation.csv`

- Leakage audit:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table3_medium_horizon_leakage_audit.txt`

- Feature importance:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table3_medium_horizon_feature_importance.csv`

- Main interpretation note:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table3_medium_horizon_interpretation.txt`

Bu dosyalar, final Table 3 adayinin leakage-safe historical-profile tasarimini, aylik stabilitesini ve yorumunu birlikte verir.

### Hyperparameter / robustness outputs

- Overfitting audit:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\nonrecursive_historical_overfitting_audit.csv`

- Regularization study:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\nonrecursive_historical_regularization_study.csv`

- Monthly stability:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\nonrecursive_historical_monthly_stability.csv`

Bu dosyalar final Table 8 modelinde overfitting, tuning secimi ve robustness validation sonuclarini icerir.

### SHAP interpretability outputs

- SHAP summary plot:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\shap_summary_final_model.png`

- SHAP bar importance plot:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\shap_bar_final_model.png`

- Mean absolute SHAP values:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\shap_values_mean_abs_final_model.csv`

- SHAP dependence for weighted_CDD:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\shap_dependence_weighted_CDD.png`

- SHAP dependence for weighted_HDD:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\shap_dependence_weighted_HDD.png`

- SHAP dependence for historical profile features:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\shap_dependence_historical_profile.png`

- SHAP interpretation note:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\shap_final_model_interpretation.txt`

Bu dosyalar final secilmis modelin interpretability artefact'leridir; modelin hangi feature ailelerine dayandigini aciklamak icin kullanilir.

### 2026 forecast files

- Current Table 3 2026 forecast file:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table3_forecast_2026.csv`

- Table 3 2026 methodology note:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table3_forecast_2026_methodology_note.txt`

- Table 8 2026 forecast file:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table8_nonrecursive_historical_population_weighted_forecast_2026.csv`

- Table 8 2026 methodology note:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\table8_forecast_2026_methodology_note.txt`

- 2026 model comparison summary:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\xgboost\forecast_2026_model_comparison.txt`

## 11. Important Figures and How to Read Them

Bu kisi grafiklerden iyi anladigi icin, asagidaki figürler ozellikle yararlidir:

### Table 8 final 2025 diagnostics

- Actual vs Predicted:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\table8_nonrecursive_historical_population_weighted_actual_vs_predicted_2025.png`
  Bu grafik modelin 2025 boyunca gercek tuketim egrisinin ne kadarini takip edebildigini gosterir.

- Residual plot:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\table8_nonrecursive_historical_population_weighted_residuals_2025.png`
  Bu grafik hata zaman icinde belirli aylarda mi birikiyor, mevsimsel sorun var mi, onu gormek icin kullanilir.

### Rolling Table 3 comparison figures

- 2025 actual vs forecasts:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\table3_2025_actual_vs_forecasts.png`
  Burada actual 2025, full-year recursive ve rolling month-ahead cizgilerini ayni anda gorebilirsin.

- Monthly RMSE comparison:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\table3_2025_monthly_rmse_comparison.png`
  Hangi ayda hangi workflow daha iyi ya da daha kotu anlamak icin en pratik grafiklerden biri budur.

- Monthly error profile:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\table3_2025_monthly_error_profile.png`
  Aylik MAE/MAPE desenlerini karsilastirmak icin yararlidir.

- Residual comparison:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\table3_2025_residual_comparison.png`
  Iki farkli Table 3 workflow'un hata dagilimini birlikte gosterir.

### New medium-horizon Table 3 figures

- 2025 actual vs forecast:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\actual_vs_forecast_2025.png`
  Yeni modelin actual 2025'i nasil takip ettigini ve diger modellerle goreli farkini okumak icin kullanilir.

- Monthly RMSE plot:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\monthly_rmse_plot.png`
  Yeni modelin aylik stabilitesini original Table 3, rolling ve Table 8 ile kiyaslar.

- Residual plot:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\residual_plot.png`
  Yeni modelin hatalari belirli donemlerde yigilmis mi, surekli bias var mi, bunu okumaya yarar.

- Feature importance plot:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\feature_importance_plot.png`
  Modelin en cok hangi feature ailelerine dayandigini gorsel olarak anlamak icin en faydali grafiklerden biridir.

### 2026 final forecast figures

- Full-year forecast plot:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\forecast_2026_full_year_plot.png`
  Table 3 ve Table 8 forecast yollarinin yil boyunca nasil ayrildigini gosterir.

- Monthly profile plot:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\forecast_2026_monthly_profile_plot.png`
  Aylik ortalama forecast seviyelerini kiyaslamak icin kullanilir.

- Seasonal pattern plot:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\forecast_2026_seasonal_pattern_plot.png`
  Kis, ilkbahar, yaz, sonbahar icin saatlik pattern farklarini gosterir.

- Table 3 vs Table 8 comparison plot:
  `C:\Users\Monster\OneDrive\Masaüstü\proje2\outputs\figures\forecast_2026_table3_vs_table8_comparison_plot.png`
  Iki modelin forecast seviyelerini birbirine gore gormek icin hizli bir ozet sunar.
