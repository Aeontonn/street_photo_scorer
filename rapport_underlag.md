# Rapport — Street Photo Scorer

---

## 1. Projektbeskrivning

**Problemformulering:** Kan en maskininlärningsmodell lära sig vad som gör ett gatufoto estetiskt tilltalande, baserat på communityns upvotes som proxy för kvalitet?

**Pipeline:**

1. Scraping av 22 374 bilder + metadata från r/streetphotography via Arctic Shift
2. Feature-extraktion med CLIP ViT-B/32 (OpenAI) → 512-dimensionella embeddings per bild
3. Unsupervised klustring: UMAP (dimensionsreduktion, 512D → 3D) + KMeans (8 stilkluster)
4. Supervised regression: förutsäga log1p(upvotes) från embeddings
5. Binär klassificering: hög/låg kvalitet (över/under median)
6. Deployment: Streamlit-app med fyra flikar — Score & Style, Similar Photos, Style Map, Technical Analysis

---

## 2. Data & förberedelse

**Datakälla:** r/streetphotography via Arctic Shift-arkivet (ingen API-nyckel krävs, alla Reddit-poster är publik historik)

**Insamlat:**

- 22 375 posts hämtades via Arctic Shift API, 22 374 bilder laddades ner
- Tidsperiod: juli 2020 – december 2023
- Metadata per post: score (upvotes), upvote_ratio, num_comments, created_utc, author

**Target-variabel:** `log1p(score)` — log-transformerade upvotes

- Motivering: score-fördelningen följer en power-law (median 6 upvotes, medelvärde 56.7 — ett fåtal posts drar upp snittet kraftigt). Log-transform gör distributionen mer normalliknande och stabiliserar träningen.
- Score-range i rådata: 0 – 3 267 upvotes
- log1p-range: 0.0 – 8.09

**Bortfallsanalys:**

- 22 375 posts hämtades, 22 374 bilder laddades ner framgångsrikt (minimalt bortfall)
- Enstaka korrumperade PNG-filer hanterades via try/except i embedding-steget och hoppades över
- Arctic Shift-arkivet innehåller alla publika Reddit-poster oavsett om originalet är raderat, vilket minimerar bortfall jämfört med att scrapa Reddit direkt

---

## 3. EDA (Exploratory Data Analysis)

**Viktiga observationer:**

- Score-distribution är kraftigt högersneddad (power-law): median 6 upvotes, medelvärde 56.7 — ett litet antal virala poster drar upp snittet kraftigt
- log1p(score) ser mer normalfördelad ut → motiverar valet av target
- Positiv korrelation score ↔ num_comments men inte perfekt
- Kluster med människor i bild har konsekvent högre medel-score än kluster med ren arkitektur eller landskap

**Subreddit-kontext:** r/streetphotography har >500k subscribers och en specifik smakprofil. Resultaten är inte generaliserbara till all fotografi.

---

## 4. Feature-extraktion — CLIP

**Modell:** CLIP ViT-B/32 (OpenAI, via HuggingFace Transformers)

- Tränad på 400 miljoner bild-text-par
- Extraherar 512-dimensionella embeddings som kodifierar semantiskt och visuellt innehåll
- Embeddingarna fångar: motiv, komposition, ljussättning, stil, färgpalett, genre

**Motivering:** CLIP är överlägsen handkodade features (histogram, kanter osv) eftersom den förstår visuellt innehåll på semantisk nivå, inte bara pixelnivå. CLIP kan skilja på "foto av människor i stadsmiljö" och "arkitekturfoto" utan explicit objektklassificering.

**Normalisering:** Embeddings normaliseras till enhetslängd (L2-norm) — standard i CLIP-litteraturen, gör kosinuslikhet till rätt avståndsmått.

**Minnesoptimering:** Embeddings lagras som float16 (halverad RAM-användning) och konverteras till float32 vid beräkning. För 22 374 embeddings à 512 dimensioner ger detta ca 22 MB i RAM istället för 44 MB.

---

## 5. Unsupervised Learning — Klustring

**Pipeline:**

1. UMAP: 512D → 3D (n_neighbors=15, min_dist=0.1, metric='cosine')
2. KMeans: 8 kluster (experimenterat med 6–12)

**Klusterresultat (22 374 foton, 8 kluster):**

| Kluster | n foton | Medel-score (upvotes) |
| ------- | ------- | --------------------- |
| 4       | 1 891   | 75.8                  |
| 6       | 3 005   | 67.4                  |
| 0       | 2 300   | 56.5                  |
| 2       | 2 910   | 54.4                  |
| 1       | 1 520   | 54.1                  |
| 5       | 2 477   | 54.0                  |
| 3       | 5 027   | 51.1                  |
| 7       | 3 244   | 49.6                  |

_(Klusternamnen fylls i efter inspektion av notebook 02)_

**Viktig observation:** Kluster 4 och 6 har ca 50% högre medel-score än kluster 3 och 7. Det finns alltså tydliga visuella stilar som premieras av communityn. Kluster 3 är också störst (5 027 foton) vilket tyder på att det fångar en "genomsnittlig" Reddit-stil med lägre engagement.

**Diskussion:** UMAP är icke-linjär och bevarar lokal struktur bättre än PCA. KMeans antar sfäriska kluster — HDBSCAN är ett alternativ som hittar godtyckliga former och hanterar brus.

**Minneshantering i appen:** Den tränade UMAP-modellen (umap_reducer.pkl) visade sig kräva för mycket RAM för att köra transform() på uppladdade bilder. Lösning: kluster tilldelas via kosinuslikhet till de förberäknade centroiderna, och UMAP-positionen för uppladdade bilder approximeras med närmaste grannens kända 3D-koordinater.

---

## 6. Supervised Learning — Scoring-modell

**Modeller tränade (regression, target: log1p(score)):**

- Ridge Regression (linjär, regulariserad, Pipeline med StandardScaler)
- Random Forest (200 träd, n_jobs=-1)
- Gradient Boosting (100 estimators, learning_rate=0.1, subsample=0.8)

**Bästa regressionsmodell:** Ridge Regression (sparad som scorer.pkl, Pipeline med StandardScaler)

**Faktiska utvärderingsvärden (testset, 80/20-split):**

- R² = 0.104
- MAE = 1.17 log1p-enheter

**Binär klassificering (hög/låg kvalitet = över/under median log-score ≈ 5 upvotes):**

- Logistic Regression (Pipeline med StandardScaler)
- Random Forest (200 träd, n_jobs=-1)
- Gradient Boosting (100 estimators)

**Bästa klassifierare:** Logistic Regression

**Faktiskt AUC-ROC = 0.50** — i praktiken samma som att gissa slumpmässigt. Det är ett viktigt resultat: binär klassificering av "hög vs låg kvalitet" baserat enbart på visuella CLIP-embeddings fungerar inte bättre än slump.

**Varför så dåliga resultat?**

- Upvotes påverkas av faktorer utanför bilden: tidpunkt för publicering, author-popularitet, trending topics, titeln på posten
- Estetik är subjektivt — CLIP fångar visuellt innehåll men inte social kontext
- Klassificeringsgränsen (median = 5 upvotes) delar datasetet vid ett mycket lågt värde, vilket gör att brus i upvote-räkning dominerar
- R² = 0.10 är ändå meningsfullt — det visar att visuell stil har _viss_ förutsägbarhet, men att den sociala kontexten är avgörande

**Score-normalisering:**
Percentilbaserad normalisering: score = "bättre än X% av de 22 374 träningsbilderna". Det ger en mer intuitiv och rättvis skala än linjär normalisering (som gav systematiskt låga scores pga power-law-fördelningen).

---

## 7. Deployment — Streamlit

**Fyra flikar:**

- **Score & Style:** Stort poängnummer, AI-verdict (hög/låg kvalitet), visuella stiletiketter från CLIP, stilkluster med förklaring
- **Similar Photos:** 6 visuellt närmaste foton från träningsdatasetet (kosinuslikhet i embeddingrum)
- **Visual Style Map:** 3D-scatter i Plotly med alla 22 374 träningsfoton färgkodade per kluster
- **Technical Analysis:** Kompositionsöverlägg, tekniska mätvärden, stilgenre-diagram, dominanta färger

**CLIP används i appen för:**

1. Embedding av uppladdad bild → scoring och klustertilldelning
2. Kosinuslikhet mot textbeskrivningar → visuella attribut (svartvit vs färg, ljus vs mörkt, osv.)
3. Kosinuslikhet mot genrebeskrivningar → fotografi-genre (candid, fine art B&W, urban landscape, osv.)

---

## 8. Tekniska utmaningar och lösningar

Under projektets gång uppstod ett antal oväntade tekniska problem som krävde lösningar och omändringar av arkitekturen.

### 8.1 Minnesproblem med sklearn NearestNeighbors

**Problem:** Den ursprungliga implementationen använde sklearn NearestNeighbors med BallTree för att hitta liknande bilder. Med 22 374 embeddings à 512 dimensioner konsumerade BallTree >4 GB RAM, vilket orsakade att Streamlit-processen slog ihjäl sig (OOM-killed).

**Lösning:** Bytte till chunked kosinuslikhet med numpy — beräknar likhet i block om 2000 embeddings åt gången. Embeddings lagras som float16 och konverteras till float32 per chunk. Minnespeak reducerades till <500 MB.

### 8.2 UMAP transform() OOM-krasch

**Problem:** umap_reducer.transform() på en uppladdad bild krävde att hela UMAP-modellen hölls i RAM och exekverade en tung beräkning, vilket orsakade minneskrascher.

**Lösning:** UMAP-modellen laddas inte in i appen alls. Kluster tilldelas via kosinuslikhet till förberäknade centroids. UMAP-positionen för den uppladdade bilden approximeras med närmaste grannes kända 3D-koordinater (acceptabel approximation för visualiseringsändamålet).

### 8.3 Segmentation fault i OpenCV

**Problem:** Det tekniska analyssteget använde OpenCV (cv2) för bildanalys. På Python 3.13 / macOS ARM (Apple Silicon) orsakade OpenCV:s Haar cascade (`detectMultiScale`) och Hough-transformens `HoughLinesP` konsekvent en segmentation fault, vilket kraschade hela Streamlit-processen.

**Lösning steg 1:** Tog bort Haar face cascade. Ansiktsdetektering ersattes med en saliensproxy (ljusaste regionen i bilden), vilket är mer robust för gatufotografi där motiv ofta är baklysta eller i rörelse.

**Lösning steg 2 (slutgiltig):** Bytte ut hela OpenCV-beroendet mot scipy/numpy/PIL. Laplacian variance beräknas med `scipy.ndimage.laplace()`, Gaussian blur med `scipy.ndimage.gaussian_filter()`, histogram med numpy, och kompositionsöverlägg ritas med `PIL.ImageDraw`. Hough-transformens funktioner (ledande linjer, horisontdetektering) togs bort från appen.

### 8.4 Segmentation fault i sklearn KMeans (loky-backend)

**Problem:** Dominanta färger extraherades med `sklearn.cluster.KMeans(n_init=5)`. sklearn använder joblib/loky för att parallellisera de 5 initialiseringarna. På Python 3.13 / macOS ARM kraschade loky med en segmentation fault.

**Symptom:** Felmeddelandet `/loky-{PID}-{random}.` i terminalen vid krasch.

**Lösning:** Bytte ut sklearn KMeans mot PIL:s inbyggda median-cut färgkvantisering (`Image.quantize(colors=n, method=Image.Quantize.MEDIANCUT)`). Ger likvärdiga dominanta färger utan sklearn/loky.

### 8.5 Korrumperade bilder i datasetet

**Problem:** Bland de 22 374 nedladdade bilderna fanns ett antal korrumperade PNG-filer som orsakade `PIL.UnidentifiedImageError` under embedding-extraktion.

**Lösning:** Lade till try/except i batch-loopen i `extract_embeddings.py` — korrumperade bilder hoppas automatiskt över utan att avbryta körningen.

### 8.6 Scraper-timeout vid stor insamling

**Problem:** Vid insamling av 22 374 bilder (körtid >2 timmar) uppstod periodvisa nätverksfel mot Arctic Shift API:t.

**Lösning:** Lade till exponentiell backoff (5s → 10s → 20s → ... → 120s, max 6 försök) och ett checkpoint-system som sparar progress var 500:e post. Vid avbrott kan scraping återupptas från senaste checkpoint.

---

## 9. Diskussion — Subjektivitet & Bias

**Upvotes som proxy — problem:**

1. **Popularitetsbias:** Posts med tidiga upvotes syns högre upp → får fler upvotes. Nätverkseffekter förstärker skillnader.

2. **Author-bias:** Kända fotografer i communityn får fler upvotes oavsett bildkvalitet. Modellen riskerar att lära sig "vem" snarare än "vad".

3. **Temporal bias:** Stilar trendiga 2020–2024 premieras. En bild från 1950-talet av Henri Cartier-Bresson scorar lågt — inte för att den är dålig, utan för att den ser annorlunda ut mot vad moderna Reddit-användare röstar upp.

4. **Community-specifik smak:** r/streetphotography ≠ universell estetisk kvalitet. Andra communities (Flickr, 500px, professionella fototävlingar) skulle ge andra signaler.

5. **Människobias:** Empiriskt observerat: bilder med människor scorer konsekvent högre. CLIP-embeddings för bilder med människor liknar mer de högt scorade träningsfotona eftersom communityn domineras av candid-shots och porträtt. Ren arkitektur i stadsmiljö straffas av modellen — ett tydligt exempel på hur datasetet formar definitionen av "kvalitet".

**Vad modellen faktiskt lär sig:**
"Vad ser ut som foton som brukade få upvotes på r/streetphotography 2020–2024" — inte "vad är estetiskt vackert".

---

## 10. Kurskrav — checklista

| Krav                                            | Status                                                       |
| ----------------------------------------------- | ------------------------------------------------------------ |
| Dataförberedelse (saknade värden, bortfall)     | ✅ Bortfall hanterat, log-transform, checkpoint-system       |
| EDA med visualiseringar                         | ✅ Notebook 01 — score-distribution, klusteröversikt         |
| Unsupervised learning                           | ✅ UMAP + KMeans (8 kluster)                                 |
| Dimensionsreduktion                             | ✅ UMAP 512D → 3D (nämn PCA som alternativ)                  |
| Supervised learning (ej bara linjär regression) | ✅ Random Forest + Gradient Boosting                         |
| Deep learning / neural networks                 | ✅ CLIP ViT-B/32 (förtränad, fine-tuning ej nödvändig)       |
| Binär klassificering med AUC-ROC                | ✅ Notebook 04 — confusion matrix, AUC-ROC, precision-recall |
| Deployment / interaktiv visualisering           | ✅ Streamlit + Plotly 3D                                     |

**Mätvärden för regression:**

- R² = 0.104 (Ridge Regression på testset)
- MAE = 1.17 log1p-enheter
- Jämförelsetabell: Ridge vs RF vs GBM

**Mätvärden för klassificering:**

- AUC-ROC = 0.50 (Logistic Regression — i praktiken slumpmässigt)
- Precision/Recall
- Confusion matrix
