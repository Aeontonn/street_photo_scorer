# Rapport — Street Photo Scorer
Anteckningar och viktiga punkter att ta med i rapporten.

---

## 1. Projektbeskrivning

**Problemformulering:** Kan en maskininlärningsmodell lära sig vad som gör ett gatufoto estetiskt tilltalande, baserat på communityns upvotes som proxy för kvalitet?

**Pipeline:**
1. Scraping av ~1434 bilder + metadata från r/streetphotography (2021–2024) via Arctic Shift
2. Feature-extraktion med CLIP (OpenAI) → 512-dimensionella embeddings per bild
3. Unsupervised klustring: UMAP (dimensionsreduktion) + KMeans (stilkluster)
4. Supervised regression: förutsäga log(upvotes) från embeddings
5. Deployment: Streamlit-app

---

## 2. Data & förberedelse

**Datakälla:** r/streetphotography via Arctic Shift-arkivet (ingen API-nyckel krävs)

**Insamlat:**
- 2000 posts hämtades, ~1434 bilder laddades ner (~28% bortfall pga raderade/utgångna bilder)
- Metadata: score (upvotes), upvote_ratio, num_comments, created_utc, author

**Target-variabel:** `log1p(score)` — log-transformerade upvotes
- Motivering: score-fördelningen följer en power-law (de flesta foton har lågt score, ett fåtal extremt högt). Log-transform gör distributionen mer normalliknande och stabiliserar träningen.
- Score-range i träningsdata: log1p(1) ≈ 0.69 till log1p(3270) ≈ 8.09

**Bortfallsanalys:**
- ~28% av bilderna var raderade eller otillgängliga
- Nyare posts (2024+) hade högre bortfall — bilder raderas fortare än metadata
- Lösning: skrapade posts från 2021–2024 för stabilare bildhosting

---

## 3. EDA (Exploratory Data Analysis)

**Viktiga observationer:**
- Score-distribution är kraftigt högersneddad (power-law)
- Median score: ~10–15 upvotes; medelvärde betydligt högre pga outliers
- log1p(score) ser mer normalfördelad ut → motiverar valet av target
- Korrelation score ↔ num_comments: positiv men inte perfekt (kommentarer ≠ gillanden)
- Author-bias: kolla om ett fåtal författare dominerar höga scores

**Att nämna:** subredditen r/streetphotography har ~500k+ subscribers och en specifik smakprofil. Det är inte ett representativt urval av "all gatufotografi".

---

## 4. Feature-extraktion — CLIP

**Modell:** CLIP ViT-B/32 (OpenAI, via HuggingFace)
- Tränad på 400 miljoner bild-text-par
- Extraherar 512-dimensionella embeddings som kodifierar semantiskt och visuellt innehåll
- Embeddingarna fångar: motiv, komposition, ljussättning, stil, färgpalett

**Motivering:** CLIP är överlägsen handkodade features (histogram, kanter osv) eftersom den förstår visuellt innehåll på en semantisk nivå, inte bara pixelnivå.

**Normalisering:** embeddings normaliseras till enhetslängd (L2-norm) för att bara mäta riktning, inte magnitud — standard i CLIP-litteraturen.

---

## 5. Unsupervised Learning — Klustring

**Pipeline:**
1. UMAP: 512D → 3D (n_neighbors=15, min_dist=0.1)
2. KMeans: 8 kluster (experimenterat med 6–12)

**Klusterresultat (fyll i efter du tittat på bilderna):**

| Kluster | Beskrivning | Medel-score |
|---------|-------------|-------------|
| 0 | ??? | 59.6 |
| 3 | ??? | 55.0 |
| 2 | ??? | 52.9 |
| 7 | ??? | 50.4 |
| 5 | ??? | 47.4 |
| 6 | ??? | 36.7 |
| 4 | ??? | 35.9 |
| 1 | ??? | 35.8 |

**Viktig observation:** Kluster 0 och 3 har nästan dubbelt så högt medel-score som kluster 1 och 4. Det finns alltså visuella stilar som premieras signifikant mer av communityn — diskutera vilka och varför.

**Diskussion:** UMAP är icke-linjär och bevarar lokal struktur bättre än PCA. KMeans antar sfäriska kluster — HDBSCAN är ett alternativ som hittar godtyckliga former och hanterar brus (outlier-foton som inte passar något kluster).

---

## 6. Supervised Learning — Scoring-modell

**Modeller tränade:**
- Ridge Regression (linjär, regulariserad)
- Random Forest (ensemble, icke-linjär)
- Gradient Boosting (ensemble, sekventiell)

**Utvärdering:**
- Bästa modell valdes baserat på R² på testset (80/20-split)
- Förväntat R²: 0.05–0.25 (lågt är normalt för subjektiv estetik)

**Varför lågt R²?**
- Upvotes påverkas av faktorer utanför bilden: tidpunkt för publicering, author-popularitet, trending topics, rubriken på posten
- Estetik är subjektivt och kontextberoende — CLIP fångar visuellt innehåll men inte social kontext
- Modellen lär sig genomsnittlig community-smak, inte individuell estetik

**Score-normalisering (viktig för rapporten):**
Ursprunglig linjär normalisering gav systematiskt låga scores eftersom distributionen är snedfördelad. Lösning: percentilbaserad normalisering — score = "bättre än X% av träningsbilderna". Det ger en mer intuitiv och rättvis skala.

---

## 7. Deployment — Streamlit

- Användaren laddar upp en valfri bild
- CLIP extraherar embedding
- Modellen predictar log(score)
- Percentilbaserat score (0–10) beräknas mot träningsdistributionen
- Visar "bättre än X% av Y foton i träningsdatasetet"

---

## 8. Diskussion — Subjektivitet & Bias

**Upvotes som proxy — problem:**

1. **Popularitetsbias:** Posts med många upvotes syns högre upp → får fler upvotes. Tidiga upvotes är oproportionerligt viktiga.

2. **Author-bias:** Kända fotografer i communityn får fler upvotes oavsett bildkvalitet. Modellen riskerar att lära sig "vem" snarare än "vad".

3. **Temporal bias:** Stilar som var trendiga 2021–2024 premieras. En bild från 1950-talet av Henri Cartier-Bresson scorar lågt inte för att den är dålig, utan för att den ser annorlunda ut mot vad moderna Reddit-användare röstar upp.

4. **Community-specifik smak:** r/streetphotography ≠ universell estetisk kvalitet. Andra communities (Flickr, 500px, professionella fototävlingar) skulle ge andra signaler.

5. **Selektionseffekt:** Fotografer som postar på Reddit är inte representativa för alla gatufotografer.

**Vad modellen faktiskt lär sig:**
"Vad ser ut som foton som brukade få upvotes på r/streetphotography 2021–2024" — inte "vad är estetiskt vackert".

5. **Människobias:** Empiriskt observerat under testning: bilder med människor scorer konsekvent högre än bilder utan. CLIP-embeddings för bilder med människor liknar troligen mer de högt scorade träningsfotona, eftersom gatufotografi på Reddit domineras av porträtt och candid-shots. Rent arkitektur- eller landskapsfotografi i stadsmiljö straffas av modellen trots att det är legitim gatufotografi. Det är ett tydligt exempel på hur datasetet formar modellens definition av "kvalitet".

**Intressant vinkel för presentationen:** testa en känd, prisbelönad gatufoto (t.ex. Cartier-Bresson, Vivian Maier) mot ett typiskt Reddit-populärt foto. Diskutera skillnaden i score och vad det säger om modellens begränsningar.

---

## 9. Kurskrav — checklista

| Krav | Status |
|------|--------|
| Dataförberedelse (saknade värden, kategoriska variabler) | ✅ bortfall hanterat, log-transform |
| EDA med visualiseringar | ✅ notebook 01 |
| Unsupervised learning | ✅ UMAP + KMeans |
| Dimensionsreduktion | ✅ UMAP (nämn också PCA som alternativ) |
| Supervised learning (ej bara linjär regression) | ✅ RF + GBM |
| Deep learning / neural networks | ✅ CLIP (förtränad) |
| Utvärdering (confusion matrix, AUC etc.) | ⚠️ regression → använd R², MAE, residualplot |
| Deployment / interaktiv visualisering | ✅ Streamlit + Plotly |

**OBS om utvärdering:** Kursen nämner confusion matrix och AUC-ROC, men det är klassifikationsmetrik. Ditt problem är regression → använd istället:
- R² (förklaringsgrad)
- MAE (medelabsolutfel)
- Residualplot (predicted vs actual)
- Jämförelsetabell över de tre modellerna

Om du vill ha klassifikation kan du binarisera target: `hög kvalitet = score > median` och träna en klassifierare. Nämn det som ett alternativt angreppssätt.
