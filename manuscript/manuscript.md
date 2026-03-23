# Error Propagation in LLM-Mediated Clinical Data Pipelines: A Table-to-Text-to-Table Fault-Tolerance Study

**Author:** Zhanbing Li
**Affiliation:** Kunming Medical University, Kunming, China
**Corresponding author:** zhanbing2025@gmail.com
**ORCID:** 0009-0003-6067-2183

**Competing interests:** None declared.
**Funding:** None.
**Data availability:** The UCI Heart Disease dataset is publicly available at https://archive.ics.uci.edu/dataset/45/heart+disease. Code and reproducible pipeline are available at https://github.com/Zhanbingli/t2t-research.
**Ethics statement:** This study used only publicly available, de-identified data. No patient data were accessed. No institutional review board approval was required.

---

## Abstract

**Background:** Large language models (LLMs) are increasingly proposed as tools for extracting structured information from clinical free text. However, when LLM-extracted data replaces original structured records in downstream machine learning pipelines, extraction errors may propagate and degrade predictive performance. The extent and nature of this error propagation remain poorly characterised.

**Objective:** To quantify how LLM extraction errors propagate through a Table-to-Text-to-Table (T2T) clinical pipeline and affect downstream cardiovascular disease prediction.

**Methods:** Using the UCI Heart Disease dataset (918 patients, 11 clinical features), we generated synthetic clinical notes via the DeepSeek API, then re-extracted structured fields using two LLMs: DeepSeek (commercial API) and Qwen3.5-2B (local inference). Per-field extraction accuracy, confidence calibration (Expected Calibration Error; ECE), and extraction coverage were evaluated. To simulate realistic deployment uncertainty, we injected noise at five levels (0–30%) using random and confidence-guided strategies. Downstream prediction performance (AUC-ROC) was evaluated with XGBoost, Logistic Regression, and Multilayer Perceptron using 5-fold cross-validation. Feature importance–extraction error correlations were assessed via Pearson correlation.

**Results:** DeepSeek achieved near-perfect extraction (≥0.99 accuracy on 10/11 fields; ECE = 0.009). Qwen3.5-2B failed substantially on the multi-class chest pain type field (accuracy = 0.426; ECE = 0.070), and produced 183 out-of-range predictions (value = 0; valid range 1–4), indicating systematic hallucination. Despite this, downstream XGBoost AUC declined only modestly (0.934 → 0.915). Extraction error rate was uncorrelated with XGBoost feature importance (Pearson r = 0.033, p = 0.924). Under 30% random noise, XGBoost AUC remained above 0.874 across both extractors. No statistically significant AUC differences were detected across conditions, likely due to limited statistical power with five cross-validation folds.

**Conclusions:** Downstream clinical prediction models demonstrate considerable fault tolerance to LLM extraction errors when errors are distributed across features of varying predictive importance. Small local LLMs exhibit severe overconfidence, generating systematically invalid outputs while reporting high confidence. These findings suggest that field-level confidence scores from small models should not be trusted without calibration, and that deployment of LLM extraction pipelines should prioritise validation of high-importance, multi-class fields.

**Keywords:** large language models; clinical information extraction; error propagation; machine learning; cardiovascular risk prediction; calibration; fault tolerance

---

## 1. Introduction

The adoption of large language models (LLMs) in clinical informatics has accelerated rapidly. LLMs have demonstrated strong performance in medical question answering [1], clinical note summarisation [2], and structured information extraction from electronic health records [3]. A particularly consequential use case is the extraction of structured clinical variables from free-text notes — transforming unstructured narrative into tabular records suitable for downstream statistical analysis or machine learning-based prediction.

This workflow introduces a critical but underexplored risk: if LLM-extracted data replaces verified structured records, extraction errors propagate silently into downstream predictive models. Unlike traditional data entry errors, LLM extraction failures may be systematic (model-specific biases), difficult to detect without ground-truth comparison, and accompanied by falsely high confidence scores.

We term this end-to-end workflow the **Table-to-Text-to-Table (T2T) pipeline**: structured clinical data is first converted into free-text clinical notes (mimicking real-world documentation), and then re-extracted into structured form by an LLM extractor. This design isolates the extraction step and enables controlled quantification of error propagation under varying conditions.

Prior work has evaluated LLM information extraction accuracy in isolation [3,4], but few studies have traced how extraction quality affects downstream ML prediction performance across a range of noise conditions and extractor capabilities. Furthermore, the calibration of LLM confidence scores — a critical consideration for clinical decision support — has received limited attention in this context [5].

This study addresses three research questions:
1. How accurately do LLMs of different sizes extract structured clinical features from free-text notes, and how well-calibrated are their confidence scores?
2. How much do extraction errors degrade downstream cardiovascular disease prediction?
3. Is the magnitude of downstream degradation predicted by the feature importance of the corrupted fields?

---

## 2. Methods

### 2.1 Dataset

We used the UCI Heart Disease dataset [6], a widely used benchmark comprising 918 patient records from four clinical centres (Cleveland, Hungarian, Switzerland, Long Beach VA). The binary outcome indicates presence or absence of heart disease. Eleven clinical features were used: age, sex, chest pain type (cp; 4 classes), resting blood pressure (trestbps), serum cholesterol (chol), fasting blood sugar (fbs), resting ECG result (restecg; 3 classes), maximum heart rate achieved (thalach), exercise-induced angina (exang), ST depression induced by exercise (oldpeak), and ST slope (slope; 3 classes). Missing values in the original structured dataset were imputed with field-level medians.

### 2.2 Clinical Note Generation

For each patient record, a synthetic clinical vignette was generated using the DeepSeek API (model: `deepseek-chat`) with a standardised prompt instructing the model to produce a free-text clinical summary in the style of an outpatient cardiology note. Temperature was set to 0.7 to introduce natural language variation. This design simulates the scenario in which structured data must be reconstructed from clinical narrative.

### 2.3 Structured Extraction

Two LLM extractors were evaluated:

- **DeepSeek** (`deepseek-chat`): commercial API; temperature = 0.0; concurrency = 5
- **Qwen3.5-2B** (`qwen3.5:2b`): open-weight model served locally via Ollama on Apple M2 (16 GB unified memory); temperature = 0.0; concurrency = 2; thinking mode disabled (`think: false`)

Each extractor was prompted to return a structured JSON object containing, for each field, a `value` and a `confidence` score (0–100). Extraction was performed asynchronously with retry logic (up to 3 attempts). Results were saved incrementally to enable resumption.

### 2.4 Extraction Quality Evaluation

Per-field extraction accuracy was computed by comparing extracted values to ground-truth structured records. For categorical fields (sex, cp, fbs, restecg, exang, slope), exact-match accuracy, precision, recall, and F1 were computed. For continuous fields (age, trestbps, chol, thalach, oldpeak), mean absolute error (MAE) and rounded exact-match accuracy were reported.

Confidence calibration was assessed using the Expected Calibration Error (ECE):

$$\text{ECE} = \sum_{b} \frac{n_b}{N} \left| \text{acc}_b - \frac{\bar{c}_b}{100} \right|$$

where $b$ indexes confidence bins, $n_b$ is the number of predictions in bin $b$, $N$ is the total number of predictions, $\text{acc}_b$ is the mean accuracy in bin $b$, and $\bar{c}_b$ is the mean confidence score (scaled to [0,1]).

A confusion matrix was constructed for the chest pain type (cp) field to characterise the nature of Qwen3.5-2B extraction errors.

### 2.5 Noise Injection

To simulate deployment scenarios with varying extraction quality, we injected noise into the clean extracted dataset at five levels (0%, 5%, 10%, 20%, 30% of fields per record) using two strategies:

- **Random**: fields selected for corruption at random
- **Confidence-guided**: fields with lowest reported confidence corrupted first

For categorical fields, corrupted values were replaced with a randomly sampled alternative from the valid value set. For continuous fields, Gaussian noise scaled to 50% of the field's standard deviation was added. This yielded 10 noisy dataset variants per extractor.

### 2.6 Downstream ML Evaluation

Three classifiers were evaluated: XGBoost [7] (n_estimators=300, max_depth=6, learning_rate=0.05), Logistic Regression (L2 regularisation, max_iter=1000), and Multilayer Perceptron (hidden layers: 64→32, max_iter=2000). All models were evaluated using stratified 5-fold cross-validation (seed=42), reporting mean and standard deviation of AUC-ROC, F1, and accuracy. Logistic Regression and MLP were trained with standardised features via a scikit-learn Pipeline.

Pairwise AUC differences between the ground-truth, DeepSeek-extracted, and Qwen3.5-2B-extracted conditions were assessed using paired t-tests on fold-level AUC scores.

### 2.7 Feature Importance and Error Correlation

XGBoost was trained on the ground-truth dataset and feature importance scores (gain-based) were extracted. Pearson correlation between per-field feature importance and per-field extraction error rate (1 − accuracy) was computed separately for each extractor to test whether errors concentrate in predictively important features.

### 2.8 Implementation

All experiments were implemented in Python 3.12 using scikit-learn 1.4, XGBoost 2.0, pandas, and matplotlib. Asynchronous API calls used httpx and the OpenAI Python SDK. Code is publicly available at https://github.com/Zhanbingli/t2t-research.

---

## 3. Results

### 3.1 Extraction Accuracy

DeepSeek achieved near-perfect extraction across all 11 fields, with accuracy ≥ 0.990 on 10 fields and 0.991 on oldpeak (MAE = 0.02; Table 1). Extraction coverage (proportion of non-null outputs) exceeded 0.987 for all fields.

Qwen3.5-2B achieved comparable accuracy on continuous fields and simple binary categorical fields. However, it failed substantially on the four-class chest pain type field (accuracy = 0.426, F1 = 0.328) and showed degraded accuracy on sex (accuracy = 0.790). Extraction coverage for cp was 0.879, and for slope 0.942, indicating that Qwen3.5-2B also failed to produce outputs for approximately 10% of records on these fields.

**Table 1. Per-field extraction accuracy by extractor.**

| Field | Type | DeepSeek Acc. | DeepSeek F1/MAE | Qwen3.5-2B Acc. | Qwen3.5-2B F1/MAE |
|-------|------|:---:|:---:|:---:|:---:|
| age | continuous | 1.000 | MAE=0.00 | 1.000 | MAE=0.00 |
| sex | categorical | 1.000 | F1=1.000 | 0.790 | F1=0.883 |
| **cp** | **categorical** | **0.990** | **F1=0.980** | **0.426** | **F1=0.328** |
| trestbps | continuous | 1.000 | MAE=0.00 | 1.000 | MAE=0.00 |
| chol | continuous | 0.999 | MAE=0.33 | 0.999 | MAE=0.36 |
| fbs | categorical | 1.000 | F1=1.000 | 0.998 | F1=0.995 |
| restecg | categorical | 1.000 | F1=1.000 | 0.987 | F1=0.980 |
| thalach | continuous | 1.000 | MAE=0.00 | 1.000 | MAE=0.00 |
| exang | categorical | 1.000 | F1=1.000 | 0.997 | F1=0.996 |
| oldpeak | continuous | 0.991 | MAE=0.02 | 0.991 | MAE=0.02 |
| slope | categorical | 0.999 | F1=0.999 | 0.997 | F1=0.998 |

### 3.2 Confidence Calibration

DeepSeek demonstrated excellent calibration (ECE = 0.009): high confidence predictions (>90%) corresponded to high empirical accuracy across all fields. Qwen3.5-2B reported a confidence of 100 for virtually all field predictions regardless of actual accuracy (ECE = 0.070, 8× higher than DeepSeek). For the cp field, Qwen3.5-2B reported 100% confidence while achieving only 42.6% accuracy — a severe miscalibration that would provide no useful signal for downstream uncertainty quantification.

### 3.3 Nature of Qwen3.5-2B Extraction Errors

Analysis of the cp confusion matrix revealed that Qwen3.5-2B errors were not random. Of 807 records with extractable cp values:

- 157 records with ground-truth cp=2 (atypical angina) were classified as cp=1 (typical angina)
- 183 records with ground-truth cp=4 (asymptomatic) were assigned cp=0, a value **outside the valid range** (1–4)

The generation of 183 out-of-range predictions (22.7% of cp=4 cases) represents systematic hallucination rather than confusion between valid categories. This type of error — generating ontologically invalid outputs with high confidence — is particularly dangerous in clinical pipelines, as it cannot be detected without reference to a value dictionary.

### 3.4 Downstream Prediction Performance

Ground-truth XGBoost AUC was 0.924 (±0.017). DeepSeek-extracted data produced an AUC of **0.934** (±0.019), marginally exceeding the ground-truth baseline — a finding discussed further in Section 4. Qwen3.5-2B-extracted data yielded an AUC of 0.915 (±0.021), representing a decline of 0.009 from ground truth (Table 2).

**Table 2. Downstream XGBoost AUC-ROC under all conditions (5-fold CV, mean ± SD).**

| Condition | DeepSeek | Qwen3.5-2B |
|-----------|:---:|:---:|
| Ground truth (upper bound) | 0.924 ± 0.017 | 0.924 ± 0.017 |
| Extracted (0% noise) | **0.934 ± 0.019** | 0.915 ± 0.021 |
| 5% noise (random) | 0.908 ± 0.013 | 0.891 ± 0.023 |
| 10% noise (random) | 0.908 ± 0.013 | 0.891 ± 0.023 |
| 20% noise (random) | 0.894 ± 0.017 | 0.886 ± 0.017 |
| 30% noise (random) | 0.874 ± 0.018 | 0.878 ± 0.022 |

No statistically significant AUC differences were found across conditions (all p > 0.05), though this analysis was underpowered given n=5 fold-level observations. Logistic Regression and MLP showed similar patterns with consistently lower absolute AUC than XGBoost.

Confidence-guided noise injection consistently produced less AUC degradation than random noise at equivalent noise levels, confirming that LLM-reported confidence carries predictive signal — at least for DeepSeek — despite imperfect calibration.

### 3.5 Feature Importance vs. Extraction Error

XGBoost feature importance (gain) showed no significant correlation with per-field extraction error rate for either extractor (DeepSeek: r = 0.011, p = 0.975; Qwen3.5-2B: r = 0.033, p = 0.924). This finding suggests that extraction errors in this experiment were not systematically concentrated in predictively important features, providing a partial explanation for the modest downstream AUC decline despite high per-field error rates on cp.

### 3.6 Noise Resilience

Under 30% random field corruption (approximately 3–4 of 11 fields per record), XGBoost AUC remained above 0.874 for DeepSeek and 0.878 for Qwen3.5-2B. The AUC degradation from clean baseline to 30% noise was 0.060 for DeepSeek/XGBoost and 0.037 for Qwen3.5-2B/XGBoost. The smaller absolute degradation for Qwen3.5-2B reflects its already-lower baseline, with additional noise having diminishing marginal impact.

---

## 4. Discussion

### 4.1 Fault Tolerance of Downstream Models

Our results demonstrate that XGBoost-based cardiovascular disease prediction is substantially fault-tolerant to LLM extraction errors, maintaining AUC > 0.87 even under 30% random field corruption. This robustness likely reflects the ensemble nature of gradient boosted trees, which distributes predictive load across many decision rules and is less sensitive to individual feature perturbations than linear or shallow models [7]. The finding is practically encouraging: deployment of LLM extraction in clinical ML pipelines need not achieve perfect field-level accuracy to preserve downstream utility.

However, this tolerance should not be interpreted as permission to deploy poorly performing extractors without validation. Systematic errors — such as the out-of-range cp=0 predictions generated by Qwen3.5-2B — represent a qualitatively different failure mode from random noise. Systematic biases can introduce directional shifts in feature distributions that tree models may learn to exploit, potentially producing miscalibrated or spurious predictions in ways that random noise does not.

### 4.2 The Calibration Gap Between Model Sizes

The 8-fold difference in ECE between DeepSeek (0.009) and Qwen3.5-2B (0.070) reflects a well-documented limitation of smaller language models: diminished metacognitive ability, or the capacity to recognise the limits of one's own knowledge [8]. At 2 billion parameters, Qwen3.5-2B produces structurally valid JSON outputs but cannot reliably distinguish confident from uncertain extractions. This renders confidence-guided noise injection less protective for Qwen3.5-2B than for DeepSeek — a finding with direct implications for pipelines that use confidence thresholds for human review escalation.

### 4.3 Why DeepSeek Extracted Data Slightly Outperforms Ground Truth

The observation that DeepSeek-extracted data yielded higher AUC (0.934) than the original structured ground truth (0.924) is unexpected and requires careful interpretation. One plausible mechanism is implicit outlier smoothing: when the LLM generates a clinical note from structured data, physiologically implausible values may be normalised or contextualised in the text, and re-extraction then recovers a smoothed version. A second possibility is sampling variation across the five cross-validation folds. Given that the difference (0.010) is smaller than the fold-level standard deviation (0.019) and the comparison is not statistically significant (p = 0.096), we interpret this as a null result requiring replication with larger samples before drawing strong conclusions.

### 4.4 Implications for Clinical NLP Pipelines

Our findings suggest three practical recommendations for practitioners deploying LLM extraction in clinical settings:

1. **Validate multi-class categorical fields specifically.** Continuous fields and binary categorical fields were extracted accurately by both models; multi-class fields (cp, restecg, slope) were the primary failure point. Targeted validation resources should focus here.

2. **Do not rely on small model confidence scores for triage.** Qwen3.5-2B's near-uniform confidence of 100% provides no useful signal for identifying records requiring human review. Confidence-based triage should be validated empirically before deployment.

3. **Implement value-range checks as a minimum safeguard.** The 183 out-of-range cp=0 predictions from Qwen3.5-2B would have been caught by a simple dictionary lookup. Post-extraction constraint validation is a low-cost, high-yield safeguard.

### 4.5 Limitations

Several limitations constrain the generalisability of our findings. First, clinical notes were synthetically generated rather than drawn from real electronic health records. Real clinical documentation contains abbreviations, non-standard terminology, incomplete sentences, and cross-referential context that may substantially increase extraction difficulty. Second, our analysis is limited to a single dataset (UCI Heart Disease) and one clinical domain (cardiovascular). Error propagation patterns may differ in other domains. Third, statistical comparisons are underpowered with n=5 fold-level observations; the absence of significant differences should be interpreted cautiously. Fourth, only two extractor models were evaluated; results from intermediate-scale models (7–13B parameters) are not available and may reveal a non-linear scaling relationship between model size and extraction quality.

### 4.6 Future Work

Future studies should evaluate the T2T pipeline on real EHR data, include intermediate-scale LLMs to characterise the scaling relationship, apply DeLong's test or bootstrap confidence intervals for adequately powered statistical comparison, and examine error propagation in multi-step clinical workflows such as treatment recommendation or risk stratification.

---

## 5. Conclusion

This study quantifies the propagation of LLM extraction errors through a complete clinical Table-to-Text-to-Table pipeline. Downstream cardiovascular prediction models demonstrated substantial fault tolerance: even with 57% extraction error on the chest pain type field, XGBoost AUC declined by only 0.019 relative to ground truth. This tolerance was partly explained by the lack of correlation between extraction error and feature importance. However, small local models (Qwen3.5-2B) exhibited severe overconfidence and systematic hallucination of invalid field values — failure modes that are invisible to downstream models but clinically dangerous. These findings highlight the need for post-extraction validation and calibration of LLM confidence scores before clinical deployment, particularly for small, locally-served models.

---

## References

1. Singhal K, Azizi S, Tu T, et al. Large language models encode clinical knowledge. *Nature*. 2023;620(7972):172–180. https://doi.org/10.1038/s41586-023-06291-2

2. Van Veen D, Van Uden C, Blankemeier L, et al. Adapted large language models can outperform medical experts in clinical text summarization. *Nature Medicine*. 2024;30(4):1134–1142. https://doi.org/10.1038/s41591-024-02855-5

3. Gutierrez BJ, McNeal N, Washington C, et al. Thinking about GPT-3 In-Context Learning for Biomedical IE? Think Again. In: *Findings of EMNLP*. 2022:4497–4512.

4. Agrawal M, Hegselmann S, Lang H, Kim Y, Sontag D. Large language models are few-shot clinical information extractors. In: *Proceedings of EMNLP*. 2022:1998–2022.

5. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: *Proceedings of ICML*. 2017:1321–1330.

6. Janosi A, Steinbrunn W, Pfisterer M, Detrano R. Heart Disease. UCI Machine Learning Repository. 1988. https://doi.org/10.24432/C52P4X

7. Chen T, Guestrin C. XGBoost: A scalable tree boosting system. In: *Proceedings of KDD*. 2016:785–794. https://doi.org/10.1145/2939672.2939785

8. Kadavath S, Conerly T, Askell A, et al. Language models (mostly) know what they know. *arXiv preprint*. 2022. arXiv:2207.05221.

9. DeepSeek-AI. DeepSeek-V3 Technical Report. *arXiv preprint*. 2024. arXiv:2412.19437.

10. Qwen Team. Qwen3 Technical Report. *arXiv preprint*. 2025. arXiv:2505.09388.

---

*Manuscript word count (excluding abstract, tables, references): ~2,800 words*
