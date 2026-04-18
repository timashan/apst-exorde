**MAS 5313: Group Project Proposal \- Batch 2025**

* **Information regarding the project:**

|  (i) Title of the Project:   | Analysis of Global Social Media Discourse: Sentiment, Emotion, and Thematic Trends Across Platforms and Languages |
| ----- | :---- |
|       **(ii) Group No:** | **No: 04** |
|  **(iii) Group Members:**  | 2025/APST/14 \- Kavishka Timashan |
|  | 2025/APST/10 \- Dilshan Anurada |
|  | 2025/APST/13 \- Jerome Clifton |
|  | 2025/APST/29 \- Anuradha Rathnayake |
|  | 2025/APST/31 \- Dulsara Kahapolage |

* **Introduction:** 

In the highly connected world today, different social media platforms, blogs, and news together represent one of the major channels for public discourse. Understanding how information, sentiment, themes, and emotional tone arise and then spread across platforms and languages can yield insights relevant to social scientists, communications researchers, policy analysts, and data scientists. The Exorde dataset is a massive, multi-source (X, Reddit, etc), multi-language snapshot of global online communication (here: one calendar week in December 2024), offering a unique opportunity to systematically study global discourse at an unprecedented scale. Its rich metadata makes it particularly well-suited for quantitative and computational social science. This project aims to leverage that resource to answer fundamental questions about how topics and emotions evolve over time and spread across platforms, and to model the dynamics of cross-platform diffusion of discourse.

* **Objectives:**  
1. **Characterise Global Discourse Volume & Temporal Dynamics.**  
2. **Obtain Thematic Distribution and Evolution over time.**   
3. **Perform a Sentiment and Emotion Analysis Across Themes and Languages.**  
4. **Perform Cross-Platform and Cross-Language Comparison**  
5. **Identify Sentiment Variation Over Time and Themes**

* **About Data:** 

Dataset: Exorde Social Media December 2024 Week 1\.   
Time span: 1 December 2024 – 7 December 2024\.   
Total entries: \~ 269,403,210 posts/articles, Multi-source posts from 6000+ platforms  
Languages covered: 122 languages.   
Metadata fields (per post):

* timestamp (exact date/time)   
* original text, URL,   
* language (detected automatically) primary theme (categorical, e.g. politics, health, economy, etc.)   
* English keywords (from translated text) are useful for cross-language analysis and topic matching.   
* sentiment score (numeric, range from \-1 to \+1)   
* Main emotion label (categorical)   
* Secondary themes (list, possibly multiple themes per post) 

* **Analysis and Methodology:**   
* **Data Processing & Cleaning**  
  * Has approximately 269,403,210 rows  
  * Improving and validating the integrity of the data by filtering outliers, handling null values, and incomplete data, such as missing sentiment scores.  
  * Identifying potential duplications or near duplicates  
* **Handling Multilingual Data:**  
  * Ensure that posts from all languages are handled appropriately.  
* **Feature Engineering**  
  * New variables, such as platform, can be derived from the URL column  
  * Deriving time-based features (eg, day of the week, hour of the day)  
  * Convert categorical string variables into label encodings for analysis  
* **Sampling and Dataset Balancing Strategy**  
  * Directly working with all 270 million posts is computationally impractical  
  * Construct a balanced analytical subset by sampling 0.05% of the dataset using stratified random sampling, with source (derived from URL) and primary theme serving as the stratification variables.

**Statistical Analysis Plan**  
The statistical analysis techniques will be employed on the sampled data and the aggregated time series as below.

| Objective | Statistical/Analytical Technique | Variables | Expected outcome |
| :---- | :---- | :---- | :---- |
| Identifying volume and temporal dynamics. | Time series graphs and tables showing volume by platform, language, and theme | Post Count, Time (Hourly) | Identification of peaks and troughs |
| Understanding thematic compositions | Frequency Tables | Post Count by Categorical Factors | Summary charts showing proportions by group |
| Identifying sentiment distribution | Descriptive Statistics (Mean, IQR, Boxplots by Group) | Sentiment Score (Continuous) | Identifying central tendency |
| Dimension reduction | Principal Component Analysis (PCA) | Mixed Numeric & Categorical Features | Interpreting PCs |
| Sentiment varies over time and across themes | Clustering, Chi-square tests / ANOVA | Sentiment score, (numeric) | Identifying clusters |

* **References:**  
* [https://huggingface.co/datasets/Exorde/exorde-social-media-december-2024-week1](https://huggingface.co/datasets/Exorde/exorde-social-media-december-2024-week1) Accessed: 02.04.2026  
* [https://exordelabs.com/blog](https://exordelabs.com/blog)  Accessed: 25.11.2025