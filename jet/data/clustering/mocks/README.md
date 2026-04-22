I've updated the JSON datasets to match your **exact sample structure** (JAV metadata with `url`, `text`, `thumbnail`, `preview`, `videoId`, `code`, `episode`). Each dataset is tailored for the corresponding clustering approach.

---

### 🔹 Dataset for Approach 1: BERT + K‑Means (Semantic Similarity)

**Why this works:**  
This dataset contains **synonyms and paraphrases** of the same underlying themes. BERT embeddings will group videos with semantically similar but lexically different descriptions (e.g., "sleeping verification" ≈ "bedroom seduction").

```json
[
  {
    "url": "https://missav.ws/en/hawa-272",
    "text": "HAWA-272 Sleeping verification 'I want to keep beautiful nakedness' Will my wife who drips love juice after seeing a stick of a younger model than her husband who co-starred in the memorial nude shooting will SEX? VOL.14",
    "thumbnail": "https://fourhoi.com/hawa-272/cover-t.jpg",
    "preview": "https://fourhoi.com/hawa-272/preview.mp4",
    "videoId": "hawa-272",
    "code": "hawa",
    "episode": "272"
  },
  {
    "url": "https://missav.ws/en/hawa-273",
    "text": "HAWA-273 While my spouse slumbers beside me, a stranger's touch awakens desires I never knew existed",
    "thumbnail": "https://fourhoi.com/hawa-273/cover-t.jpg",
    "preview": "https://fourhoi.com/hawa-273/preview.mp4",
    "videoId": "hawa-273",
    "code": "hawa",
    "episode": "273"
  },
  {
    "url": "https://missav.ws/en/hawa-274",
    "text": "HAWA-274 In the dead of night, her body betrays her husband as a younger man slips into their bed",
    "thumbnail": "https://fourhoi.com/hawa-274/cover-t.jpg",
    "preview": "https://fourhoi.com/hawa-274/preview.mp4",
    "videoId": "hawa-274",
    "code": "hawa",
    "episode": "274"
  },
  {
    "url": "https://missav.ws/en/nsps-500",
    "text": "NSPS-500 My husband watches from the closet as I'm ravaged by his younger coworker",
    "thumbnail": "https://fourhoi.com/nsps-500/cover-t.jpg",
    "preview": "https://fourhoi.com/nsps-500/preview.mp4",
    "videoId": "nsps-500",
    "code": "nsps",
    "episode": "500"
  },
  {
    "url": "https://missav.ws/en/nsps-501",
    "text": "NSPS-501 Cuckold fantasy becomes reality when wife surrenders to best friend's charm",
    "thumbnail": "https://fourhoi.com/nsps-501/cover-t.jpg",
    "preview": "https://fourhoi.com/nsps-501/preview.mp4",
    "videoId": "nsps-501",
    "code": "nsps",
    "episode": "501"
  },
  {
    "url": "https://missav.ws/en/nsps-502",
    "text": "NSPS-502 He begged me to sleep with another man - now he regrets it deeply",
    "thumbnail": "https://fourhoi.com/nsps-502/cover-t.jpg",
    "preview": "https://fourhoi.com/nsps-502/preview.mp4",
    "videoId": "nsps-502",
    "code": "nsps",
    "episode": "502"
  }
]
```

---

### 🔹 Dataset for Approach 2: TF‑IDF + K‑Means (Keyword‑Driven)

**Why this works:**  
Each cluster is defined by **high‑frequency, domain‑specific terms** like "cuckold", "NTR", "wife" vs. "sleeping", "verification", "romantic". TF‑IDF will easily separate these.

```json
[
  {
    "url": "https://missav.ws/en/nsps-842",
    "text": "NSPS-842 Husbands Who Awakened To Cuckold Propensity Please Mess Your Wife Best",
    "thumbnail": "https://fourhoi.com/nsps-842/cover-t.jpg",
    "preview": "https://fourhoi.com/nsps-842/preview.mp4",
    "videoId": "nsps-842",
    "code": "nsps",
    "episode": "842"
  },
  {
    "url": "https://missav.ws/en/jur-011",
    "text": "JUR-011 The third shocking transfer. Drenched in oil and infidelity, she appears in the full-scale cuckold series!! Nude model NTR Shocking video of a wife drowning in shame with her boss",
    "thumbnail": "https://fourhoi.com/jur-011/cover-t.jpg",
    "preview": "https://fourhoi.com/jur-011/preview.mp4",
    "videoId": "jur-011",
    "code": "jur",
    "episode": "011"
  },
  {
    "url": "https://missav.ws/en/bnsps-314",
    "text": "BNSPS-314 Cuckold Desires A Wife I've Never Seen! selection of masterpieces",
    "thumbnail": "https://fourhoi.com/bnsps-314/cover-t.jpg",
    "preview": "https://fourhoi.com/bnsps-314/preview.mp4",
    "videoId": "bnsps-314",
    "code": "bnsps",
    "episode": "314"
  },
  {
    "url": "https://missav.ws/en/nsps-488",
    "text": "NSPS-488 My First Cuckold 6 ~My Wife Goes Crazy with Another Man Through the Magic Mirror~",
    "thumbnail": "https://fourhoi.com/nsps-488/cover-t.jpg",
    "preview": "https://fourhoi.com/nsps-488/preview.mp4",
    "videoId": "nsps-488",
    "code": "nsps",
    "episode": "488"
  },
  {
    "url": "https://missav.ws/en/hawa-300",
    "text": "HAWA-300 Sleeping verification vol.15 - Romantic night when wife wakes to unexpected pleasure",
    "thumbnail": "https://fourhoi.com/hawa-300/cover-t.jpg",
    "preview": "https://fourhoi.com/hawa-300/preview.mp4",
    "videoId": "hawa-300",
    "code": "hawa",
    "episode": "300"
  },
  {
    "url": "https://missav.ws/en/hawa-301",
    "text": "HAWA-301 Sleeping wife seduction - Romantic encounter while husband travels abroad",
    "thumbnail": "https://fourhoi.com/hawa-301/cover-t.jpg",
    "preview": "https://fourhoi.com/hawa-301/preview.mp4",
    "videoId": "hawa-301",
    "code": "hawa",
    "episode": "301"
  }
]
```

---

### 🔹 Dataset for Approach 4: Hierarchical Agglomerative Clustering

**Why this works:**  
Nested themes allow the dendrogram to show **multi‑level groupings**. The top split separates "Cuckold/NTR" from "Romantic Sleeping", with sub‑clusters within Cuckold (e.g., "husband watches" vs. "wife betrayal").

```json
[
  {
    "url": "https://missav.ws/en/nsps-500",
    "text": "NSPS-500 Cuckold husband watches wife with younger man - real hidden camera",
    "thumbnail": "https://fourhoi.com/nsps-500/cover-t.jpg",
    "preview": "https://fourhoi.com/nsps-500/preview.mp4",
    "videoId": "nsps-500",
    "code": "nsps",
    "episode": "500"
  },
  {
    "url": "https://missav.ws/en/nsps-501",
    "text": "NSPS-501 Watching my wife get pleasured by a stranger - cuckold confession",
    "thumbnail": "https://fourhoi.com/nsps-501/cover-t.jpg",
    "preview": "https://fourhoi.com/nsps-501/preview.mp4",
    "videoId": "nsps-501",
    "code": "nsps",
    "episode": "501"
  },
  {
    "url": "https://missav.ws/en/jur-020",
    "text": "JUR-020 NTR drama - Wife cheats with boss while husband works late",
    "thumbnail": "https://fourhoi.com/jur-020/cover-t.jpg",
    "preview": "https://fourhoi.com/jur-020/preview.mp4",
    "videoId": "jur-020",
    "code": "jur",
    "episode": "020"
  },
  {
    "url": "https://missav.ws/en/jur-021",
    "text": "JUR-021 Betrayal - She chose her ex-boyfriend over me",
    "thumbnail": "https://fourhoi.com/jur-021/cover-t.jpg",
    "preview": "https://fourhoi.com/jur-021/preview.mp4",
    "videoId": "jur-021",
    "code": "jur",
    "episode": "021"
  },
  {
    "url": "https://missav.ws/en/hawa-400",
    "text": "HAWA-400 Sleeping next to my wife while another man takes her - silent cuckold",
    "thumbnail": "https://fourhoi.com/hawa-400/cover-t.jpg",
    "preview": "https://fourhoi.com/hawa-400/preview.mp4",
    "videoId": "hawa-400",
    "code": "hawa",
    "episode": "400"
  },
  {
    "url": "https://missav.ws/en/hawa-401",
    "text": "HAWA-401 Romantic sleeping wife wakes to gentle lovemaking",
    "thumbnail": "https://fourhoi.com/hawa-401/cover-t.jpg",
    "preview": "https://fourhoi.com/hawa-401/preview.mp4",
    "videoId": "hawa-401",
    "code": "hawa",
    "episode": "401"
  },
  {
    "url": "https://missav.ws/en/hawa-402",
    "text": "HAWA-402 Tender moment with sleeping beauty - pure romance",
    "thumbnail": "https://fourhoi.com/hawa-402/cover-t.jpg",
    "preview": "https://fourhoi.com/hawa-402/preview.mp4",
    "videoId": "hawa-402",
    "code": "hawa",
    "episode": "402"
  }
]
```

---

### 🔹 Dataset for Approach 5: Query‑Focused Clustering with HDBSCAN

**Why this works:**  
Simulates a search for **"cuckold videos"**. Contains a dense cluster of pure cuckold titles, some related but not exact matches (e.g., "sleeping wife" with mild NTR), and a few completely unrelated videos (outliers) that HDBSCAN will flag as noise (`label = -1`).

```json
[
  {
    "url": "https://missav.ws/en/nsps-600",
    "text": "NSPS-600 True Cuckold Story - Husband Films Wife's Affair",
    "thumbnail": "https://fourhoi.com/nsps-600/cover-t.jpg",
    "preview": "https://fourhoi.com/nsps-600/preview.mp4",
    "videoId": "nsps-600",
    "code": "nsps",
    "episode": "600"
  },
  {
    "url": "https://missav.ws/en/jur-050",
    "text": "JUR-050 Cuckold NTR - Wife Humiliated in Front of Husband",
    "thumbnail": "https://fourhoi.com/jur-050/cover-t.jpg",
    "preview": "https://fourhoi.com/jur-050/preview.mp4",
    "videoId": "jur-050",
    "code": "jur",
    "episode": "050"
  },
  {
    "url": "https://missav.ws/en/nsps-601",
    "text": "NSPS-601 I Made My Husband Watch Me Cheat - Cuckold Confessions 3",
    "thumbnail": "https://fourhoi.com/nsps-601/cover-t.jpg",
    "preview": "https://fourhoi.com/nsps-601/preview.mp4",
    "videoId": "nsps-601",
    "code": "nsps",
    "episode": "601"
  },
  {
    "url": "https://missav.ws/en/bnsps-400",
    "text": "BNSPS-400 Best Cuckold Scenes Compilation - Wives Betrayed",
    "thumbnail": "https://fourhoi.com/bnsps-400/cover-t.jpg",
    "preview": "https://fourhoi.com/bnsps-400/preview.mp4",
    "videoId": "bnsps-400",
    "code": "bnsps",
    "episode": "400"
  },
  {
    "url": "https://missav.ws/en/hawa-500",
    "text": "HAWA-500 Sleeping wife verification - husband may suspect but doesn't know",
    "thumbnail": "https://fourhoi.com/hawa-500/cover-t.jpg",
    "preview": "https://fourhoi.com/hawa-500/preview.mp4",
    "videoId": "hawa-500",
    "code": "hawa",
    "episode": "500"
  },
  {
    "url": "https://missav.ws/en/rom-100",
    "text": "ROM-100 Pure Love Story - Couple's First Night Together",
    "thumbnail": "https://fourhoi.com/rom-100/cover-t.jpg",
    "preview": "https://fourhoi.com/rom-100/preview.mp4",
    "videoId": "rom-100",
    "code": "rom",
    "episode": "100"
  },
  {
    "url": "https://missav.ws/en/sis-999",
    "text": "SIS-999 Schoolgirl Uniform Special - Innocent First Time",
    "thumbnail": "https://fourhoi.com/sis-999/cover-t.jpg",
    "preview": "https://fourhoi.com/sis-999/preview.mp4",
    "videoId": "sis-999",
    "code": "sis",
    "episode": "999"
  }
]
```

---

### 📁 Usage Instructions

1. **Save each JSON block** as a separate file (e.g., `approach1_data.json`, `approach2_data.json`, etc.) or paste directly into the Python code from the previous answer.
2. The **code remains unchanged** – the `text` field is used for clustering; the other fields are carried through for output.
3. For **Approach 5**, set the `user_query` variable to something like `"cuckold videos"` to see HDBSCAN route to the dense cluster.
