# Evaluation Summary

This summary evaluates embedding models for semantic search in a Retrieval-Augmented Generation (RAG) context, analyzing precision, recall, and MRR across different models and chunk sizes, with detailed comparisons to highlight optimal configurations.

- **Chunk Sizes Used**: 150, 250, 350
- **Top-K Results Evaluated**: 3
- **Best Model (by Average Precision)**: Snowflake/snowflake-arctic-embed-s (at chunk size 150)
  - Precision@3: 0.2000
  - Recall@3: 0.0154
  - MRR: 0.3667

## Model Performance (Averaged Across Chunk Sizes)

| Model | Precision | Recall | MRR | Strengths | Weaknesses |
|-------|-----------|--------|-----|-----------|------------|
| sentence-transformers/all-MiniLM-L6-v2 | 0.1556 | 0.0127 | 0.4333 | High MRR: Ranks relevant chunks higher. | Low precision: Includes more irrelevant chunks.; Low recall: Misses relevant chunks. |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | 0.1333 | 0.0108 | 0.4000 | High MRR: Ranks relevant chunks higher. | Low precision: Includes more irrelevant chunks.; Low recall: Misses relevant chunks. |
| Snowflake/snowflake-arctic-embed-s | 0.2000 | 0.0154 | 0.3667 | High precision: Retrieves highly relevant chunks.; High recall: Captures most relevant chunks. | Low MRR: Poor ranking of relevant chunks. |

## Performance by Chunk Size and Model

This section compares precision, recall, and MRR for each model at different chunk sizes to identify optimal configurations for RAG.

| Model | Chunk Size | Precision | Recall | MRR |
|-------|------------|-----------|--------|-----|
| sentence-transformers/all-MiniLM-L6-v2 | 150 | 0.1556 | 0.0127 | 0.4333 |
| sentence-transformers/all-MiniLM-L6-v2 | 250 | 0.1556 | 0.0127 | 0.4333 |
| sentence-transformers/all-MiniLM-L6-v2 | 350 | 0.1556 | 0.0127 | 0.4333 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | 150 | 0.1333 | 0.0108 | 0.4000 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | 250 | 0.1333 | 0.0108 | 0.4000 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | 350 | 0.1333 | 0.0108 | 0.4000 |
| Snowflake/snowflake-arctic-embed-s | 150 | 0.2000 | 0.0154 | 0.3667 |
| Snowflake/snowflake-arctic-embed-s | 250 | 0.2000 | 0.0154 | 0.3667 |
| Snowflake/snowflake-arctic-embed-s | 350 | 0.2000 | 0.0154 | 0.3667 |

## Comparative Analysis

The following comparisons highlight key differences in model and chunk size performance:

### Precision
- **Best Model for Precision**: Snowflake/snowflake-arctic-embed-s at chunk size 150 (precision=0.2000)
- **sentence-transformers/all-MiniLM-L6-v2**:
  - Chunk Size 150: Precision=0.1556
  - Chunk Size 250: Precision=0.1556
  - Chunk Size 350: Precision=0.1556
  - **Trend**: Precision remains stable with larger chunk sizes.
- **sentence-transformers/multi-qa-MiniLM-L6-cos-v1**:
  - Chunk Size 150: Precision=0.1333
  - Chunk Size 250: Precision=0.1333
  - Chunk Size 350: Precision=0.1333
  - **Trend**: Precision remains stable with larger chunk sizes.
- **Snowflake/snowflake-arctic-embed-s**:
  - Chunk Size 150: Precision=0.2000
  - Chunk Size 250: Precision=0.2000
  - Chunk Size 350: Precision=0.2000
  - **Trend**: Precision remains stable with larger chunk sizes.
### Recall
- **Best Model for Recall**: Snowflake/snowflake-arctic-embed-s at chunk size 150 (recall=0.0154)
- **sentence-transformers/all-MiniLM-L6-v2**:
  - Chunk Size 150: Recall=0.0127
  - Chunk Size 250: Recall=0.0127
  - Chunk Size 350: Recall=0.0127
  - **Trend**: Recall remains stable with larger chunk sizes.
- **sentence-transformers/multi-qa-MiniLM-L6-cos-v1**:
  - Chunk Size 150: Recall=0.0108
  - Chunk Size 250: Recall=0.0108
  - Chunk Size 350: Recall=0.0108
  - **Trend**: Recall remains stable with larger chunk sizes.
- **Snowflake/snowflake-arctic-embed-s**:
  - Chunk Size 150: Recall=0.0154
  - Chunk Size 250: Recall=0.0154
  - Chunk Size 350: Recall=0.0154
  - **Trend**: Recall remains stable with larger chunk sizes.
### Mrr
- **Best Model for Mrr**: sentence-transformers/all-MiniLM-L6-v2 at chunk size 150 (mrr=0.4333)
- **sentence-transformers/all-MiniLM-L6-v2**:
  - Chunk Size 150: Mrr=0.4333
  - Chunk Size 250: Mrr=0.4333
  - Chunk Size 350: Mrr=0.4333
  - **Trend**: Mrr remains stable with larger chunk sizes.
- **sentence-transformers/multi-qa-MiniLM-L6-cos-v1**:
  - Chunk Size 150: Mrr=0.4000
  - Chunk Size 250: Mrr=0.4000
  - Chunk Size 350: Mrr=0.4000
  - **Trend**: Mrr remains stable with larger chunk sizes.
- **Snowflake/snowflake-arctic-embed-s**:
  - Chunk Size 150: Mrr=0.3667
  - Chunk Size 250: Mrr=0.3667
  - Chunk Size 350: Mrr=0.3667
  - **Trend**: Mrr remains stable with larger chunk sizes.

## Performance by Query Type

| Model | Short Query Precision | Short Query Recall | Short Query MRR | Long Query Precision | Long Query Recall | Long Query MRR |
|-------|----------------------|--------------------|-----------------|---------------------|-------------------|----------------|
| sentence-transformers/all-MiniLM-L6-v2 | 0.1111 | 0.0086 | 0.3333 | 0.0000 | 0.0000 | 0.0000 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | 0.1111 | 0.0086 | 0.3333 | 0.0000 | 0.0000 | 0.0000 |
| Snowflake/snowflake-arctic-embed-s | 0.1111 | 0.0090 | 0.3000 | 0.0000 | 0.0000 | 0.0000 |

## Error Analysis

**Failed Queries**: 10 queries retrieved no relevant chunks:
- What is the premise of 'The Daily Life of a Middle-Aged Online Shopper in Another World'?
- Which 2025 isekai anime features a character reincarnated as a villainess from an otome game?
- What are the genres and studio for 'Magic Maker: How to Create Magic in Another World'?
- What is the unique skill of the protagonist in 'Campfire Cooking in Another World with My Absurd Skill Season 2'?
- What challenges does the protagonist face in 'Promise of Wizard'?

## Top Results per Query

The highest-scoring chunk for each query, with the model that produced it.
| Query | Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|-------|--------|----------|--------|-------|----------|--------------|
| What is the premise of 'The Daily Life of a Middle-Aged Online Shopper in Another World'? | sentence-transformers/all-MiniLM-L6-v2 | doc_3 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 9 to Apr 3, 2025 Genres: Isekai, Ecchi, Fantasy Episodes: 13 Studios: East Fish Studio The Daily Life of a Middle-Aged Online Shopper in Another World Kenichi, a solitary ... |
| Which 2025 isekai anime features a character reincarnated as a villainess from an otome game? | sentence-transformers/all-MiniLM-L6-v2 | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |
| What are the genres and studio for 'Magic Maker: How to Create Magic in Another World'? | sentence-transformers/all-MiniLM-L6-v2 | doc_5 | 0 | No Header | 1.0000 | False | ## 14 Magic Maker: How to Create Magic in Another World |
| Which 2025 isekai anime has the shortest episode count, and what is it? | sentence-transformers/all-MiniLM-L6-v2 | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |
| What is the unique skill of the protagonist in 'Campfire Cooking in Another World with My Absurd Skill Season 2'? | sentence-transformers/all-MiniLM-L6-v2 | doc_27 | 0 | No Header | 1.0000 | False | ## 03 Campfire Cooking in Another World with My Absurd Skill Season 2 |
| Which 2025 isekai anime involves a former power ranger as the protagonist? | sentence-transformers/all-MiniLM-L6-v2 | doc_8 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 12 to Mar 30, 2025 Genres: Action, Adventure, Fantasy, Comedy, Isekai Episodes: 12 Studios: Satelight The Red Ranger Becomes an Adventurer in Another World Tougo Asagaki, ... |
| When does 'The Beginning After the End' air, and on which platform can it be watched? | sentence-transformers/all-MiniLM-L6-v2 | doc_32 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: A-CAT The Beginning After the End follows King Grey, a powerful monarch who, after a life fi... |
| What challenges does the protagonist face in 'Promise of Wizard'? | sentence-transformers/all-MiniLM-L6-v2 | doc_16 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 6 to Mar 24, 2025 Genres: Fantasy, Isekai Episodes: 12 Studios: LIDENFILMS Promise of Wizard the narrative centers around the Sage, a human from modern Japan who is summon... |
| How does the protagonist in 'Possibly the Greatest Alchemist of All Time' gain power? | sentence-transformers/all-MiniLM-L6-v2 | doc_20 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 1 to Mar 19, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Studio Comet Possibly the Greatest Alchemist of All Time Takumi Iruma an ordinary individual une... |
| Which 2025 isekai anime are set to air in April 2025? | sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_26 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Adventure, Ecchi, Fantasy, Isekai Episodes: 2 Studios: Drive KONOSUBA - God's blessing on this wonderful world! 3 Bonus Stage has two episode... |
| What is the setting and main theme of 'Lord of the Mysteries'? | sentence-transformers/all-MiniLM-L6-v2 | doc_14 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Summer 2025 to ??? 2025 Genres: Action, Drama, Fantasy, Mystery, Thriller, Isekai Episodes: 12 Studios: B.CMAY PICTURES Lord of the Mysteries , the question looms: who can tru... |
| Which 2025 isekai anime combines comedy and slice of life genres? | sentence-transformers/all-MiniLM-L6-v2 | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |
| What is the background of the protagonist in 'Hell Mode' before being transported to another world? | sentence-transformers/all-MiniLM-L6-v2 | doc_22 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: TBA Hell Mode follows Kenichi Yamada, a hardcore gamer whose favorite online game is shut... |
| Which studios are producing 'I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic'? | sentence-transformers/all-MiniLM-L6-v2 | doc_24 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 7 to Mar 25, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Marvy Jack, Studio DEEN I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic... |
| What is the main goal of the protagonist in 'Teogonia'? | sentence-transformers/all-MiniLM-L6-v2 | doc_10 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: Asahi Production Teogonia in the unforgiving landscape known as the borderlands, hum... |

## Detailed Results per Query

### Query: What is the premise of 'The Daily Life of a Middle-Aged Online Shopper in Another World'?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_3 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 9 to Apr 3, 2025 Genres: Isekai, Ecchi, Fantasy Episodes: 13 Studios: East Fish Studio The Daily Life of a Middle-Aged Online Shopper in Another World Kenichi, a solitary ... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_3 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 9 to Apr 3, 2025 Genres: Isekai, Ecchi, Fantasy Episodes: 13 Studios: East Fish Studio The Daily Life of a Middle-Aged Online Shopper in Another World Kenichi, a solitary ... |
| Snowflake/snowflake-arctic-embed-s | doc_1 | 0 | No Header | 1.0000 | False | ## Jump to - 15 The Daily Life of a Middle-Aged Online Shopper in Another World - 14 Magic Maker: How to Create Magic in Another World - 13 The Red Ranger Becomes an Adventurer in Another World - 12 T... |

### Query: Which 2025 isekai anime features a character reincarnated as a villainess from an otome game?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_12 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 10 to Mar 28, 2025 Genres: Comedy, Fantasy, Isekai Episodes: 12 Studios: Ajiado The Middle-Aged Man That Reincarnated as a Villainess Kenzaburou Tondabayashi, a 52-year-ol... |
| Snowflake/snowflake-arctic-embed-s | doc_12 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 10 to Mar 28, 2025 Genres: Comedy, Fantasy, Isekai Episodes: 12 Studios: Ajiado The Middle-Aged Man That Reincarnated as a Villainess Kenzaburou Tondabayashi, a 52-year-ol... |

### Query: What are the genres and studio for 'Magic Maker: How to Create Magic in Another World'?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_5 | 0 | No Header | 1.0000 | False | ## 14 Magic Maker: How to Create Magic in Another World |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_6 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 9 to Mar 27, 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: Studio DEEN Magic Maker: How to Create Magic in Another World is about a 30-year-old man... |
| Snowflake/snowflake-arctic-embed-s | doc_6 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 9 to Mar 27, 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: Studio DEEN Magic Maker: How to Create Magic in Another World is about a 30-year-old man... |

### Query: Which 2025 isekai anime has the shortest episode count, and what is it?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_0 | 0 | No Header | 1.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing up to unleash a thrilling lineup of action-packed series that are sure to grab your attention with ... |
| Snowflake/snowflake-arctic-embed-s | doc_0 | 0 | No Header | 1.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing up to unleash a thrilling lineup of action-packed series that are sure to grab your attention with ... |

### Query: What is the unique skill of the protagonist in 'Campfire Cooking in Another World with My Absurd Skill Season 2'?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_27 | 0 | No Header | 1.0000 | False | ## 03 Campfire Cooking in Another World with My Absurd Skill Season 2 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |
| Snowflake/snowflake-arctic-embed-s | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |

### Query: Which 2025 isekai anime involves a former power ranger as the protagonist?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_8 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 12 to Mar 30, 2025 Genres: Action, Adventure, Fantasy, Comedy, Isekai Episodes: 12 Studios: Satelight The Red Ranger Becomes an Adventurer in Another World Tougo Asagaki, ... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_8 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 12 to Mar 30, 2025 Genres: Action, Adventure, Fantasy, Comedy, Isekai Episodes: 12 Studios: Satelight The Red Ranger Becomes an Adventurer in Another World Tougo Asagaki, ... |
| Snowflake/snowflake-arctic-embed-s | doc_8 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 12 to Mar 30, 2025 Genres: Action, Adventure, Fantasy, Comedy, Isekai Episodes: 12 Studios: Satelight The Red Ranger Becomes an Adventurer in Another World Tougo Asagaki, ... |

### Query: When does 'The Beginning After the End' air, and on which platform can it be watched?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_32 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: A-CAT The Beginning After the End follows King Grey, a powerful monarch who, after a life fi... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_32 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: A-CAT The Beginning After the End follows King Grey, a powerful monarch who, after a life fi... |
| Snowflake/snowflake-arctic-embed-s | doc_17 | 0 | No Header | 1.0000 | False | ## 08 Headhunted to Another World: From Salaryman to Big Four! |

### Query: What challenges does the protagonist face in 'Promise of Wizard'?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_16 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 6 to Mar 24, 2025 Genres: Fantasy, Isekai Episodes: 12 Studios: LIDENFILMS Promise of Wizard the narrative centers around the Sage, a human from modern Japan who is summon... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_16 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 6 to Mar 24, 2025 Genres: Fantasy, Isekai Episodes: 12 Studios: LIDENFILMS Promise of Wizard the narrative centers around the Sage, a human from modern Japan who is summon... |
| Snowflake/snowflake-arctic-embed-s | doc_15 | 0 | No Header | 1.0000 | False | ## 09 Promise of Wizard [ ](https://myanimelist.net/login.php?from=%2Fanime%2F57152%2FMahoutsukai_no_Yakusoku%3Fq%3DMahoutsukai%2520no%2520Yakusoku%26cat%3Danime&error=login_required&) |

### Query: How does the protagonist in 'Possibly the Greatest Alchemist of All Time' gain power?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_20 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 1 to Mar 19, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Studio Comet Possibly the Greatest Alchemist of All Time Takumi Iruma an ordinary individual une... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_20 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 1 to Mar 19, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Studio Comet Possibly the Greatest Alchemist of All Time Takumi Iruma an ordinary individual une... |
| Snowflake/snowflake-arctic-embed-s | doc_20 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 1 to Mar 19, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Studio Comet Possibly the Greatest Alchemist of All Time Takumi Iruma an ordinary individual une... |

### Query: Which 2025 isekai anime are set to air in April 2025?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_26 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Adventure, Ecchi, Fantasy, Isekai Episodes: 2 Studios: Drive KONOSUBA - God's blessing on this wonderful world! 3 Bonus Stage has two episode... |
| Snowflake/snowflake-arctic-embed-s | doc_0 | 0 | No Header | 1.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing up to unleash a thrilling lineup of action-packed series that are sure to grab your attention with ... |
| sentence-transformers/all-MiniLM-L6-v2 | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |

### Query: What is the setting and main theme of 'Lord of the Mysteries'?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_14 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Summer 2025 to ??? 2025 Genres: Action, Drama, Fantasy, Mystery, Thriller, Isekai Episodes: 12 Studios: B.CMAY PICTURES Lord of the Mysteries , the question looms: who can tru... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_14 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Summer 2025 to ??? 2025 Genres: Action, Drama, Fantasy, Mystery, Thriller, Isekai Episodes: 12 Studios: B.CMAY PICTURES Lord of the Mysteries , the question looms: who can tru... |
| Snowflake/snowflake-arctic-embed-s | doc_14 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Summer 2025 to ??? 2025 Genres: Action, Drama, Fantasy, Mystery, Thriller, Isekai Episodes: 12 Studios: B.CMAY PICTURES Lord of the Mysteries , the question looms: who can tru... |

### Query: Which 2025 isekai anime combines comedy and slice of life genres?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |
| Snowflake/snowflake-arctic-embed-s | doc_0 | 0 | No Header | 1.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing up to unleash a thrilling lineup of action-packed series that are sure to grab your attention with ... |

### Query: What is the background of the protagonist in 'Hell Mode' before being transported to another world?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_22 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: TBA Hell Mode follows Kenichi Yamada, a hardcore gamer whose favorite online game is shut... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_22 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: TBA Hell Mode follows Kenichi Yamada, a hardcore gamer whose favorite online game is shut... |
| Snowflake/snowflake-arctic-embed-s | doc_21 | 0 | No Header | 1.0000 | False | ## 06 Hell Mode: Gamer Who Likes to Speedrun Becomes Peerless in a Parallel World with Obsolete Setting |

### Query: Which studios are producing 'I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic'?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_24 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 7 to Mar 25, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Marvy Jack, Studio DEEN I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_24 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 7 to Mar 25, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Marvy Jack, Studio DEEN I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic... |
| Snowflake/snowflake-arctic-embed-s | doc_24 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 7 to Mar 25, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Marvy Jack, Studio DEEN I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic... |

### Query: What is the main goal of the protagonist in 'Teogonia'?

| Model | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| sentence-transformers/all-MiniLM-L6-v2 | doc_10 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: Asahi Production Teogonia in the unforgiving landscape known as the borderlands, hum... |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | doc_10 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: Asahi Production Teogonia in the unforgiving landscape known as the borderlands, hum... |
| Snowflake/snowflake-arctic-embed-s | doc_10 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: Asahi Production Teogonia in the unforgiving landscape known as the borderlands, hum... |

