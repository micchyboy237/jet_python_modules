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

The highest-scoring chunk for each query.
| Query | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| What is the premise of 'The Daily Life of a Middle-Aged Online Shopper in Another World'? | doc_1 | 0 | No Header | 1.0000 | False | ## Jump to - 15 The Daily Life of a Middle-Aged Online Shopper in Another World - 14 Magic Maker: How to Create Magic in Another World - 13 The Red Ranger Becomes an Adventurer in Another World - 12 T... |
| Which 2025 isekai anime features a character reincarnated as a villainess from an otome game? | doc_12 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 10 to Mar 28, 2025 Genres: Comedy, Fantasy, Isekai Episodes: 12 Studios: Ajiado The Middle-Aged Man That Reincarnated as a Villainess Kenzaburou Tondabayashi, a 52-year-ol... |
| What are the genres and studio for 'Magic Maker: How to Create Magic in Another World'? | doc_6 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 9 to Mar 27, 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: Studio DEEN Magic Maker: How to Create Magic in Another World is about a 30-year-old man... |
| Which 2025 isekai anime has the shortest episode count, and what is it? | doc_0 | 0 | No Header | 1.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing up to unleash a thrilling lineup of action-packed series that are sure to grab your attention with ... |
| What is the unique skill of the protagonist in 'Campfire Cooking in Another World with My Absurd Skill Season 2'? | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |
| Which 2025 isekai anime involves a former power ranger as the protagonist? | doc_8 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 12 to Mar 30, 2025 Genres: Action, Adventure, Fantasy, Comedy, Isekai Episodes: 12 Studios: Satelight The Red Ranger Becomes an Adventurer in Another World Tougo Asagaki, ... |
| When does 'The Beginning After the End' air, and on which platform can it be watched? | doc_17 | 0 | No Header | 1.0000 | False | ## 08 Headhunted to Another World: From Salaryman to Big Four! |
| What challenges does the protagonist face in 'Promise of Wizard'? | doc_15 | 0 | No Header | 1.0000 | False | ## 09 Promise of Wizard [ ](https://myanimelist.net/login.php?from=%2Fanime%2F57152%2FMahoutsukai_no_Yakusoku%3Fq%3DMahoutsukai%2520no%2520Yakusoku%26cat%3Danime&error=login_required&) |
| How does the protagonist in 'Possibly the Greatest Alchemist of All Time' gain power? | doc_20 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 1 to Mar 19, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Studio Comet Possibly the Greatest Alchemist of All Time Takumi Iruma an ordinary individual une... |
| Which 2025 isekai anime are set to air in April 2025? | doc_0 | 0 | No Header | 1.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing up to unleash a thrilling lineup of action-packed series that are sure to grab your attention with ... |
| What is the setting and main theme of 'Lord of the Mysteries'? | doc_14 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Summer 2025 to ??? 2025 Genres: Action, Drama, Fantasy, Mystery, Thriller, Isekai Episodes: 12 Studios: B.CMAY PICTURES Lord of the Mysteries , the question looms: who can tru... |
| Which 2025 isekai anime combines comedy and slice of life genres? | doc_0 | 0 | No Header | 1.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing up to unleash a thrilling lineup of action-packed series that are sure to grab your attention with ... |
| What is the background of the protagonist in 'Hell Mode' before being transported to another world? | doc_21 | 0 | No Header | 1.0000 | False | ## 06 Hell Mode: Gamer Who Likes to Speedrun Becomes Peerless in a Parallel World with Obsolete Setting |
| Which studios are producing 'I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic'? | doc_24 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 7 to Mar 25, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Marvy Jack, Studio DEEN I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic... |
| What is the main goal of the protagonist in 'Teogonia'? | doc_10 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: Asahi Production Teogonia in the unforgiving landscape known as the borderlands, hum... |

## Detailed Results per Query

### Query: What is the premise of 'The Daily Life of a Middle-Aged Online Shopper in Another World'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_1 | 0 | No Header | 1.0000 | False | ## Jump to - 15 The Daily Life of a Middle-Aged Online Shopper in Another World - 14 Magic Maker: How to Create Magic in Another World - 13 The Red Ranger Becomes an Adventurer in Another World - 12 T... |
| doc_2 | 0 | No Header | 0.0000 | False | ## 15 The Daily Life of a Middle-Aged Online Shopper in Another World |

### Query: Which 2025 isekai anime features a character reincarnated as a villainess from an otome game?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_12 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 10 to Mar 28, 2025 Genres: Comedy, Fantasy, Isekai Episodes: 12 Studios: Ajiado The Middle-Aged Man That Reincarnated as a Villainess Kenzaburou Tondabayashi, a 52-year-ol... |
| doc_11 | 0 | No Header | 0.0000 | False | ## 11 The Middle-Aged Man That Reincarnated as a Villainess |

### Query: What are the genres and studio for 'Magic Maker: How to Create Magic in Another World'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_6 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 9 to Mar 27, 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: Studio DEEN Magic Maker: How to Create Magic in Another World is about a 30-year-old man... |
| doc_5 | 0 | No Header | 0.0000 | False | ## 14 Magic Maker: How to Create Magic in Another World |

### Query: Which 2025 isekai anime has the shortest episode count, and what is it?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_0 | 0 | No Header | 1.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing up to unleash a thrilling lineup of action-packed series that are sure to grab your attention with ... |
| doc_4 | 0 | No Header | 0.0000 | False | #### Related- [ 15 Best Upcoming Action Anime of 2025 ](https://animebytes.in/2025-best-action-anime-15-shows-you-cant-miss/) |

### Query: What is the unique skill of the protagonist in 'Campfire Cooking in Another World with My Absurd Skill Season 2'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 12 Studios: Mappa Campfire Cooking in Another World with My Absurd Skill Season 2 follows Tsuyoshi Mu... |
| doc_27 | 0 | No Header | 0.0000 | False | ## 03 Campfire Cooking in Another World with My Absurd Skill Season 2 |

### Query: Which 2025 isekai anime involves a former power ranger as the protagonist?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_0 | 0 | No Header | 0.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing up to unleash a thrilling lineup of action-packed series that are sure to grab your attention with ... |
| doc_8 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 12 to Mar 30, 2025 Genres: Action, Adventure, Fantasy, Comedy, Isekai Episodes: 12 Studios: Satelight The Red Ranger Becomes an Adventurer in Another World Tougo Asagaki, ... |

### Query: When does 'The Beginning After the End' air, and on which platform can it be watched?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_17 | 0 | No Header | 1.0000 | False | ## 08 Headhunted to Another World: From Salaryman to Big Four! |
| doc_2 | 0 | No Header | 0.0000 | True | ## 15 The Daily Life of a Middle-Aged Online Shopper in Another World |

### Query: What challenges does the protagonist face in 'Promise of Wizard'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_15 | 0 | No Header | 1.0000 | False | ## 09 Promise of Wizard [ ](https://myanimelist.net/login.php?from=%2Fanime%2F57152%2FMahoutsukai_no_Yakusoku%3Fq%3DMahoutsukai%2520no%2520Yakusoku%26cat%3Danime&error=login_required&) |
| doc_23 | 0 | No Header | 0.0000 | False | ## 05 I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic |

### Query: How does the protagonist in 'Possibly the Greatest Alchemist of All Time' gain power?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_20 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 1 to Mar 19, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Studio Comet Possibly the Greatest Alchemist of All Time Takumi Iruma an ordinary individual une... |
| doc_11 | 0 | No Header | 0.0000 | False | ## 11 The Middle-Aged Man That Reincarnated as a Villainess |

### Query: Which 2025 isekai anime are set to air in April 2025?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_0 | 0 | No Header | 1.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing up to unleash a thrilling lineup of action-packed series that are sure to grab your attention with ... |
| doc_4 | 0 | No Header | 0.0000 | False | #### Related- [ 15 Best Upcoming Action Anime of 2025 ](https://animebytes.in/2025-best-action-anime-15-shows-you-cant-miss/) |

### Query: What is the setting and main theme of 'Lord of the Mysteries'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_14 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Summer 2025 to ??? 2025 Genres: Action, Drama, Fantasy, Mystery, Thriller, Isekai Episodes: 12 Studios: B.CMAY PICTURES Lord of the Mysteries , the question looms: who can tru... |
| doc_11 | 0 | No Header | 0.0000 | False | ## 11 The Middle-Aged Man That Reincarnated as a Villainess |

### Query: Which 2025 isekai anime combines comedy and slice of life genres?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_0 | 0 | No Header | 1.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing up to unleash a thrilling lineup of action-packed series that are sure to grab your attention with ... |
| doc_4 | 0 | No Header | 0.0000 | False | #### Related- [ 15 Best Upcoming Action Anime of 2025 ](https://animebytes.in/2025-best-action-anime-15-shows-you-cant-miss/) |

### Query: What is the background of the protagonist in 'Hell Mode' before being transported to another world?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_21 | 0 | No Header | 1.0000 | False | ## 06 Hell Mode: Gamer Who Likes to Speedrun Becomes Peerless in a Parallel World with Obsolete Setting |
| doc_11 | 0 | No Header | 0.0000 | False | ## 11 The Middle-Aged Man That Reincarnated as a Villainess |

### Query: Which studios are producing 'I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_24 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 7 to Mar 25, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studios: Marvy Jack, Studio DEEN I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic... |
| doc_23 | 0 | No Header | 0.0000 | False | ## 05 I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic |

### Query: What is the main goal of the protagonist in 'Teogonia'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_10 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 12 Studios: Asahi Production Teogonia in the unforgiving landscape known as the borderlands, hum... |
| doc_11 | 0 | No Header | 0.0000 | False | ## 11 The Middle-Aged Man That Reincarnated as a Villainess |

