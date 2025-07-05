# Evaluation Summary

This summary presents the evaluation results of different embedding models for semantic search.
- **Chunk Sizes Used**: 150, 250, 350
- **Top-K Results Evaluated**: 3
- **Best Model**: sentence-transformers/all-MiniLM-L6-v2
  - Precision@3: 0.1111
  - Recall@3: 0.0086
  - MRR: 0.3333

## Model Performance

| Model | Precision | Recall | MRR |
|-------|-----------|--------|-----|
| sentence-transformers/all-MiniLM-L6-v2 | 0.1111 | 0.0086 | 0.3333 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | 0.1111 | 0.0086 | 0.3333 |
| Snowflake/snowflake-arctic-embed-s | 0.1111 | 0.0090 | 0.3000 |

## Top Results per Query

The highest-scoring chunk for each query.
| Query | Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|-------|--------|----------|--------|-------|----------|--------------|
| What is the premise of 'The Daily Life of a Middle-Aged Online Shopper in Another World'? | doc_3 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 9 to Apr 3, 2025 Genres: Isekai, Ecchi, Fantasy Episodes: 13 Studios: E... |
| Which 2025 isekai anime features a character reincarnated as a villainess from an otome game? | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 1... |
| What are the genres and studio for 'Magic Maker: How to Create Magic in Another World'? | doc_5 | 0 | No Header | 1.0000 | False | ## 14 Magic Maker: How to Create Magic in Another World |
| Which 2025 isekai anime has the shortest episode count, and what is it? | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 1... |
| What is the unique skill of the protagonist in 'Campfire Cooking in Another World with My Absurd Skill Season 2'? | doc_27 | 0 | No Header | 1.0000 | False | ## 03 Campfire Cooking in Another World with My Absurd Skill Season 2 |
| Which 2025 isekai anime involves a former power ranger as the protagonist? | doc_8 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 12 to Mar 30, 2025 Genres: Action, Adventure, Fantasy, Comedy, Isekai E... |
| When does 'The Beginning After the End' air, and on which platform can it be watched? | doc_32 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 S... |
| What challenges does the protagonist face in 'Promise of Wizard'? | doc_16 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 6 to Mar 24, 2025 Genres: Fantasy, Isekai Episodes: 12 Studios: LIDENFI... |
| How does the protagonist in 'Possibly the Greatest Alchemist of All Time' gain power? | doc_20 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 1 to Mar 19, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studi... |
| Which 2025 isekai anime are set to air in April 2025? | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 1... |
| What is the setting and main theme of 'Lord of the Mysteries'? | doc_14 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Summer 2025 to ??? 2025 Genres: Action, Drama, Fantasy, Mystery, Thriller, ... |
| Which 2025 isekai anime combines comedy and slice of life genres? | doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 1... |
| What is the background of the protagonist in 'Hell Mode' before being transported to another world? | doc_22 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 1... |
| Which studios are producing 'I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic'? | doc_24 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 7 to Mar 25, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studi... |
| What is the main goal of the protagonist in 'Teogonia'? | doc_10 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Action, Adventure, Fantasy, Isekai Episod... |

## Detailed Results per Query

### Query: What is the premise of 'The Daily Life of a Middle-Aged Online Shopper in Another World'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_3 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 9 to Apr 3, 2025 Genres: Isekai, Ecchi, Fantasy Episodes: 13 Studios: E... |
| doc_2 | 0 | No Header | 0.0000 | False | ## 15 The Daily Life of a Middle-Aged Online Shopper in Another World |

### Query: Which 2025 isekai anime features a character reincarnated as a villainess from an otome game?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_0 | 0 | No Header | 0.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing... |
| doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 1... |

### Query: What are the genres and studio for 'Magic Maker: How to Create Magic in Another World'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_23 | 0 | No Header | 0.0000 | False | ## 05 I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic |
| doc_5 | 0 | No Header | 1.0000 | False | ## 14 Magic Maker: How to Create Magic in Another World |

### Query: Which 2025 isekai anime has the shortest episode count, and what is it?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_0 | 0 | No Header | 0.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing... |
| doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 1... |

### Query: What is the unique skill of the protagonist in 'Campfire Cooking in Another World with My Absurd Skill Season 2'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_1 | 0 | No Header | 0.0000 | False | ## Jump to - 15 The Daily Life of a Middle-Aged Online Shopper in Another World - 14 Magic Maker: Ho... |
| doc_27 | 0 | No Header | 1.0000 | False | ## 03 Campfire Cooking in Another World with My Absurd Skill Season 2 |

### Query: Which 2025 isekai anime involves a former power ranger as the protagonist?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_0 | 0 | No Header | 0.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing... |
| doc_8 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 12 to Mar 30, 2025 Genres: Action, Adventure, Fantasy, Comedy, Isekai E... |

### Query: When does 'The Beginning After the End' air, and on which platform can it be watched?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_32 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 S... |
| doc_26 | 0 | No Header | 0.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Adventure, Ecchi, Fantasy, Isekai Episode... |

### Query: What challenges does the protagonist face in 'Promise of Wizard'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_16 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 6 to Mar 24, 2025 Genres: Fantasy, Isekai Episodes: 12 Studios: LIDENFI... |
| doc_15 | 0 | No Header | 0.0000 | False | ## 09 Promise of Wizard [ ](https://myanimelist.net/login.php?from=%2Fanime%2F57152%2FMahoutsukai_no... |

### Query: How does the protagonist in 'Possibly the Greatest Alchemist of All Time' gain power?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_20 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 1 to Mar 19, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studi... |
| doc_11 | 0 | No Header | 0.0000 | False | ## 11 The Middle-Aged Man That Reincarnated as a Villainess |

### Query: Which 2025 isekai anime are set to air in April 2025?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_0 | 0 | No Header | 0.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing... |
| doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 1... |

### Query: What is the setting and main theme of 'Lord of the Mysteries'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_10 | 0 | No Header | 0.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Action, Adventure, Fantasy, Isekai Episod... |
| doc_14 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Summer 2025 to ??? 2025 Genres: Action, Drama, Fantasy, Mystery, Thriller, ... |

### Query: Which 2025 isekai anime combines comedy and slice of life genres?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_0 | 0 | No Header | 0.0000 | True | # 15 Best 2025 Isekai Anime That's You Need To Know As we dive into 2025, the anime scene is gearing... |
| doc_28 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Adventure, Comedy, Fantasy, Isekai Episodes: 1... |

### Query: What is the background of the protagonist in 'Hell Mode' before being transported to another world?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_22 | 0 | No Header | 1.0000 | False | ###### Start & End Date: ??? 2025 to ??? 2025 Genres: Action, Adventure, Fantasy, Isekai Episodes: 1... |
| doc_21 | 0 | No Header | 0.0000 | False | ## 06 Hell Mode: Gamer Who Likes to Speedrun Becomes Peerless in a Parallel World with Obsolete Sett... |

### Query: Which studios are producing 'I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_24 | 0 | No Header | 1.0000 | False | ###### Start & End Date: Jan 7 to Mar 25, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studi... |
| doc_23 | 0 | No Header | 0.0000 | False | ## 05 I'm a Noble on the Brink of Ruin, So I Might as Well Try Mastering Magic |

### Query: What is the main goal of the protagonist in 'Teogonia'?

| Doc ID | Chunk ID | Header | Score | Relevant | Text Preview |
|--------|----------|--------|-------|----------|--------------|
| doc_20 | 0 | No Header | 0.0000 | False | ###### Start & End Date: Jan 1 to Mar 19, 2025 Genres: Adventure, Fantasy, Isekai Episodes: 12 Studi... |
| doc_10 | 0 | No Header | 1.0000 | False | ###### Start & End Date: April 2025 to Spring 2025 Genres: Action, Adventure, Fantasy, Isekai Episod... |

