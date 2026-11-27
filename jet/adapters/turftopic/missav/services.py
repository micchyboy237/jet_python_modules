from typing import Literal, Union, overload

import numpy as np

# from jet.code.extraction.sentence_extraction import extract_sentences
from jet.logger import logger


@overload
def get_missav_videos_by_code(
    code: list[str] | None = None,
    limit: int | None = None,
    *,
    include_embeddings: Literal[False] = False,
) -> list[str]: ...


@overload
def get_missav_videos_by_code(
    code: list[str] | None = None,
    limit: int | None = None,
    *,
    include_embeddings: Literal[True],
) -> tuple[list[str], np.ndarray]: ...


def get_missav_videos_by_code(
    code: list[str] | None = None,
    limit: int | None = None,
    *,
    include_embeddings: bool = False,
) -> Union[list[str], tuple[list[str], np.ndarray]]:
    """
    Load video items grouped by (code, episode) from the missav DB, mirroring
    the logic in extract_videos_grouped_by_episode_code.sh.

    Each row is reduced to a representative video (preferring the "normal"
    non-uncensored/non-engsub version, then uncensored, then engsub) with
    merged, deduplicated genres. Results are formatted as strings containing
    title, details, and genres.

    Args:
        code: List of codes to filter on (e.g. ['ssni', 'ssis', 'snis']).
              Defaults to ('ssni', 'ssis', 'snis') when None.
        limit: Max number of grouped rows to return. When None (default),
               no LIMIT clause is applied and all matching rows are returned.
        include_embeddings: When True, also return a numpy array of embeddings
            (one row per formatted string). The embedding chosen for each
            group is taken from the video item whose ``details`` field is the
            longest (matching the representative ``details`` selection above).
            Groups whose representative row has a NULL embedding are skipped.
            The embedding column is only fetched from the DB when this is True.

    Returns:
        - When ``include_embeddings=False``: a list of formatted strings,
          one per grouped video.
        - When ``include_embeddings=True``: a tuple
          ``(formatted_strings, embeddings)`` where ``embeddings`` is a
          ``np.ndarray`` of shape ``(n, 768)`` and ``float32``, aligned
          index-wise with ``formatted_strings``.
    """
    import psycopg
    from psycopg.rows import dict_row

    codes = code or ["ssni", "ssis", "snis"]
    conninfo = "dbname=missav_db1 user=jethroestrada host=localhost port=5432"

    # Only select the embedding column (and its per-group aggregation) when
    # the caller actually needs it. Avoids shipping 768 floats per row over
    # the wire and skipping the array_agg over vector values in the executor.
    embedding_select = (
        ",\n"
        "            (array_agg(v.embedding ORDER BY length(v.details) ASC NULLS LAST))[1] AS embedding"
        if include_embeddings
        else ""
    )

    sql = f"""
        SELECT
            v.code,
            v.episode,
            (
                array_agg(v.id ORDER BY
                    CASE
                        WHEN v.url NOT LIKE '%%-uncensored-leak'
                         AND v.url NOT LIKE '%%-english-subtitle' THEN 0
                        WHEN v.url LIKE '%%-uncensored-leak'        THEN 1
                        ELSE 2
                    END,
                    v.id
                )
            )[1]                                                              AS id,
            (
                array_agg(v.video_id ORDER BY
                    CASE
                        WHEN v.url NOT LIKE '%%-uncensored-leak'
                         AND v.url NOT LIKE '%%-english-subtitle' THEN 0
                        WHEN v.url LIKE '%%-uncensored-leak'        THEN 1
                        ELSE 2
                    END,
                    v.id
                )
            )[1]                                                                 AS video_id,
            count(DISTINCT v.id)                                                 AS video_count,
            (array_agg(v.title ORDER BY length(v.title) ASC NULLS LAST))[1]      AS title,
            (array_agg(v.thumbnail ORDER BY v.id))[1]                            AS thumbnail,
            (array_agg(v.preview   ORDER BY v.id))[1]                            AS preview,
            (array_agg(v.details ORDER BY length(v.details) ASC NULLS LAST))[1]  AS details,
            (array_agg(v.release_date ORDER BY
                CASE
                    WHEN v.url NOT LIKE '%%-uncensored-leak'
                     AND v.url NOT LIKE '%%-english-subtitle' THEN 0
                    WHEN v.url LIKE '%%-uncensored-leak'        THEN 1
                    ELSE 2
                END,
                v.id
            ))[1]                                                             AS release_date,
            array_agg(DISTINCT g.genre)
                FILTER (WHERE g.genre IS NOT NULL
                          AND g.genre <> ''
                          AND lower(btrim(g.genre)) NOT IN ('uncensored leak', 'english subtitle')) AS genres,
            min(v.created_at)                                                 AS created_at,
            max(v.updated_at)                                                 AS updated_at,
            (array_agg(v.url ORDER BY v.id)
                FILTER (WHERE v.url NOT LIKE '%%-uncensored-leak'
                           AND v.url NOT LIKE '%%-english-subtitle'))[1]       AS url,
            (array_agg(v.url ORDER BY v.id)
                FILTER (WHERE v.url LIKE '%%-uncensored-leak'))[1]             AS url_uncnsrd,
            (array_agg(v.url ORDER BY v.id)
                FILTER (WHERE v.url LIKE '%%-english-subtitle'))[1]            AS url_engsub{embedding_select}
        FROM public.videos v
        LEFT JOIN LATERAL unnest(v.genres) AS g(genre) ON true
        WHERE
            v.code = ANY(%(codes)s)
            AND btrim(v.code)    <> ''
            AND btrim(v.episode) <> ''
            AND v.title          IS NOT NULL
            AND v.details        IS NOT NULL
            AND v.release_date   IS NOT NULL
            AND v.genres         IS NOT NULL
            AND cardinality(v.genres) > 0
        GROUP BY
            v.code,
            v.episode
        ORDER BY
            v.code,
            v.episode
    """

    params: dict = {"codes": codes}
    if limit is not None:
        sql += "\n        LIMIT %(limit)s"
        params["limit"] = limit

    formatted: list[str] = []
    embeddings: list[list[float]] = []
    embedding_dim: int | None = None

    try:
        with psycopg.connect(conninfo, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
    except Exception as e:
        logger.error("Failed to load missav videos: %s", e)
        raise

    for row in rows:
        # Only skip when embeddings were requested AND the representative row
        # had no embedding. When include_embeddings=False, `embedding` isn't
        # even in the row, so this check must be guarded.
        if include_embeddings and row["embedding"] is None:
            continue

        title = row["title"]
        details = row["details"]
        formatted.append(f"{title}\n\n{details}")

        if include_embeddings:
            vec = _parse_embedding(row["embedding"])
            if vec is None:
                # Defensive: DB said it had one but parsing yielded nothing.
                # Keep alignment by dropping the string we just appended.
                formatted.pop()
                continue
            if embedding_dim is None:
                embedding_dim = len(vec)
            elif len(vec) != embedding_dim:
                # Skip malformed rows instead of crashing the whole load.
                logger.warning(
                    "Skipping row with embedding dim %d (expected %d) for code=%r episode=%r",
                    len(vec),
                    embedding_dim,
                    row["code"],
                    row["episode"],
                )
                formatted.pop()
                continue
            embeddings.append(vec)

    if not include_embeddings:
        return formatted

    if not embeddings:
        # Default to 768 (all-mpnet / MiniLM family) so callers get a
        # consistently-shaped empty array.
        return formatted, np.empty((0, embedding_dim or 768), dtype=np.float32)

    emb_array = np.asarray(embeddings, dtype=np.float32)
    return formatted, emb_array
