# rag_stanza.py
from typing import List, Dict, TypedDict, Optional, Callable
import stanza

# --- types ---
class Entity(TypedDict):
    text: str
    type: str
    start_char: int
    end_char: int

class SentenceData(TypedDict):
    text: str
    tokens: List[str]
    lemmas: List[str]
    pos: List[str]
    deps: List[Dict[str, object]]  # each: {"text": str, "head": int, "deprel": str, "index": int}
    constituency: Optional[str]
    entities: List[Entity]

class Chunk(TypedDict):
    text: str
    sentence_indices: List[int]
    est_token_count: int
    metadata: Dict[str, object]

# --- pipeline helpers ---
def build_stanza_pipeline(lang: str = "en",
                          processors: str = "tokenize,mwt,pos,lemma,depparse,ner,constituency",
                          use_gpu: bool = False) -> stanza.Pipeline:
    """
    Build a Stanza pipeline with sensible defaults for RAG context building.
    Ensure the processors needed by depparse/constituency are included.
    """
    return stanza.Pipeline(lang=lang, processors=processors, use_gpu=use_gpu, verbose=False)

def parse_sentences(text: str, pipeline: stanza.Pipeline) -> List[SentenceData]:
    """
    Parse text into SentenceData entries using the provided stanza pipeline.
    """
    doc = pipeline(text)
    sentences: List[SentenceData] = []

    for sent in doc.sentences:
        tokens = [w.text for w in sent.words]
        lemmas = [w.lemma if getattr(w, "lemma", None) is not None else w.text for w in sent.words]
        pos = [w.upos for w in sent.words]
        deps = [{"text": w.text, "head": w.head, "deprel": w.deprel, "index": w.id} for w in sent.words]

        # constituency: stanza stores parse_tree; convert to bracket string if present
        constituency_str: Optional[str] = None
        if hasattr(sent, "constituency") and getattr(sent, "constituency", None) is not None:
            constituency_str = str(sent.constituency)

        entities: List[Entity] = []
        # stanza stores named entities at doc level; filter those overlapping sentence offsets
        for ent in doc.ents:
            if ent.start_char >= sent.tokens[0].start_char and ent.end_char <= sent.tokens[-1].end_char:
                entities.append({
                    "text": ent.text,
                    "type": ent.type,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char
                })

        sentences.append(SentenceData(
            text=sent.text,
            tokens=tokens,
            lemmas=lemmas,
            pos=pos,
            deps=deps,
            constituency=constituency_str,
            entities=entities
        ))
    return sentences

# --- salience & chunking ---
def sentence_salience_score(s: SentenceData) -> float:
    """
    Heuristic salience: prioritize sentences with named entities and syntactic heads.
    score = num_entities * 2 + unique_content_pos_count / 10 + root_token_indicator
    This is intentionally lightweight and deterministic.
    """
    num_entities = len(s["entities"])
    # content POS set
    content_pos = {"NOUN", "PROPN", "VERB", "ADJ"}
    content_count = sum(1 for p in s["pos"] if p in content_pos)
    # root presence indicator: presence of a 'root' deprel in deps
    root_indicator = 1.0 if any(d["deprel"].lower() == "root" for d in s["deps"]) else 0.0
    score = num_entities * 2.0 + (content_count / 10.0) + root_indicator
    return score

def chunk_sentences_for_rag(sentences: List[SentenceData],
                            max_tokens: int = 200,
                            overlap: int = 30,
                            token_counter: Optional[Callable[[List[str]], int]] = None) -> List[Chunk]:
    """
    Group sentences into chunks suited for RAG (est. token counts).
    token_counter: optional function that returns estimated token length given tokens list.
    Default uses simple token count (len(tokens)).
    """
    if token_counter is None:
        token_counter = lambda toks: len(toks)

    chunks: List[Chunk] = []
    current: List[int] = []
    current_tokens = 0

    for i, s in enumerate(sentences):
        s_token_count = token_counter(s["tokens"])
        # If single sentence exceeds max_tokens, create a chunk for it alone.
        if s_token_count >= max_tokens and not current:
            chunks.append(Chunk(
                text=s["text"],
                sentence_indices=[i],
                est_token_count=s_token_count,
                metadata={"salience": sentence_salience_score(s)}
            ))
            continue

        # if adding stays within limit, add
        if current_tokens + s_token_count <= max_tokens:
            current.append(i)
            current_tokens += s_token_count
        else:
            # finalize current chunk
            chunk_text = " ".join(sentences[j]["text"] for j in current)
            chunk_meta = {
                "salience": max(sentence_salience_score(sentences[j]) for j in current),
                "entities": [e for j in current for e in sentences[j]["entities"]],
            }
            chunks.append(Chunk(text=chunk_text, sentence_indices=current.copy(), est_token_count=current_tokens, metadata=chunk_meta))

            # start new chunk with overlap
            # determine overlap indices from tail of current
            overlap_indices = []
            tok_sum = 0
            # pick overlap sentences from the end until overlap token limit reached
            for j in reversed(current):
                tok = token_counter(sentences[j]["tokens"])
                if tok_sum + tok > overlap:
                    break
                overlap_indices.insert(0, j)
                tok_sum += tok

            current = overlap_indices.copy()
            current_tokens = tok_sum
            # now add current sentence if it fits
            if s_token_count + current_tokens <= max_tokens:
                current.append(i)
                current_tokens += s_token_count
            else:
                # otherwise, finalize current (if any) and put this sentence alone
                if current:
                    chunk_text = " ".join(sentences[j]["text"] for j in current)
                    chunk_meta = {
                        "salience": max(sentence_salience_score(sentences[j]) for j in current),
                        "entities": [e for j in current for e in sentences[j]["entities"]],
                    }
                    chunks.append(Chunk(text=chunk_text, sentence_indices=current.copy(), est_token_count=current_tokens, metadata=chunk_meta))
                    current = []
                    current_tokens = 0

                chunks.append(Chunk(text=s["text"], sentence_indices=[i], est_token_count=s_token_count, metadata={"salience": sentence_salience_score(s), "entities": s["entities"]}))

    # finalize leftover
    if current:
        chunk_text = " ".join(sentences[j]["text"] for j in current)
        chunk_meta = {
            "salience": max(sentence_salience_score(sentences[j]) for j in current),
            "entities": [e for j in current for e in sentences[j]["entities"]],
        }
        chunks.append(Chunk(text=chunk_text, sentence_indices=current.copy(), est_token_count=current_tokens, metadata=chunk_meta))

    return chunks

# --- small convenience: pipeline + chunk wrapper ---
def build_context_chunks(text: str,
                         lang: str = "en",
                         max_tokens: int = 200,
                         overlap: int = 30,
                         pipeline: Optional[stanza.Pipeline] = None) -> List[Chunk]:
    """
    High-level convenience function: builds pipeline if needed, parses, chunks, returns chunks.
    """
    if pipeline is None:
        pipeline = build_stanza_pipeline(lang=lang)
    sents = parse_sentences(text, pipeline)
    return chunk_sentences_for_rag(sents, max_tokens=max_tokens, overlap=overlap)
