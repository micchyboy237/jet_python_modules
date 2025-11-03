def is_valid_sentence(sentence: str) -> bool:
    """
    Validate if the input string is a grammatically valid sentence.
    Args:
        sentence: The input sentence to validate.
    Returns:
        bool: True if the sentence is valid, False otherwise.
    """
    if not sentence or not sentence.strip():
        return False

    from jet.libs.stanza.pipeline import StanzaPipelineCache
    _nlp_cache = StanzaPipelineCache()
    nlp = _nlp_cache.get_pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    doc = nlp(sentence.strip())
    if len(doc.sentences) != 1:
        return False
    sent = doc.sentences[0]
    root_word = None
    for w in sent.words:
        if w.head == 0 and w.deprel == 'root':
            root_word = w
            break
    if root_word is None:
        return False
    if root_word.upos not in ('VERB', 'AUX'):
        return False
    has_subj = False
    for w in sent.words:
        if w.deprel in ('nsubj', 'nsubj:pass', 'csubj') and w.head == root_word.id:
            has_subj = True
            break
    if not has_subj:
        return False
    if len([w for w in sent.words if w.upos not in ('PUNCT',)]) < 2:
        return False
    return True
