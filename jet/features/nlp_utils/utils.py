def get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("V"):
        return "v"
    elif treebank_tag.startswith("N"):
        return "n"
    elif treebank_tag.startswith("J"):
        return "a"
    elif treebank_tag.startswith("R"):
        return "r"
    return "n"
