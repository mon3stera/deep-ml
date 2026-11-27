def bpe_decode(ids, vocab):
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """
    res = ""
    for id in ids:
        t = vocab[id]
        if t[0] == 'G':
            t = ' ' + t[1:]
        res += t
    return res