def most_frequent_pair(sequences):
    """
    Args:
        sequences: list[list[int]] - list of token ID sequences
    Returns:
        tuple(int, int) or None - the most frequent adjacent pair, with ties
        broken by first appearance. Returns None if no pair exists.
    """
    count = {}
    
    for s in sequences:  
        for i in range(0, len(s) - 1):
            pair = (s[i], s[i + 1])
            if not pair in count:
                count[pair] = (1, i)
            else:
                (cnt, pos) = count[pair]
                count[pair] = (cnt + 1, pos)

    mfre = 0
    res = None
    
    for pair, v in count.items():
        if v[0] > mfre:
            mfre = v[0]
            res = pair
    return res


    
