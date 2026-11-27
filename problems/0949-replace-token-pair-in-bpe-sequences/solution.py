def replace_pair(sequences, pair, new_id):
    # Your code here
    res = []
    for seq in sequences:
        j = 0
        r = []
        while j < len(seq):
            if j + 1 == len(seq):
                r.append(seq[j])
                break
            cur_pair = (seq[j], seq[j + 1])
            if pair != cur_pair:
                r.append(seq[j])
                j += 1
            else:
                r.append(new_id)
                j += 2
        res.append(r)
    return res
