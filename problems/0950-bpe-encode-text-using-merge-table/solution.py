def bpe_encode(text, token_to_id, merges):
    # 1. 把字符转成初始 token id 列表
    ids = [token_to_id[ch] for ch in text]

    while len(ids) >= 2:
        # 2. 找到当前序列中，在 merges 里 new_id 最小（优先级最高）的 pair
        best_pair = None
        best_id = float('inf')

        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            if pair in merges and merges[pair] < best_id:
                best_id = merges[pair]
                best_pair = pair

        # 如果没有任何相邻 pair 可以合并，停止
        if best_pair is None:
            break

        # 3. 从左到右替换所有不重叠的 best_pair
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                new_ids.append(best_id)
                i += 2  # 合并了两个元素，跳过 2 步
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids

    # 4. 返回最终结果
    return ids