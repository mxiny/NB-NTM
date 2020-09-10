import random
import numpy as np


def data_set(data_url, vocab_size):
    """process data input."""
    data_list = []
    word_count = []
    with open(data_url) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            id_freqs = line.split()
            doc = {}
            count = 0
            for id_freq in id_freqs:
                items = id_freq.split(':')
                # python starts from 0
                doc[int(items[0]) - 1] = int(items[1])
                count += int(items[1])
            if count > 0:
                data_list.append(doc)
                word_count.append(count)

    data_mat = np.zeros((len(data_list), vocab_size), dtype=np.float)
    for doc_idx, doc in enumerate(data_list):
        for word_idx, count in doc.items():
            data_mat[doc_idx, word_idx] += count

    return data_list, data_mat, word_count


def create_batches(data_size, batch_size, shuffle=True):
    """create index by batches."""
    batches = []
    ids = list(range(data_size))
    if shuffle:
        random.shuffle(ids)
    for i in range(int(data_size / batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(ids[start:end])
    # the batch of whose length is less than batch_size
    rest = data_size % batch_size
    if rest > 0:
        batches.append(ids[-rest:] + [-1] * (batch_size - rest))  # -1 as padding

    return batches


def fetch_batch_data(data, count, idx_batch, vocab_size):
    """fetch input data by batch."""
    batch_size = len(idx_batch)
    data_batch = np.zeros((batch_size, vocab_size))
    count_batch = []
    mask = np.zeros(batch_size)
    for i, doc_id in enumerate(idx_batch):
        if doc_id != -1:
            for word_id, freq in data[doc_id].items():
                data_batch[i, word_id] = freq
            count_batch.append(count[doc_id])
            mask[i] = 1.0
        else:
            count_batch.append(0)

    return data_batch, count_batch, mask


def fetch_whole_data(data, vocab_size):
    whole_data = np.zeros(vocab_size)
    for doc in data:
        for idx, count in doc.items():
            whole_data[idx - 1] += count
    
    return whole_data


def print_topic_word(vocab_dir, save_dir, topic_word, N):
    # print top N words of each topic
    word_list = []
    with open(vocab_dir) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            word_list.append(line.split()[0])
    topic_size, vocab_size = np.shape(topic_word)

    with open(save_dir, 'w', encoding='utf-8') as fout:
        for topic_idx in range(topic_size):
            top_word_list = []
            print('-------------------------------------- topic ', topic_idx, '--------------------------------------',
                  file=fout)
            top_word_idx = np.argsort(topic_word[topic_idx, :])[-N:]
            for i in range(N):
                top_word_list.append(word_list[top_word_idx[i]])

            # print words
            for word in top_word_list:
                print(word, ' ', end='', file=fout)
            print('\n', file=fout)

    print('save done!')


def compute_coherence(doc_word, topic_word, N):
    topic_size, word_size = np.shape(topic_word)
    doc_size = np.shape(doc_word)[0]
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)

    # compute coherence of each topic
    sum_coherence_score = 0.0
    for i in range(topic_size):
        word_array = topic_list[i]
        sum_score = 0.0
        for n in range(N):
            flag_n = doc_word[:, word_array[n]] > 0
            p_n = np.sum(flag_n) / doc_size
            for l in range(n + 1, N):
                flag_l = doc_word[:, word_array[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_n * p_l * p_nl > 0:
                    p_l = p_l / doc_size
                    p_nl = p_nl / doc_size
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        sum_coherence_score += sum_score * (2 / (N * N - N))
    sum_coherence_score = sum_coherence_score / topic_size
    return sum_coherence_score
