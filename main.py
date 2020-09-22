import torch
import torch.optim as optim
import numpy as np
import time
import os
import NBNTM
import GNBNTM
import utils

# hyper-parameter initialization
model = 'GNBNTM'
shape_prior = 2
scale_prior = 0.5
topic_num = 20
vocab_num = 2000  # 20news & reuters: 2000, MXM: 5000
hidden_num = 256
batch_size = 64
learning_rate = 5e-4
top_word_num = 10  # print how many words for each topic
data_name = '20news'
epochs = 600  # 20news & reuters: 600 (approximate 1 hour), MXM: 100 (approximate 4 hours)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)


def run(net, optimizer, data_list, corpus_word_count, is_train):
    perplexity = torch.tensor(0, dtype=torch.float)
    kld = torch.tensor(0, dtype=torch.float)
    doc_count = torch.tensor(0, dtype=torch.float)

    idx_batches = utils.create_batches(len(data_list), batch_size, shuffle=is_train)
    for idx_batch in idx_batches:
        # get batch data
        batch, batch_word_count, mask = utils.fetch_batch_data(data_list, corpus_word_count,
                                                               idx_batch, vocab_num)
        batch = torch.tensor(batch, dtype=torch.float, device=device)
        batch_word_count = torch.tensor(batch_word_count, dtype=torch.float, device=device)
        mask = torch.tensor(mask, dtype=torch.float, device=device)

        # forward propagation
        shape, scale, lam, out = net(batch)

        # compute batch loss
        batch_likelihood, batch_kld = net.compute_batch_loss(batch, out, shape, scale)
        batch_loss = (batch_likelihood + batch_kld) * mask

        # compute cumulative loss
        perplexity += torch.sum(batch_loss / (batch_word_count + 1e-12)).detach()
        kld += (torch.sum(batch_kld) / torch.sum(mask)).detach()
        doc_count += torch.sum(mask).detach()

        # train or validate
        if is_train:
            optimizer.zero_grad()
            batch_loss.backward(mask)
            optimizer.step()

    perplexity = torch.exp(perplexity / doc_count)
    kld = kld / len(idx_batches)

    return perplexity, kld


def evaluate_coherence(net, doc_word, n_list):
    topic_word = net.out_fc.weight.detach().t()
    topic_word = torch.softmax(topic_word, dim=1).cpu().numpy()
    coherence = 0.0
    for n in n_list:
        coherence += utils.compute_coherence(doc_word, topic_word, n)

    coherence /= len(n_list)

    return coherence


def print_result(epoch, state, perplexity, kld):
    print('| Epoch ', state, ': {:d} |'.format(epoch + 1),
          '| Perplexity: {:.5}'.format(perplexity),
          '| KLD: {:.5}'.format(kld))
    if state == 'test':
        print('\n')


def record_result(result_dir, result_list):
    with open(result_dir, 'w') as fout:
        for i in result_list:
            fout.write(str(i))
            fout.write('\n')


def main():
    # select model: NB or GNB
    if model == 'NBNTM':
        net = NBNTM.NBNTM(device, vocab_num, hidden_num, topic_num, shape_prior, scale_prior)
    else:
        net = GNBNTM.GNBNTM(device, vocab_num, hidden_num, topic_num, shape_prior, scale_prior)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # load data
    data_dir = 'data/' + data_name + '/'
    train_list, train_mat, train_count = utils.data_set(data_dir + 'train.feat', vocab_num)
    test_list, test_mat, test_count = utils.data_set(data_dir + 'test.feat', vocab_num)
    
    # auxiliary dir setting
    if not os.path.exists('./result'):
        os.mkdir('./result')
        os.mkdir('./result/NBNTM')
        os.mkdir('./result/GNBNTM')
    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')
        os.mkdir('./checkpoint/NBNTM')
        os.mkdir('./checkpoint/GNBNTM')

    flag_str = (data_name + '_shape_' + str(shape_prior) + '_scale_' + str(scale_prior)
                + '_K_' + str(topic_num) + '_V_' + str(vocab_num) + '_H_' + str(hidden_num)
                + '_batch_' + str(batch_size) + '_lr_' + str(learning_rate) + '_epoch_' + str(epochs))
    result_dir = './result/' + model + '/' + flag_str
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # record in file
    train_ppl_time = []
    best_train_ppl = 1e12
    best_coherence = -1

    start_time = time.time()
    addition_time = 0
    for epoch in range(epochs):
        # train
        perplexity, kld = run(net, optimizer, train_list, train_count, True)
        current_time_cost = time.time() - start_time
        train_ppl_time.append([perplexity.detach().item(), current_time_cost])
        print_result(epoch, 'train', perplexity, kld)

        temp_time = time.time()
        # prepare for test
        if epoch % 10 == 9:
            if perplexity < best_train_ppl:
                best_train_ppl = perplexity
                # save model
                state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epochs': epoch}
                torch.save(state, './checkpoint/' + model + '/' + flag_str + '_best_ppl')

            # coherence
            coherence = evaluate_coherence(net, train_mat, [5])
            print('train coherence = ', coherence)

            if coherence > best_coherence:
                best_coherence = coherence
                # save model
                state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epochs': epoch}
                torch.save(state, './checkpoint/' + model + '/' + flag_str + '_best_coherence')
        addition_time += time.time() - temp_time

    end_time = time.time()
    print(f'time cost:{end_time - start_time - addition_time}')

    record_result(result_dir + './train_ppl_time_record', train_ppl_time)

    # test perplexity
    checkpoint = torch.load('./checkpoint/' + model + '/' + flag_str + '_best_ppl', map_location='cuda:0')
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epochs']
    perplexity, kld = run(net, optimizer, test_list, test_count, False)
    print_result(epoch, 'test', perplexity, kld)

    # test coherence
    checkpoint = torch.load('./checkpoint/' + model + '/' + flag_str + '_best_coherence', map_location='cuda:0')
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('whole coherence = ', evaluate_coherence(net, np.concatenate((train_mat, test_mat)), [5, 10, 15]))

    # save topic words
    utils.print_topic_word('data/' + data_name + '/' + data_name + '.vocab', model + '_topic_words.txt', net.out_fc.weight.detach().cpu().t(), 15)


if __name__ == '__main__':
    main()
