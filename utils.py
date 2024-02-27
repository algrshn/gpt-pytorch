import sys
import torch
import torch.nn.functional as F
import numpy as np
from math import log2


def display_progress(batch_num, num_of_batches):
    
    total=num_of_batches-1
    bar_len = 60
    filled_len = int(round(bar_len * batch_num / float(total)))

    percents = round(100.0 * batch_num / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    sys.stdout.flush()
    
    if(percents == 100.0):
        sys.stdout.write('')
        sys.stdout.flush()


def inference_greedy_search(model, ids_list, vocab_size, device, how_many_tokens_to_generate):

    for k in range(how_many_tokens_to_generate):

        prompt_onehot_np = np.zeros((1, len(ids_list), vocab_size), dtype=np.float32)

        for i in range(len(ids_list)):
            prompt_onehot_np[0, i, ids_list[i]] = 1

        prompt_onehot = torch.from_numpy(prompt_onehot_np).to(device)

        output_pred = model(prompt_onehot)

        pred_id = torch.argmax(F.softmax(torch.squeeze(output_pred, dim=0), dim=0), dim=0).detach().cpu().numpy()[-1]

        ids_list.append(pred_id)

    return ids_list


def inference_beam_search(model, ids_list, vocab_size, device, how_many_tokens_to_generate, beam_size):

    beam = [(ids_list, 0)]
    for k in range(how_many_tokens_to_generate):
        candidates = []

        # Generate candidates for each beam
        for seq, score in beam:
            prompt_onehot_np = np.zeros((1, len(seq), vocab_size), dtype=np.float32)

            for i in range(len(seq)):
                prompt_onehot_np[0, i, seq[i]] = 1

            prompt_onehot = torch.from_numpy(prompt_onehot_np).to(device)

            output_pred = model(prompt_onehot)

            topk_probs, topk_ids = torch.topk(F.softmax(torch.squeeze(output_pred, dim=0), dim=0)[:,-1], beam_size)

            for i in range(beam_size):
                candidate_seq = seq + [topk_ids[i].item()]
                candidate_score = score - log2(topk_probs[i].item())
                candidates.append((candidate_seq, candidate_score))

        # Select top-k candidates
        candidates.sort(key=lambda x: x[1])
        beam = candidates[:beam_size]

    return beam[0][0]

