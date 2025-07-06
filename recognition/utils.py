import torch
import torch.nn as nn
import editdistance  # ensure `pip install editdistance` if not installed
from recognition.dataset import CHARS  # fixed import

def ctc_loss_fn(preds, labels, pred_lengths, label_lengths):
    """
    Compute CTC loss given logits and ground-truth labels/lengths.
    preds: [T, B, num_classes] (logits, NOT softmaxed)
    labels: [B, max_len] (padded with 0 for blanks)
    pred_lengths: [B] length of each prediction (usually T for all)
    label_lengths: [B] length of each label sequence
    """
    log_probs = preds.log_softmax(2)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = ctc_loss(log_probs, labels, pred_lengths, label_lengths)
    return loss

def label_to_string(indices):
    """
    Convert a sequence of label indices to a string.
    Skip index 0 (CTC blank). Maps 1→CHARS[0], 2→CHARS[1], etc.
    """
    s = ""
    for idx in indices:
        if idx != 0 and idx <= len(CHARS):
            s += CHARS[idx - 1]
    return s

def compute_ner(pred_logits, label_indices, label_lengths):
    """
    Compute Normalized Edit Distance (i.e., Character Error Rate) for a batch.
    pred_logits: [T, B, num_classes] model output
    label_indices: [B, max_len] padded ground-truth indices
    label_lengths: [B] lengths of each ground-truth sequence
    Returns: average CER over the batch (0.0 = perfect, 1.0 = 100% error).
    """
    T, B, num_classes = pred_logits.shape
    preds = pred_logits.argmax(dim=2)  # [T, B]
    total_cer = 0.0
    for b in range(B):
        # Collapse repeats and remove blanks for predicted sequence
        pred_seq = []
        prev = 0
        for t in range(T):
            idx = preds[t, b].item()
            if idx != prev and idx != 0:
                pred_seq.append(idx)
            prev = idx
        # Ground-truth sequence (remove padding 0s)
        gt_seq = label_indices[b, :label_lengths[b]].tolist()
        # Compute edit distance between predicted and ground-truth character sequences
        pred_str = [CHARS[i - 1] for i in pred_seq]
        gt_str = [CHARS[i - 1] for i in gt_seq]
        cer = editdistance.eval(pred_str, gt_str) / max(len(gt_str), 1)
        total_cer += cer
    return total_cer / B
