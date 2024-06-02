import numpy as np
import pandas as pd
from preprocess_data import SEMITONES

# from tqdm import tqdm

# size of values range of each event type
RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_TIME_SHIFT = 100
RANGE_VEL = 32

MAX_TOKEN_SEQUENCE = 4000

# each event we want to convert to int
# So, different events has different values range
# 'note_on': [0; RANGE_NOTE_ON)
# 'note_off': [RANGE_NOTE_ON; RANGE_NOTE_ON + RANGE_NOTE_OFF)
# 'time_shift': [RANGE_NOTE_ON + RANGE_NOTE_OFF; RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)
# 'velocity': [RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT; 'end_of_scope')

win_min = 0.5
win_max = 2.5


def parse_token(token):
    if token < RANGE_NOTE_ON:
        return 'note_on', token
    if token < RANGE_NOTE_ON + RANGE_NOTE_OFF:
        return 'note_off', token - RANGE_NOTE_ON
    if token < RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT:
        return 'time_shift', token - (RANGE_NOTE_ON + RANGE_NOTE_OFF)
    return 'velocity', token - (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)


def time_shifts_array(chromas):
    time_shifts = [] # храним (индекс токена в исходном списке, текущее время) (сумма предыдущих time_shifts)
    curr_time = 0
    for i, token in enumerate(chromas):
        name, value = parse_token(token)
        if name == 'time_shift':
            curr_time += value
            time_shifts.append((i, curr_time))
    return time_shifts

# last time shift == sum of all timeshifts
# at token <timeshift> time **after** this event occurs


def make_piece_from_chromas(time_shifts, chromas, j, i):
    if j == -1:
        l = -1
    else:
        l = time_shifts[j][0]
    if i == len(time_shifts) - 1:
        r = len(chromas) - 1
    else:
        r = time_shifts[i][0]

    return np.sum(chromas[l + 1: r + 1], axis=0) / (r - l)


def make_shifted_pieces(piece):
    output_vec = np.zeros((SEMITONES, 2 * SEMITONES))

    for root in range(SEMITONES):
        output_vec[root, :SEMITONES] = np.roll(piece[:SEMITONES], -root)
        output_vec[root, SEMITONES:] = np.roll(piece[SEMITONES:], -root)
    return output_vec


def nochord_probs(nochord_model, piece):
    # 0 -- chord presence, 1 -- chord absence
    return nochord_model.predict_proba(piece)


def root_probs(root_model, piece, nochord_probs_vec):
    piece_12_shifts = make_shifted_pieces(piece)
    y_proba = root_model.predict_proba(piece_12_shifts)

    return [nochord_probs_vec[1], *y_proba[:, 1]]


def mode_probs(mode_model, piece, nochord_root_probs_vec):
    probs = np.zeros(2 * SEMITONES + 1)
    # probs[0] = nochord_root_probs_vec[0] # remove nochord label : set prob = 0
    probs[0] = 0.

    majmin_probs = mode_model.predict_proba(make_shifted_pieces(piece))
    major_probs = majmin_probs[:, 1]
    minor_probs = majmin_probs[:, 0]

    for root in range(SEMITONES):
        # +1 because 0th elem is a chord absence prob, from 1..24 -- chords probabilities
        probs[root + 1] = major_probs[root] * nochord_root_probs_vec[root + 1]
        probs[root + SEMITONES + 1] = minor_probs[root] * nochord_root_probs_vec[root + 1]
        
        # 1.naumtsev
        # p(root, mode) = p(chord) * p(root, mode | chodr) <- we know this only
        # p(root, mode | nochord) <- dont know this (train only on chord peaces)
        # chord_probs = 1.0 - probs[0]
        # probs[root + 1] = chord_probs * major_probs[root] * nochord_root_probs_vec[root + 1]
        # probs[root + SEMITONES + 1] = chord_probs * minor_probs[root] * nochord_root_probs_vec[root + 1]
        
        # 2.naumtsev
        # TRESHOLD_EXISTS_CHORD = 0.5
        # chord_probs = 1.0 - probs[0]
        # if chord_probs > TRESHOLD_EXISTS_CHORD:
        #     probs[0] = 0.
            
        # probs[root + 1] = chord_probs * major_probs[root] * nochord_root_probs_vec[root + 1]
        # probs[root + SEMITONES + 1] = chord_probs * minor_probs[root] * nochord_root_probs_vec[root + 1]
        # 2-2.naumtsev
        # TRESHOLD_EXISTS_CHORD = 0.5
        # chord_probs = 1.0 - probs[0]
        # if chord_probs > TRESHOLD_EXISTS_CHORD:
        #     probs[0] = 0.
        #     probs[root + 1] = chord_probs * major_probs[root] * nochord_root_probs_vec[root + 1]
        #     probs[root + SEMITONES + 1] = chord_probs * minor_probs[root] * nochord_root_probs_vec[root + 1]
        # else:
        #     probs[root + 1] = 0.
        #     probs[root + SEMITONES + 1] = 0.
            
        
        # 3.naumtsev
        # TRESHOLD_EXISTS_CHORD = 0.5
        # chord_probs = 1.0 - probs[0]
        # if chord_probs > TRESHOLD_EXISTS_CHORD:
        #     probs[0] = 0.
            
        # probs[root + 1] = major_probs[root] * nochord_root_probs_vec[root + 1]
        # probs[root + SEMITONES + 1] = minor_probs[root] * nochord_root_probs_vec[root + 1]
        
        # 4.naumtsev
        # TRESHOLD_EXISTS_CHORD = 0.5
        # chord_probs = 1.0 - probs[0]
        # if chord_probs > TRESHOLD_EXISTS_CHORD:
        #     probs[0] = 0.
        # else:
        #     probs[root + 1] = 0.
        #     probs[root + SEMITONES + 1] = 0.
        # probs[root + 1] = major_probs[root] * nochord_root_probs_vec[root + 1]
        # probs[root + SEMITONES + 1] = minor_probs[root] * nochord_root_probs_vec[root + 1]
        
        
        # 5.naumtsev
        # TRESHOLD_EXISTS_CHORD = 0.3
        # chord_probs = 1.0 - probs[0]
        # if chord_probs > TRESHOLD_EXISTS_CHORD:
        #     probs[0] = 0.
        # else:
        #     probs[root + 1] = 0.
        #     probs[root + SEMITONES + 1] = 0.
        
        # probs[root + 1] = major_probs[root] * nochord_root_probs_vec[root + 1]
        # probs[root + SEMITONES + 1] = minor_probs[root] * nochord_root_probs_vec[root + 1]
                
        
    return probs


def prob_root_notes(nochord_model, root_model, mode_model, time_shifts, chromas, i, j):
    piece = make_piece_from_chromas(time_shifts, chromas, i, j)
    nochord_probs_vec = nochord_probs(nochord_model, piece)
    nochord_root_probs_vec = root_probs(root_model, piece, nochord_probs_vec)
    probs = mode_probs(mode_model, piece, nochord_root_probs_vec)

    return np.log(np.max(probs)), np.argwhere(probs == np.max(probs)).flatten(), probs


def fill_dp(nochord_model, root_model, mode_model, chromas, time_shifts):
    n = len(time_shifts)
    dp = np.array([None] * (n + 1) ** 2).reshape(n + 1, n + 1)

    for idx, ts_time in time_shifts:
        if ts_time < win_min:
            continue
        if ts_time >= win_max:
            break
        dp[0][idx + 1] = prob_root_notes(nochord_model, root_model, mode_model, time_shifts, chromas, -1, idx)

    for idx, ts_time in time_shifts:
        next_token_idx = idx
        next_ts_time = time_shifts[next_token_idx][1]
        while next_ts_time - ts_time <= win_max:
            if next_ts_time - ts_time < win_min:
                next_token_idx += 1
                if next_token_idx >= len(time_shifts):
                    break
                next_ts_time = time_shifts[next_token_idx][1]
            else:
                dp[idx + 1][next_token_idx + 1] = prob_root_notes(nochord_model, root_model, mode_model, time_shifts, chromas, idx, next_token_idx)
                next_token_idx += 1
                if next_token_idx >= len(time_shifts):
                    break
                next_ts_time = time_shifts[next_token_idx][1]

    return dp


def fill_pref(time_shifts, dp):
    n = len(time_shifts)
    pref_0 = [-100] * (n + 1)  # sum log probs
    pref_1 = [-1] * (n + 1)  # ind of prev best harmony piece
    for j, ts_time in time_shifts:
        if ts_time >= win_min and dp[0][j + 1]:
            pref_0[j + 1] = dp[0][j + 1][0]
            pref_1[j + 1] = 0
            break

    for j, ts_time in time_shifts:
        prev_token_i = j - 1
        prev_ts_time = time_shifts[prev_token_i][1]

        while ts_time - prev_ts_time <= win_max and prev_token_i >= 0:
            if dp[prev_token_i + 1][j + 1] and pref_0[prev_token_i + 1] + dp[prev_token_i + 1][j + 1][0] > pref_0[j + 1]:
                pref_0[j + 1] = pref_0[prev_token_i + 1] + dp[prev_token_i + 1][j + 1][0]
                pref_1[j + 1] = prev_token_i + 1
            prev_token_i -= 1
            prev_ts_time = time_shifts[prev_token_i][1]
    return pref_0, pref_1


def optimal_pieces_inds_array(pref_1):
    optimal_pieces_inds = []
    ind = len(pref_1) - 1
    while pref_1[ind] == -1:
        ind -= 1
    while ind >= 0:
        optimal_pieces_inds.append(ind)
        ind = pref_1[ind]

    return optimal_pieces_inds[::-1]


def harmony_seq(optimal_pieces_inds, time_shifts, dp):
    harmonies = []
    times = []
    curr_step = 0
    l = 0
    r = optimal_pieces_inds[curr_step + 1]
    res = []

    while curr_step < len(optimal_pieces_inds) - 1:
        harmony = dp[l][r][1][0]
        harmonies.append(harmony)
        # naumtsevalex: end_l end_r ok? 
        time = (time_shifts[l][1], time_shifts[r][1])
        times.append(time)
        res.append([time[0], time[1], int(harmony)])

        l = optimal_pieces_inds[curr_step]
        r = optimal_pieces_inds[curr_step + 1]
        curr_step += 1
    return res


def harmony_seq_pipeline(chromas, time_shifts, nochord_model, root_model, mode_model):
    # tokens_df = pd.read_csv(file)
    # tokens = tokens_df['tokens'].tolist()

    # use code below to convert .nc file from Kostka-Payne dataset to MusicTransformer tokens

    # from kp_corpus_process import convert_nc_to_tokens
    # tokens = convert_nc_to_tokens(file)
    #
    # import os
    # name = os.path.splitext(os.path.basename(file))[0]
    #
    # pd.DataFrame({'tokens': tokens}).to_csv(f'/home/azatvaleev/harmony_labeling/kp-labeling/{name}.tsv', sep='\t', index=False)

    # time_shifts = time_shifts_array(tokens)
    time_shifts_idx = list(enumerate(time_shifts))

    dp = fill_dp(nochord_model, root_model, mode_model, chromas, time_shifts_idx)

    pref_0, pref_1 = fill_pref(time_shifts_idx, dp)

    optimal_pieces_inds = optimal_pieces_inds_array(pref_1)
    res = harmony_seq(optimal_pieces_inds, time_shifts_idx, dp)

    # naumtsevalex wtf?
    if res[0] == res[1]:
        res.pop(0)

    res[0][0] = 0.

    return res
