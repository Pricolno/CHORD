import os
import numpy as np
import json

DATA_PATH = '/storage/naumtsevalex/harmony/data'

def read_json(json_path):
    with open(json_path, 'r') as fp:
        res = json.load(fp)
    return res

def save_txt_file(file_path, txt):
    with open(file_path, 'w') as f:
        f.write(txt)
    

def calc_mean_score(labels_path):
    with open(labels_path, 'r') as fp:
        result_labelng_json = json.load(fp)
    
    mean_score = 0
    for key in result_labelng_json.keys():
        # print(result_labelng_json[key]['score'])
        mean_score += result_labelng_json[key]['score']
    mean_score /= len(result_labelng_json)
    return mean_score

# def save_lab2_file(label_json, saved_path):

def convert_labeling_to_str(labeling):
    # notes = ['A','Bb','B','C','Db','D','Eb','E','F','Gb','G','Ab']
    notes = ['C','Db','D','Eb','E','F','Gb','G','Ab', 'A','Bb','B']
    
    comp_str = ''
    for l, r, y in labeling:
        if y == 0:
            y_str = 'N' 
        elif 1 <= y <= 12:
            y_str = f'{notes[y - 1]}:maj'
        elif 13 <= y <= 24:
            y_str = f'{notes[y - 13]}:min'
        
        piece_str = f'{l}\t{r}\t{y_str}\n'
        comp_str += piece_str
    
    return comp_str
        
    

def label_json2file_label(json_path, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    result_labels = read_json(json_path)
    print(f'Labeling saved in {dst_dir}')
    for comp_id in result_labels:
        cur_comp_dir = os.path.join(dst_dir, f'{comp_id}')
        os.makedirs(cur_comp_dir, exist_ok=True)
        gt_label_str = convert_labeling_to_str(result_labels[comp_id]['gt_labeling']) 
        pred_label_str = convert_labeling_to_str(result_labels[comp_id]['pred_labeling']) 
        save_txt_file(os.path.join(cur_comp_dir, 'gt_label.lab'), gt_label_str)
        save_txt_file(os.path.join(cur_comp_dir, 'pred_label.lab'), pred_label_str)
        
        
        score = result_labels[comp_id]['score']
        save_txt_file(os.path.join(cur_comp_dir, 'score.lab'), f'{score=}')
        


if __name__ == "__main__":
    npy_root = os.path.join(DATA_PATH, 'castom_npy')
    all_chroma_vectors =  np.load(os.path.join(npy_root, '01_all_chroma_vectors.npy'))
    all_chord_labels =  np.load(os.path.join(npy_root, '01_all_chord_labels.npy'))
    print(f"{all_chord_labels.shape=}\n{all_chroma_vectors.shape=}\n")
    # print(f"{np.max(all_chord_labels)=}")
    print(all_chord_labels[:10])
    print(all_chroma_vectors[:3])
    print()
    
    X_merged =  np.load(os.path.join(npy_root, 'X_merged.npy'))
    y_train_merged =  np.load(os.path.join(npy_root, 'y_train_merged.npy'))
    print(f"{X_merged.shape=}\n{y_train_merged.shape=}\n")
    
    
    all_chroma_vectors_extended = np.load(os.path.join(npy_root, '01_all_chroma_vectors_extended.npy'))
    all_chord_labels_extended = np.load(os.path.join(npy_root, '01_all_chord_labels_extended.npy'))
    print(f"{all_chroma_vectors_extended.shape=}\n{all_chord_labels_extended.shape=}")
    print(all_chroma_vectors_extended[:3])
    print(all_chord_labels_extended[:3])
    print()
    
    
    # print(all_chord_labels_extended[:3])
    # print(all_chord_labels[:3])
    
    
    #mean_score=0.6476877612123901
    # labels_path = '/storage/naumtsevalex/harmony/data/castom_npy/fold_0_win_min_0.5_win_max_2.5_wo_product_w_random_pieces/labeling.json'
    
    # mean_score=0.6026907016098273
    # labels_path = '/storage/naumtsevalex/harmony/data/castom_npy/2_multChordProb_fold_{i}_win_min_0.5_win_max_2.5_wo_product_w_random_pieces/labeling.json' 
    
    # mean_score=0.4217606037788068
    # labels_path = '/storage/naumtsevalex/harmony/data/castom_npy/multChordProb_fold_{i}_win_min_0.5_win_max_2.5_wo_product_w_random_pieces/labeling.json' 
    
    
    # mean_score=0.6026907016098273
    # labels_path = '/storage/naumtsevalex/harmony/data/castom_npy/2-2_multChordProb_fold_{i}_win_min_0.5_win_max_2.5_wo_product_w_random_pieces/labeling.json' 
    
    # mean_score=0.6475922898720105
    # labels_path = '/storage/naumtsevalex/harmony/data/castom_npy/3_multChordProb_fold_{i}_win_min_0.5_win_max_2.5_wo_product_w_random_pieces/labeling.json' 
    # labels_path = '/storage/naumtsevalex/harmony/data/results/00_test_folder_nSplits_10/woNochord_withNegSamples/woNochord_withNegSamples_labeling.json' 
    labels_path = '/storage/naumtsevalex/harmony/data/results/00_test_folder_nSplits_10/woNochord_withNegSamples_v2_mask050Last/woNochord_withNegSamples_v2_mask050Last_labeling.json'
    
    # Our:
    # with no-chord accur: 0.6439007792558303
    # without no-chord(new without negative) (0.6283534082500749,,) 
    # without no-chord(new wit negative): 0.6449914486717138,)
    
    # Article:
    # Chord accuracy 67.33
    # Non-no-chord accuracy 70.03
    # WCSR (root) 74.77
    # WCSR (majmin) 70.62
    
    mean_score = calc_mean_score(labels_path)
    print(f"{labels_path=}\n{mean_score=}\n")
    
    
    # print()
    # composition_idx = all_chord_labels_extended[:, 0].astype(int)
    # composition_idx_unique = np.unique(composition_idx)
    # print(f"{len(composition_idx_unique)}")
    
    # result_label_dir = '/storage/naumtsevalex/harmony/data/castom_npy/drop_nochord_fold_0_win_min_0.5_win_max_2.5_wo_product_w_random_pieces'
    # result_label_path = os.path.join(result_label_dir, 'labeling.json')
    # result_label_path = '/storage/naumtsevalex/harmony/data/results/00_test_folder_nSplits_10/woNochord_withNegSamples/woNochord_withNegSamples_labeling.json'
    # result_label = read_json(result_label_path)
    # print(result_label['22'].keys())
    
    # print([result_label[comp_id]['score'] for comp_id in result_label.keys()])
    
    # label_json2file_label(result_label_path, os.path.join(results_label_dir, 'labs'))