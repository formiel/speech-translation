import os
import shutil
import json
import subprocess
import argparse

INPUT_DIR = '/gpfswork/rech/dbn/umz16dj/espnet/egs/must_c/st_multilingual/dump'
INPUT_DIR_DICT = '/gpfswork/rech/dbn/umz16dj/espnet/egs/must_c/st_multilingual/data/lang_1spm'
OUTPUT_DIR_DEBUG = '/gpfswork/rech/dbn/umz16dj/Data/MUSTC_debug_v2'
OUPUT_DIR_ALL = '/gpfswork/rech/dbn/umz16dj/Data/MUSTC_espnet_v2'

NUM_SAMPLES = [10, 12, 18, 16, 17, 10, 11, 10]
NUM_SAMPLES_TST = [1, 2, 3, 2, 1, 3, 1, 1]

def get_info(tgt_langs="de_es_fr_it_nl_pt_ro_ru"):
    pairs = ['en-'+ l for l in tgt_langs.split('_')]
    splits = ['train_sp', 'dev', 'tst-COMMON', 'tst-HE']
    if len(pairs) == 8:
        suffix = '_v2'
    elif len(pairs) == 2:
        suffix = '_v2.1'
    else: 
        suffix = ''
    return pairs, splits, suffix


def split_small_data_mustc(input_path, output_path, num_samples=10):
    with open(input_path, 'r') as f:
        data = json.load(f)['utts']
    keys = list(data.keys())

    new_data = {'utts' : {}} 
    for k in keys[:num_samples]:
        new_data['utts'][k] = data[k]

    with open(output_path, 'w') as f:
        json.dump(new_data, f)
        print('Finished splitting data from {} to {}.'.format(input_path, output_path))


def create_small_data(tgt_langs="de_es_fr_it_nl_pt_ro_ru"):
    """
    Create small datasets with same directory structure as original data
    """
    pairs, splits, suffix = get_info(tgt_langs=tgt_langs)

    i=0

    for idx, p in enumerate(pairs):
        for s in splits:
            if len(pairs) > 1:
                fname = f'data-{p}_bpe8000.tc{suffix}.json'
            else:
                fname = 'data_bpe8000.tc.json'
            input_path = os.path.join(INPUT_DIR, '{}.{}.{}'.format(s, p, p.split('-')[-1]), 'deltafalse', fname)

            if len(pairs) > 1:
                output_path = os.path.join(OUTPUT_DIR_DEBUG, tgt_langs, p, '{}.{}.{}'.format(s, p, p.split('-')[-1]))
            else:
                output_path = os.path.join(OUTPUT_DIR_DEBUG, tgt_langs, '{}.{}.{}'.format(s, p, p.split('-')[-1]), 'deltafalse')

            os.makedirs(output_path, exist_ok=True)
            if 'tst' in s:
                split_small_data_mustc(input_path, os.path.join(output_path, fname), num_samples=NUM_SAMPLES_TST[idx])
            else:
                split_small_data_mustc(input_path, os.path.join(output_path, fname), num_samples=NUM_SAMPLES[idx])
            i += 1
    print('Total numbers of files: {}'.format(i))


def check_small_data(tgt_langs="de_es_fr_it_nl_pt_ro_ru"):
    pairs, splits, suffix= get_info(tgt_langs=tgt_langs)

    for idx, p in enumerate(pairs):
        for s in splits:
            if len(pairs) > 1:
                output_path = os.path.join(OUTPUT_DIR_DEBUG, tgt_langs, p, '{}.{}.{}'.format(s, p, p.split('-')[-1]))
                fname = f'data-{p}_bpe8000.tc{suffix}.json'
            else:
                output_path = os.path.join(OUTPUT_DIR_DEBUG, tgt_langs, '{}.{}.{}'.format(s, p, p.split('-')[-1]), 'deltafalse')
                fname = 'data_bpe8000.tc.json'
            num_samples = check_json(os.path.join(output_path, fname))
            print(f'Number of samples in {os.path.join(output_path, fname)}: {num_samples}')
            if 'tst' in s:
                assert num_samples == NUM_SAMPLES_TST[idx]
            else:
                assert num_samples == NUM_SAMPLES[idx]


def create_data_links_jsons(mode="debug", tgt_langs="de_es_fr_it_nl_pt_ro_ru"):
    """
    Create symbolic links for all jsons in the same directory
    """
    pairs, splits, suffix = get_info(tgt_langs=tgt_langs)
    assert mode in ["debug", "all"]
    assert len(pairs) > 1

    if mode == "debug":
        input_dir = os.path.join(OUTPUT_DIR_DEBUG, tgt_langs)
        output_dir = input_dir
    elif mode == "all":
        input_dir = INPUT_DIR
        output_dir = os.path.join(OUPUT_DIR_ALL, tgt_langs)

    for s in splits:
        os.makedirs(os.path.join(output_dir, s), exist_ok=True)
        for p in pairs:
            fname = f'data-{p}_bpe8000.tc{suffix}.json'
            if mode == "debug":
                ip = os.path.join(input_dir, p, '{}.{}.{}'.format(s, p, p.split('-')[-1]), fname)
                op = os.path.join(output_dir, s, '{}.json'.format(p))
            elif mode == "all":
                ip = os.path.join(input_dir, '{}.{}.{}'.format(s, p, p.split('-')[-1]), "deltafalse", fname)
                op = os.path.join(output_dir, s, '{}.json'.format(p))
            print('{} -> {}'.format(ip, op))
            subprocess.call(["ln", "-s", ip, op])


def create_data_links_dicts(mode="debug", tgt_langs="de_es_fr_it_nl_pt_ro_ru"):
    """
    Create symbolic links for all dictionary in the same directory
    """
    pairs, splits, suffix = get_info(tgt_langs=tgt_langs)
    assert mode in ["debug", "all"]
    assert len(pairs) > 1

    if mode == "debug":
        output_dir = os.path.join(OUTPUT_DIR_DEBUG, tgt_langs, 'lang_1spm')
    elif mode == "all":
        output_dir = os.path.join(OUPUT_DIR_ALL, tgt_langs, 'lang_1spm')
    os.makedirs(output_dir, exist_ok=True)

    fname = f'train_sp.en-{tgt_langs}.{tgt_langs}_bpe8000_units_tc{suffix}.txt'
    ip = os.path.join(INPUT_DIR_DICT, fname)
    op = os.path.join(output_dir, fname)
    print('{} -> {}'.format(ip, op))
    subprocess.call(["ln", "-s", ip, op])


def check_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)['utts']
    return len(data.keys())


def read_json_dir(json_dir):
    json_files = [os.path.join(json_dir, fname) for fname in os.listdir(json_dir)]
    return json_files


def compare_features(lang, split):
    """
    Compare feature paths in json_to_check and json_ref
    """
    dir_to_check = '/gpfsscratch/rech/wod/umz16dj/Data/MUSTC_test/debug_multi_pairs_shared_vocab'
    json_to_check = f'{dir_to_check}/en-{lang}/{split}.en-{lang}.{lang}/data-en-{lang}_bpe8000.tc.json'
    dir_ref = '/gpfswork/rech/wod/umz16dj/espnet/egs/must_c/st1/dump'
    json_ref = f'{dir_ref}/{split}.en-{lang}.{lang}/deltafalse/data_bpe8000.tc.json'

    with open(json_to_check, 'r') as f:
        js_data = json.load(f)['utts']
    keys = list(js_data.keys())
    fpaths = [js_data[k]['input'][0]['feat'] for k in keys]

    with open(json_ref, 'r') as f:
        js_data_ref = json.load(f)['utts']
    fpaths_ref = [js_data_ref[k]['input'][0]['feat'] for k in keys]

    for i, p in enumerate(fpaths):
        print(f'new: {p}')
        print(f'ref: {fpaths_ref[i]}')
        assert p == fpaths_ref[i]
    
    return


def rename_subfolders(path_to_dir, str_to_remove, str_to_replace=""):

    for sf in os.listdir(path_to_dir):
        if str_to_remove in sf:
            sf_new = sf.replace(str_to_remove, str_to_replace)
            old_path = os.path.join(path_to_dir, sf)
            new_path = os.path.join(path_to_dir, sf_new)
            print(f'old_path: {old_path} to new_path: {new_path}')
            os.rename(old_path, new_path)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--toto', type=str)
    args = parser.parse_args()

    return args


def move_folders(old_path, new_path=None, term="_lang_de_nl"):

    for f in os.listdir(old_path):
        if term not in f:
            if new_path is not None:
                print(f'Moving {f} from {old_path} to {new_path}')
                shutil.move(os.path.join(old_path, f), os.path.join(new_path, f))
            else:
                print(f'Removing {f}')
                try:
                    shutil.rmtree(os.path.join(old_path, f))
                except:
                    os.remove(os.path.join(old_path, f))


def main():

    modes = ['debug', 'all']
    tgt_langs = ['de_nl', 'de_es_fr_it_nl_pt_ro_ru']

    for tgt_lang in tgt_langs:
        for mode in modes:
            if mode == 'debug':
                create_small_data(tgt_langs=tgt_lang)
                check_small_data(tgt_langs=tgt_lang)

            create_data_links_jsons(mode=mode, tgt_langs=tgt_lang)
            create_data_links_dicts(mode=mode, tgt_langs=tgt_lang)


# def main():
#     path = "/Users/hang/Google Drive/Research/mustc/tensorboard"
#     move_folders(old_path=path)

if __name__ == "__main__":
    main()