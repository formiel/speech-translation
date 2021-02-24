import os
import re
import shutil
import json
import subprocess
import argparse


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


def create_data_links_jsons(input_dir, output_dir, tgt_langs="de_es_fr_it_nl_pt_ro_ru"):
    """
    Create symbolic links to save jsons in the following structure: 
        output_dir/tgt_langs/${split}/${lang_pair}.json
        where: 
            - ${split} is "train_sp", "dev", "tst-COMMON", or "tst-HE"
            - ${lang_pair} is "en-de", "en-es", etc.
    """
    pairs, splits, suffix = get_info(tgt_langs=tgt_langs)
    assert len(pairs) > 1
    output_dir = os.path.join(output_dir, tgt_langs)

    for s in splits:
        os.makedirs(os.path.join(output_dir, s), exist_ok=True)
        for p in pairs:
            fname = f'data-{p}_bpe8000.tc{suffix}.json'
            ip = os.path.join(input_dir, '{}.{}.{}'.format(s, p, p.split('-')[-1]), "deltafalse", fname)
            op = os.path.join(output_dir, s, '{}.json'.format(p))
            print('{} -> {}'.format(ip, op))
            subprocess.call(["ln", "-s", ip, op])


def create_data_links_dicts(input_dir_dict, output_dir, tgt_langs="de_es_fr_it_nl_pt_ro_ru"):
    """
    Create symbolic links to save dictionary under: output_dir/tgt_langs/lang_1spm.
    """
    pairs, splits, suffix = get_info(tgt_langs=tgt_langs)
    assert len(pairs) > 1

    output_dir = os.path.join(output_dir, tgt_langs, 'lang_1spm')
    os.makedirs(output_dir, exist_ok=True)

    fname = f'train_sp.en-{tgt_langs}.{tgt_langs}_bpe8000_units_tc{suffix}.txt'
    ip = os.path.join(input_dir_dict, fname)
    op = os.path.join(output_dir, fname)
    print('{} -> {}'.format(ip, op))
    subprocess.call(["ln", "-s", ip, op])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tgt-langs', default='de_es_fr_it_nl_pt_ro_ru', type=str, 
                        help='Target languages seperated by _')
    parser.add_argument('--input-dir', default='./dump', type=str, 
                        help='Path to directory where features are saved')
    parser.add_argument('--input-dir-dict', default='./data/lang_1spm', type=str, 
                        help='Path to director where dictionaries are saved')
    parser.add_argument('--output-dir', type=str,
                        help='Path to directory to save symlinks')

    args = parser.parse_args()

    create_data_links_jsons(args.input_dir, args.output_dir, tgt_langs=args.tgt_langs)
    create_data_links_dicts(args.input_dir_dict, args.output_dir, tgt_langs=args.tgt_langs)

if __name__ == "__main__":
    main()