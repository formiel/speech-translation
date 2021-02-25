"""
Organize multilingual data to prepare for training
"""
import os
import re
import shutil
import json
import subprocess
import argparse

SPLITS = ['train_sp', 'dev', 'tst-COMMON', 'tst-HE']

def get_info(tgt_langs="de_es_fr_it_nl_pt_ro_ru", use_lid=True, use_joint_dict=True):
    pairs = ['en-'+ l for l in tgt_langs.split('_')]
    if len(pairs) > 1:
        assert use_lid

    prefix = 'dict1'
    if not use_joint_dict:
        prefix = 'dict2'
    
    suffix = f'lgs_{tgt_langs}'
    if len(pairs) == 8:
        suffix = 'lgs_all8'
    elif len(pairs) == 1:
        suffix = f'lgs_{tgt_langs}_id_{use_lid}'

    return pairs, prefix, suffix


def create_data_links_jsons(input_dir, output_dir, 
                            tgt_langs="de_es_fr_it_nl_pt_ro_ru",
                            use_lid=True, use_joint_dict=True,
                            nbpe_src=8000, nbpe=8000):
    """
    Create symbolic links to save jsons in the following structure: 
        output_dir/tgt_langs/use_${prefix}/src${nbpe_src}_tgt${nbpe}/${split}/${lang_pair}.json
        where: 
            - ${split} is "train_sp", "dev", "tst-COMMON", or "tst-HE"
            - ${lang_pair} is "en-de", "en-es", etc.
    """
    pairs, prefix, suffix = get_info(tgt_langs=tgt_langs, use_lid=use_lid, use_joint_dict=use_joint_dict)
    assert len(pairs) > 1
    output_dir = os.path.join(output_dir, tgt_langs, f'use_{prefix}', f'src{nbpe_src}_tgt{nbpe}')

    for s in SPLITS:
        os.makedirs(os.path.join(output_dir, s), exist_ok=True)
        for p in pairs:
            if use_joint_dict:
                fname = f'data_{prefix}_{p}_bpe{nbpe}_tc_{suffix}.json'
            else:
                fname = f'data_{prefix}_{p}_bpe_src{nbpe_src}lc.rm_tgt{nbpe}tc_{suffix}.json'
            src = os.path.join(input_dir, '{}.{}.{}'.format(s, p, p.split('-')[-1]), "deltafalse", fname)
            dst = os.path.join(output_dir, s, '{}.json'.format(p))
            print('{} -> {}'.format(src, dst))
            subprocess.call(["ln", "-s", src, dst])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tgt-langs', default='de_es_fr_it_nl_pt_ro_ru', type=str, 
                        help='Target languages seperated by _')
    parser.add_argument('--input-dir', default='./dump', type=str, 
                        help='Path to directory where features are saved')
    parser.add_argument('--output-dir', type=str,
                        help='Path to directory to save symlinks')
    parser.add_argument('--use-lid', action='store_true',
                        help='Use language ID in the target sequence')
    parser.add_argument('--use-joint-dict', action='store_true',
                        help='Use joint dictionary for source and target')
    parser.add_argument('--nbpe', type=int, default=8000)
    parser.add_argument('--nbpe-src', type=int, default=8000)

    args = parser.parse_args()

    create_data_links_jsons(args.input_dir, args.output_dir, 
                            tgt_langs=args.tgt_langs,
                            use_lid=args.use_lid,
                            use_joint_dict=args.use_joint_dict,
                            nbpe=args.nbpe,
                            nbpe_src=args.nbpe_src)


if __name__ == "__main__":
    main()