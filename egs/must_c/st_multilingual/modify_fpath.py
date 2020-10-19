import os
import json
import argparse


def modify_fpath(input_dir, new_path=''):
    for split in os.listdir(input_dir):
        dir_path = os.path.join(input_dir, split, "deltafalse")
        new_dir_path = os.path.join(new_path, split, "delatafalse") if bool(new_path) else dir_path
        jfiles = [f for f in os.listdir(dir_path) if f.endswith('.json')]

        for jfile in jfiles:
            jpath = os.path.join(dir_path, jfile)
            if jpath.endswith('.json'):
                with open(jpath, "r") as f:
                    data = json.load(f)["utts"]
                    print(f'Number of utterances: {len(data)}')
                    for k, v in data.items():
                        data[k]['input'][0]['feat'] = os.path.join(new_dir_path, v["input"][0]["feat"].split("/")[-1])

                tmp_jpath = os.path.join(dir_path, 'tmp_' + jfile.split('.')[0] + '.json')
                os.rename(jpath, tmp_jpath)

                new_data = {"utts": data}
                print(f'Saving json file with modified path to {jpath}')
                with open(jpath, "w") as f:
                    json.dump(new_data, f)
            
                print(f'Removing {tmp_jpath}')
                os.remove(tmp_jpath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, help='Local path to `dump` directory.')
    parser.add_argument('--new-path', default='', type=str, help='Local path to `dump` directory.')
    args = parser.parse_args()

    modify_fpath(args.input_dir, args.new_path)

if __name__ == "__main__":
    main()