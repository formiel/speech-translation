import os

def main():
    rootdir = './espnet'

    for subdir, dirs, files in os.walk(rootdir):
        # for file in files:
        #     print(os.path.join(subdir, file))
        # print(subdir)
        init_file = os.path.join(subdir, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, mode='a'): pass

if __name__ == "__main__":
    main()