pip install --user --no-cache-dir
module load python/3.7.5 cuda/10.1.1 cudnn/7.6.5.32-cuda-10.1 nccl/2.6.4-1-cuda gcc/7.3.0 openmpi/4.0.2-cuda git/2.21.0

#SBATCH --qos=qos_gpu-t4

## Installation of ESPNet in JZ
**Build kaldi**
module load \
    pytorch-gpu/py3/1.4.0 \
    automake/1.16.1 \
    autoconf/2.69 \
    git/2.21.0 \
    sox/14.4.2 \
    subversion/1.9.7 \
    openblas/0.3.6 \
    intel-mkl/19.0.5 \
    ffmpeg/N-94431

After installing kaldi, make sure there's no kaldi/tools/env.sh and no kaldi/tools/python/python
(otherwise there will be "no module sentencepiece" etc.)

cd kaldi/tools
bash extras/check_dependencies.sh
touch python/.use_default_python
make -j$(nproc)

cd kaldi/src
./configure --shared \
    --use-cuda=no \
    --mkl-root=/gpfslocalsys/intel/parallel_studio_xe_2019_update5_cluster_edition/compilers_and_libraries_2019.5.281/linux/mkl \
    --mkl-libdir=/gpfslocalsys/intel/parallel_studio_xe_2019_update5_cluster_edition/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin
make depend -j$(nproc)
make -j$(nproc)

DOWNLOAD_DIR=$WORK/setup/ make -j10

ZLIB
rm -f /home/khue/.local/lib/libz.a
cp libz.a /home/khue/.local/lib
chmod 644 /home/khue/.local/lib/libz.a
cp libz.so.1.2.11 /home/khue/.local/lib
chmod 755 /home/khue/.local/lib/libz.so.1.2.11
rm -f /home/khue/.local/share/man/man3/zlib.3
cp zlib.3 /home/khue/.local/share/man/man3
chmod 644 /home/khue/.local/share/man/man3/zlib.3
rm -f /home/khue/.local/lib/pkgconfig/zlib.pc
cp zlib.pc /home/khue/.local/lib/pkgconfig
chmod 644 /home/khue/.local/lib/pkgconfig/zlib.pc
rm -f /home/khue/.local/include/zlib.h /home/khue/.local/include/zconf.h
cp zlib.h zconf.h /home/khue/.local/include
chmod 644 /home/khue/.local/include/zlib.h /home/khue/.local/include/zconf.h

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH


WGET:
dependencies (should be installed in order): gmp, nettle, openssl, unbound, gnutls

----------------------------------------------------------------------
Libraries have been installed in:
   /gpfswork/rech/wod/umz16dj/kaldi/tools/openfst-1.6.7/lib/fst

If you ever happen to want to link against installed libraries
in a given directory, LIBDIR, you must either use libtool, and
specify the full pathname of the library, or use the `-LLIBDIR'
flag during linking and do at least one of the following:
   - add LIBDIR to the `LD_LIBRARY_PATH' environment variable
     during execution
   - add LIBDIR to the `LD_RUN_PATH' environment variable
     during linking
   - use the `-Wl,-rpath -Wl,LIBDIR' linker flag
   - have your system administrator add LIBDIR to `/etc/ld.so.conf'

See any operating system documentation about shared libraries for
more information, such as the ld(1) and ld.so(8) manual pages.
----------------------------------------------------------------------

----------------------------------------------------------------------
Libraries have been installed in:
   /gpfswork/rech/wod/umz16dj/kaldi/tools/openfst-1.6.7/lib

If you ever happen to want to link against installed libraries
in a given directory, LIBDIR, you must either use libtool, and
specify the full pathname of the library, or use the `-LLIBDIR'
flag during linking and do at least one of the following:
   - add LIBDIR to the `LD_LIBRARY_PATH' environment variable
     during execution
   - add LIBDIR to the `LD_RUN_PATH' environment variable
     during linking
   - use the `-Wl,-rpath -Wl,LIBDIR' linker flag
   - have your system administrator add LIBDIR to `/etc/ld.so.conf'

See any operating system documentation about shared libraries for
more information, such as the ld(1) and ld.so(8) manual pages.
----------------------------------------------------------------------

/gpfslocalsys/intel/parallel_studio_xe_2019_update5_cluster_edition/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin

**ESPNET**
cd espnet
ln -s $WORK/kaldi tools/kaldi
TMPDIR=$WORK/tmp pip install --user . -vv
cd tools
git clone https://github.com/moses-smt/mosesdecoder.git moses

**Install espnet** 
cd $WORK/espnet
TMPDIR=$WORK/tmp pip install --user . -vv

TMPDIR=$WORK/tmp pip install --user . && pip install --user -e .

**Test in interactive mode**
salloc --nodes=1 --ntasks=4 --cpus-per-task=10 --ntasks-per-node=4 --gres=gpu:4 --partition=gpu_p1 --time=20:00:00
srun --ntasks=4 --gres=gpu:4 --pty bash 

./run.sh  --stage 0 --stop-stage 2

./run.sh  --ngpu 4 --stage 4 --stop-stage 5 --train-config ./conf/train.yaml |& tee output.txt


## Commonly used commands

**Download data from JZ**
rsync -chavzP --stats \
    umz16dj@jean-zay.idris.fr:/gpfswork/rech/dbn/umz16dj/espnet/egs/must_c/st_multilingual/tensorboard/* \
    /Users/hang/Google\ Drive/Research/mustc/tensorboard/

rsync -chavzP --stats \
    umz16dj@jean-zay.idris.fr:/gpfswork/rech/dbn/umz16dj/espnet/egs/must_c/st_multilingual/tensorboard_debug/* \
    /Users/hang/Google\ Drive/Research/mustc/tensorboard_debug/

rsync -chavzP --stats /gpfsscratch/rech/dbn/umz16dj/Experiments_espnet/exp/* /gpfsstore/rech/dbn/umz16dj/Experiments_espnet/exp/

rsync -chavzP --stats \
    umz16dj@jean-zay.idris.fr:/gpfsscratch/rech/dbn/umz16dj/Experiments_espnet/debug/debug_decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl/decode_debug_decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl_common_decode_separate_tst-COMMON_en-de_model.acc.best \
    /Users/hang/Data/mustc/

rsync -chavzP --stats \
    /gpfsscratch/rech/dbn/umz16dj/Experiments_espnet/exp/* \
    /gpfsstore/rech/dbn/umz16dj/Experiments_espnet/exp/

scp -3 -r \
    lethip@decore2.imag.fr:/home/getalp/lethip/FBK-Fairseq-ST \
    umz16dj@jean-zay.idris.fr:/gpfswork/rech/wod/umz16dj/

scp -3 -r lethip@decore2.imag.fr:/home/getalp/lethip/shared/MUSTC_v1.0 umz16dj@jean-zay.idris.fr:/gpfsstore/rech/dbn/umz16dj/Data/

scp -3 -r umz16dj@jean-zay.idris.fr:/gpfswork/rech/dbn/umz16dj/espnet/egs/must_c/st_multilingual/exp lethip@decore2.imag.fr:/home/getalp/lethip/Experiments/MUSTC_espnet_v2/

scp -3 -r umz16dj@jean-zay.idris.fr:/gpfsscratch/rech/dbn/umz16dj/Data/MUSTC_espnet_v2/de_nl/tst-COMMON lethip@decore2.imag.fr:/home/getalp/lethip/Data/MUSTC_espnet_v2/

https://dl.fbaipublicfiles.com/simultaneous_translation/mustc_en-it.tar.gz

**Commands to debug en-de using st1**
ngpu=1
stage=5
stop_stage=5

tag=debug_org
tgt_lang=de
resume=

train_config=./conf/tuning/train_pytorch_transformer_short_long_debug_org.yaml
decode_config=./conf/tuning/decode_pytorch_transformer.en-de.yaml
preprocess_config=./conf/specaug.yaml

./run.sh  --ngpu ${ngpu} --stage $stage --stop-stage $stop_stage \
        --tag $tag \
        --tgt-lang ${tgt_lang} \
        --train-config ${train_config} \
        --decode-config ${decode_config} \
        --preprocess-config ${preprocess_config} \
        --dumpdir $SCRATCH/Data/MUSTC_debug/de \
        --verbose 1 \
        --resume $resume


## Investigate outputs
*of Inaguma's experiments using MuST-C in ESPNet*
**results of make_batchset(json_file)**
*adaptive batch size based on maxlen of input (maxlen_in) and maxlen of output (maxlen_out)*
[('sp0.9-ted_00001_0048700_0071650', {'input': [{'feat': '/.../feats.1.ark:126826', 'name': 'input1', 'shape': [2548, 83]}], 'lang': 'de', 
  'output': [{'name': 'target1', 'shape': [21, 7981], 'text': 'Jetzt muss ich meine Schuhe ausziehen , um überhaupt an Bord zu kommen ! ( Applaus )', 'token': '▁Jetzt ▁muss ▁ich ▁meine ▁Sch u he ▁aus ziehen ▁, ▁um ▁überhaupt ▁an ▁B ord ▁zu ▁kommen ▁! ▁( ▁Applaus ▁)', 'tokenid': '3494 6504 5964 6409 3972 2165 923 4560 2477 2720 7569 7967 4468 2904 1714 7920 6183 2715 2718 2862 2719'}, 
              {'name': 'target2', 'shape': [23, 7981], 'text': 'now i have to take off my shoes or boots to get on an airplane laughter applause', 'token': '▁now ▁i ▁have ▁to ▁take ▁off ▁my ▁sh oes ▁or ▁bo ots ▁to ▁get ▁on ▁an ▁air pl ane ▁l aughter ▁appl ause', 'tokenid': '6606 5959 5851 7492 7407 6640 6509 7144 1620 6672 4754 1768 7492 5722 6654 4468 4428 1826 235 6220 390 4510 401'}], 
  
  'utt2spk': 'sp0.9-ted_00001'})]
len(batch) = 1
-----
[('sp0.9-ted_00001_0121570_0135280', {'input': [{'feat': '/.../feats.1.ark:564418', 'name': 'input1', 'shape': [1521, 83]}], 'lang': 'de', 'output': [{'name': 'target1', 'shape': [29, 7981], 'text': 'Haben Sie schon mal vom Phantomschmerz gehört ? ( Lachen ) Wir saßen in einem gemieteten Ford Taurus .', 'token': '▁Haben ▁Sie ▁schon ▁mal ▁vom ▁Ph ant om sch merz ▁gehört ▁? ▁( ▁Lachen ▁) ▁Wir ▁sa ßen ▁in ▁einem ▁gem ie teten ▁F ord ▁T aur us ▁.', 'tokenid': '3367 4021 7066 6365 7717 3857 255 1658 1999 1535 5675 2791 2718 3593 2719 4309 7017 2531 6006 5261 5681 1054 2134 3187 1714 4109 397 2296 2722'}, {'name': 'target2', 'shape': [16, 7981], 'text': 'laughter you &apos;ve heard of phantom limb pain laughter', 'token': '▁l aughter ▁you ▁ &apos; ve ▁heard ▁of ▁ph ant om ▁lim b ▁pain ▁l aughter', 'tokenid': '6220 390 7900 2714 6 2340 5859 6639 6747 255 1658 6302 416 6696 6220 390'}], 'utt2spk': 'sp0.9-ted_00001'})]
len(batch) = 1
-----
[('sp0.9-ted_00001_0035010_0048700', {'input': [{'feat': '/.../feats.1.ark:32', 'name': 'input1', 'shape': [1519, 83]}], 'lang': 'de', 'output': [{'name': 'target1', 'shape': [51, 7981], 'text': 'Das meine ich ernst , teilweise deshalb - weil ich es wirklich brauchen kann ! ( Lachen ) Versetzen Sie sich mal in meine Lage ! ( Lachen ) ( Applaus ) Ich bin bin acht Jahre lang mit der Air Force Two geflogen .', 'token': '▁Das ▁meine ▁ich ▁ernst ▁, ▁teil weise ▁deshalb ▁- ▁weil ▁ich ▁es ▁wirklich ▁brauchen ▁kann ▁! ▁( ▁Lachen ▁) ▁Vers etzen ▁Sie ▁sich ▁mal ▁in ▁meine ▁Lage ▁! ▁( ▁Lachen ▁) ▁( ▁Applaus ▁) ▁Ich ▁bin ▁bin ▁acht ▁Jahre ▁lang ▁mit ▁der ▁A ir ▁For ce ▁Two ▁ge fl ogen ▁.', 'tokenid': '3051 6409 5964 5383 2720 7434 2393 5100 2721 7768 5964 5407 7832 4777 6140 2715 2718 3593 2719 4226 741 4021 7163 6365 6006 6409 3594 2715 2718 3593 2719 2718 2862 2719 3419 4727 4727 4397 3479 6229 6459 5094 2792 1262 3224 489 4179 5651 778 1625 2722'}, {'name': 'target2', 'shape': [20, 7981], 'text': 'and i say that sincerely partly because mock sob i need that laughter', 'token': '▁and ▁i ▁say ▁that ▁s in cer ely ▁part ly ▁because ▁mo ck ▁so b ▁i ▁need ▁that ▁l aughter', 'tokenid': '4474 5959 7042 7457 7016 1196 498 626 6705 1509 4615 6463 526 7208 416 5959 6551 7457 6220 390'}], 'utt2spk': 'sp0.9-ted_00001'})]
len(batch) = 1
-----
[('sp0.9-ted_00001_0087750_0098200', {'input': [{'feat': '/.../feats.1.ark:452013', 'name': 'input1', 'shape': [1159, 83]}], 'lang': 'de', 'output': [{'name': 'target1', 'shape': [95, 7981], 'text': 'Kurz nachdem Tipper und ich aus dem ( vorgetäuschtes Schluchzen ) Weißen Haus ausgezogen waren , fuhren wir von unserem Haus in Nashville zu unserer kleinen Farm 50 Meilen östlich von Nashville - und wir fuhren selbst . ( Lachen ) Ich weiß , für Sie ist das nichts Ungewöhnliches , aber ... ( Lachen ) Ich sah in den Rückspiegel und plötzlich traf mich eine Erkenntnis .', 'token': '▁Kurz ▁nachdem ▁T ipp er ▁und ▁ich ▁aus ▁dem ▁( ▁vor get äus cht es ▁Schl uch zen ▁) ▁We ißen ▁Haus ▁ausge zogen ▁waren ▁, ▁fu hren ▁wir ▁von ▁unserem ▁Haus ▁in ▁N ash v ille ▁zu ▁unserer ▁kleinen ▁F arm ▁50 ▁Me ilen ▁ öst lich ▁von ▁N ash v ille ▁- ▁und ▁wir ▁fu hren ▁selbst ▁. ▁( ▁Lachen ▁) ▁Ich ▁weiß ▁, ▁für ▁Sie ▁ist ▁das ▁nichts ▁Un gewöhn liches ▁, ▁aber ▁... ▁( ▁Lachen ▁) ▁Ich ▁sah ▁in ▁den ▁Rück sp iegel ▁und ▁plötzlich ▁traf ▁mich ▁eine ▁Erkennt nis ▁.', 'tokenid': '3581 6524 4109 1256 667 7576 5964 4560 5080 2718 7719 865 2606 514 704 3975 2178 2461 2719 4274 1383 3374 4561 2485 7745 2720 5607 963 7828 7718 7597 3374 6006 3750 316 2337 1163 7920 7599 6166 3187 298 2776 3670 1154 2714 2639 1470 7718 3750 316 2337 1163 2721 7576 7828 5607 963 7117 2722 2718 3593 2719 3419 7775 2720 5635 4021 6100 5048 6583 4190 868 1475 2720 4374 2724 2718 3593 2719 3419 7027 6006 5085 3957 2052 1059 7576 6782 7518 6435 5260 3164 1589 2722'}, {'name': 'target2', 'shape': [32, 7981], 'text': 'laughter we were driving from our home in nashville to a little farm we have 50 miles east of nashville driving ourselves laughter', 'token': '▁l aughter ▁we ▁were ▁driving ▁from ▁our ▁home ▁in ▁n ash v ille ▁to ▁a ▁little ▁farm ▁we ▁have ▁50 ▁miles ▁e ast ▁of ▁n ash v ille ▁driving ▁ourselves ▁l aughter', 'tokenid': '6220 390 7757 7792 5200 5600 6682 5918 6006 6521 316 2337 1163 7492 4372 6312 5500 7757 5851 2776 6441 5214 327 6639 6521 316 2337 1163 5200 6683 6220 390'}], 'utt2spk': 'sp0.9-ted_00001'})]
len(batch) = 1
-----
[('sp0.9-ted_00001_0077220_0085240', {'input': [{'feat': '/.../feats.1.ark:377509', 'name': 'input1', 'shape': [889, 83]}], 'lang': 'de', 'output': [{'name': 'target1', 'shape': [11, 7981], 'text': 'Eine wahre Geschichte - kein Wort daran ist erfunden .', 'token': '▁Eine ▁w ahre ▁Geschichte ▁- ▁kein ▁Wort ▁daran ▁ist ▁erfunden ▁.', 'tokenid': '3111 7729 149 3312 2721 6146 4325 5040 6100 5360 2722'}, {'name': 'target2', 'shape': [28, 7981], 'text': 'it &apos;s a true story every bit of this is true soon after tipper and i left the mock sob white house', 'token': '▁it ▁ &apos; s ▁a ▁true ▁story ▁every ▁bit ▁of ▁this ▁is ▁true ▁soon ▁after ▁t ipp er ▁and ▁i ▁left ▁the ▁mo ck ▁so b ▁white ▁house', 'tokenid': '6101 2714 6 1993 4372 7546 7331 5425 4739 6639 7474 6094 7546 7245 4418 7404 1256 667 4474 5959 6265 7458 6463 526 7208 416 7806 5931'}], 'utt2spk': 'sp0.9-ted_00001'}), 
('sp0.9-ted_00001_0135450_0139910', {'input': [{'feat': '/.../feats.1.ark:691378', 'name': 'input1', 'shape': [494, 83]}], 'lang': 'de', 'output': [{'name': 'target1', 'shape': [18, 7981], 'text': 'Es war Zeit zum Abendessen und wir hielten Ausschau nach einem Restaurant .', 'token': '▁Es ▁war ▁Zeit ▁zum ▁Aben dessen ▁und ▁wir ▁h ielten ▁Aus sch au ▁nach ▁einem ▁Rest aurant ▁.', 'tokenid': '3171 7744 4347 7926 2796 566 7576 7828 5818 1068 2895 1999 375 6523 5261 3941 398 2722'}, {'name': 'target2', 'shape': [15, 7981], 'text': 'it was dinnertime and we started looking for a place to eat', 'token': '▁it ▁was ▁d inn ert ime ▁and ▁we ▁started ▁looking ▁for ▁a ▁place ▁to ▁eat', 'tokenid': '6101 7749 5018 1229 696 1180 4474 7757 7302 6326 5570 4372 6767 7492 5224'}], 'utt2spk': 'sp0.9-ted_00001'})]
len(batch) = 2
-----
[('sp0.9-ted_00001_0072600_0076720', {'input': [{'feat': '/.../feats.1.ark:339027', 'name': 'input1', 'shape': [455, 83]}], 'lang': 'de', 'output': [{'name': 'target1', 'shape': [15, 7981], 'text': 'Ich erzähle Ihnen mal eine Geschichte , dann verstehen Sie mich vielleicht besser .', 'token': '▁Ich ▁erz ähle ▁Ihnen ▁mal ▁eine ▁Geschichte ▁, ▁dann ▁verstehen ▁Sie ▁mich ▁vielleicht ▁besser ▁.', 'tokenid': '3419 5400 2553 3428 6365 5260 3312 2720 5038 7672 4021 6435 7698 4690 2722'}, {'name': 'target2', 'shape': [22, 7981], 'text': 'i &apos;ll tell you one quick story to illustrate what that &apos;s been like for me', 'token': '▁i ▁ &apos; ll ▁tell ▁you ▁one ▁quick ▁story ▁to ▁ill ustr ate ▁what ▁that ▁ &apos; s ▁been ▁like ▁for ▁me', 'tokenid': '5959 2714 6 1493 7439 7900 6656 6889 7331 7492 5984 2310 335 7798 7457 2714 6 1993 4627 6300 5570 6390'}], 'utt2spk': 'sp0.9-ted_00001'}
), 
('sp0.9-ted_00001_0141970_0145180', {'input': [{'feat': '/.../feats.1.ark:738711', 'name': 'input1', 'shape': [355, 83]}], 'lang': 'de', 'output': [{'name': 'target1', 'shape': [18, 7981], 'text': 'Wir kamen zur Ausfahrt 238 , Lebanon , Tennessee .', 'token': '▁Wir ▁kamen ▁zur ▁Aus fahr t ▁23 8 ▁, ▁Le ban on ▁, ▁T enn esse e ▁.', 'tokenid': '4309 6139 7928 2895 752 2103 2761 29 2720 3603 420 1675 2720 4109 648 717 583 2722'}, {'name': 'target2', 'shape': [14, 7981], 'text': 'we got to exit 238 lebanon tennessee', 'token': '▁we ▁got ▁to ▁ex it ▁23 8 ▁le ban on ▁t enn esse e', 'tokenid': '7757 5768 7492 5434 1317 2761 29 6252 420 1675 7404 648 717 583'}], 'utt2spk': 'sp0.9-ted_00001'}
), 
('sp0.9-ted_00001_0117050_0118670', {'input': [{'feat': '/.../feats.1.ark:548927', 'name': 'input1', 'shape': [178, 83]}], 'lang': 'de', 'output': [{'name': 'target1', 'shape': [10, 7981], 'text': 'Hinter mir war gar keine Autokolonne .', 'token': '▁Hinter ▁mir ▁war ▁gar ▁keine ▁Aut ok ol onne ▁.', 'tokenid': '3398 6453 7744 5647 6147 2900 1635 1638 1685 2722'}, {'name': 'target2', 'shape': [8, 7981], 'text': 'there was no motorcade back there', 'token': '▁there ▁was ▁no ▁motor c ade ▁back ▁there', 'tokenid': '7465 7749 6592 6490 486 111 4586 7465'}], 'utt2spk': 'sp0.9-ted_00001'}
), 
('sp0.9-ted_00001_0140900_0141450', {'input': [{'feat': '/.../feats.1.ark:733097', 'name': 'input1', 'shape': [59, 83]}], 'lang': 'de', 'output': [{'name': 'target1', 'shape': [9, 7981], 'text': 'Wir waren auf der I-40 .', 'token': '▁Wir ▁waren ▁auf ▁der ▁I - 4 0 ▁.', 'tokenid': '4309 7745 4553 5094 3417 13 24 16 2722'}, {'name': 'target2', 'shape': [6, 7981], 'text': 'we were on i40', 'token': '▁we ▁were ▁on ▁i 4 0', 'tokenid': '7757 7792 6654 5959 24 16'}], 'utt2spk': 'sp0.9-ted_00001'}
)
]
len(batch) = 4

**********


**results of LoadInputsAndTargets(batch)**
features, targets = load_tr(train[0]) 

features: list of arrays, each array has dimension of (input_length x 83). 
len(features) = batch_size
targets: zipped object (list of arrays, each array is a tuple of 2 arrays.)

*features*: 
[[ 0.8094555   1.081493    0.83185744 ...  0.5747049   3.8700805
  -0.27677333]
 [ 0.82913566  1.0578123   0.9041011  ...  0.6309407   3.8235848
  -3.75317   ]
 [ 0.14192903  0.10036486  0.23958266 ...  0.73044664  3.6376014
  -4.9359536 ]
 ...
 [-0.09050822 -0.454628    0.11542428 ...  0.8458344  -2.2621756
  -0.34447995]
 [ 0.1823529   0.35616565  0.27798983 ...  0.90641296 -2.2621756
  -0.0172314 ]
 [ 0.21772379  0.20477334  0.01991782 ...  1.1298233  -2.1900644
  -0.11879128]]
len(f) = 2548

*targets: <zip object at 0x7f7374180c80>*
y1 = (array([3494, 6504, 5964, 6409, 3972, 2165,  923, 4560, 2477, 2720, 7569,
       7967, 4468, 2904, 1714, 7920, 6183, 2715, 2718, 2862, 2719]),)
y2 = (array([6606, 5959, 5851, 7492, 7407, 6640, 6509, 7144, 1620, 6672, 4754,
       1768, 7492, 5722, 6654, 4468, 4428, 1826,  235, 6220,  390, 4510,
        401]),)


*targets: <zip object at 0x7f7374180c80>*
y1 = (14, array([3494, 6504, 5964, 6409, 3972, 2165,  923, 4560, 2477, 2720, 7569,
       7967, 4468, 2904, 1714, 7920, 6183, 2715, 2718, 2862, 2719]),)
y2 = (5, array([6606, 5959, 5851, 7492, 7407, 6640, 6509, 7144, 1620, 6672, 4754,
       1768, 7492, 5722, 6654, 4468, 4428, 1826,  235, 6220,  390, 4510,
        401]),)


**results of CustomConverter**
converter = CustomConverter()
xs_pad, ilens, ys_pad, ys_pad_asr = converter([load_tr(smallset[0])])

*xs_pad*
tensor([[[ 0.8095,  1.0815,  0.8319,  ...,  0.5747,  3.8701, -0.2768],
         [ 0.8291,  1.0578,  0.9041,  ...,  0.6309,  3.8236, -3.7532],
         [ 0.1419,  0.1004,  0.2396,  ...,  0.7304,  3.6376, -4.9360],
         ...,
         [-0.0905, -0.4546,  0.1154,  ...,  0.8458, -2.2622, -0.3445],
         [ 0.1824,  0.3562,  0.2780,  ...,  0.9064, -2.2622, -0.0172],
         [ 0.2177,  0.2048,  0.0199,  ...,  1.1298, -2.1901, -0.1188]],

        [[-0.2730, -0.1446, -0.5941,  ...,  0.8124,  0.9636, -0.3175],
         [-0.2730,  0.0251, -0.1532,  ...,  1.0797,  0.9509, -0.1779],
         [ 0.4965,  0.1838, -0.3298,  ..., -0.1764,  0.9254, -0.5369],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]])
xs_pad.size() = torch.Size([2, 2548, 83])

*ilens*
tensor([2548, 1521])

*ys_pad*
tensor([[3494, 6504, 5964, 6409, 3972, 2165,  923, 4560, 2477, 2720, 7569, 7967,
         4468, 2904, 1714, 7920, 6183, 2715, 2718, 2862, 2719,   -1,   -1,   -1,
           -1,   -1,   -1,   -1,   -1],
        [3367, 4021, 7066, 6365, 7717, 3857,  255, 1658, 1999, 1535, 5675, 2791,
         2718, 3593, 2719, 4309, 7017, 2531, 6006, 5261, 5681, 1054, 2134, 3187,
         1714, 4109,  397, 2296, 2722]])
ys_pad.size() = torch.Size([2, 29])

*ys_pad_asr*
None

**results of ChainerDataLoader**
(tensor([[[-0.2730, -0.1446, -0.5941,  ...,  0.8124,  0.9636, -0.3175],
         [-0.2730,  0.0251, -0.1532,  ...,  1.0797,  0.9509, -0.1779],
         [ 0.4965,  0.1838, -0.3298,  ..., -0.1764,  0.9254, -0.5369],
         ...,
         [ 0.6286,  0.4465, -0.3298,  ..., -0.7679, -0.7022,  1.6736],
         [ 0.3049, -0.0680,  0.6068,  ..., -1.1375, -0.7022, -0.1978],
         [ 0.5531,  0.3151,  0.4704,  ..., -0.0655, -0.6442, -0.2776]]]), tensor([1521]), tensor([[3367, 4021, 7066, 6365, 7717, 3857,  255, 1658, 1999, 1535, 5675, 2791,
         2718, 3593, 2719, 4309, 7017, 2531, 6006, 5261, 5681, 1054, 2134, 3187,
         1714, 4109,  397, 2296, 2722]]), None)
-----
(tensor([[[-0.2373, -0.1912, -0.7128,  ...,  1.3555, -0.6151, -0.0723],
         [-0.5975, -0.7388, -0.5606,  ...,  1.3659, -0.6151,  0.0118],
         [-0.7976, -0.5432, -0.4085,  ...,  1.2518, -0.6032,  0.1970],
         ...,
         [ 0.5753,  0.8414,  0.6048,  ...,  0.2501, -1.4204, -1.0258],
         [ 0.8240,  0.7493,  0.3914,  ...,  0.0990, -1.4204, -0.4174],
         [ 0.7802,  0.8184, -0.7128,  ...,  0.4044, -1.4204, -0.0050]]]), tensor([1519]), tensor([[3051, 6409, 5964, 5383, 2720, 7434, 2393, 5100, 2721, 7768, 5964, 5407,
         7832, 4777, 6140, 2715, 2718, 3593, 2719, 4226,  741, 4021, 7163, 6365,
         6006, 6409, 3594, 2715, 2718, 3593, 2719, 2718, 2862, 2719, 3419, 4727,
         4727, 4397, 3479, 6229, 6459, 5094, 2792, 1262, 3224,  489, 4179, 5651,
          778, 1625, 2722]]), None)
-----
(tensor([[[ 0.2312,  0.6313,  0.3679,  ...,  0.6264,  1.7241, -0.9485],
         [ 0.0150,  0.3550,  0.6938,  ...,  0.6420,  1.5775, -1.9284],
         [ 1.0021,  0.7695,  0.6356,  ...,  0.5484,  1.2110, -2.0509],
         ...,
         [-0.3816, -0.4621,  0.1700,  ...,  1.0163, -0.7788, -1.3772],
         [-0.4405, -0.7115,  0.3213,  ...,  1.0319, -0.9596, -1.3160],
         [-0.3228, -0.1059,  0.5890,  ...,  0.9696, -1.3212, -1.0710]]]), tensor([1159]), tensor([[3581, 6524, 4109, 1256,  667, 7576, 5964, 4560, 5080, 2718, 7719,  865,
         2606,  514,  704, 3975, 2178, 2461, 2719, 4274, 1383, 3374, 4561, 2485,
         7745, 2720, 5607,  963, 7828, 7718, 7597, 3374, 6006, 3750,  316, 2337,
         1163, 7920, 7599, 6166, 3187,  298, 2776, 3670, 1154, 2714, 2639, 1470,
         7718, 3750,  316, 2337, 1163, 2721, 7576, 7828, 5607,  963, 7117, 2722,
         2718, 3593, 2719, 3419, 7775, 2720, 5635, 4021, 6100, 5048, 6583, 4190,
          868, 1475, 2720, 4374, 2724, 2718, 3593, 2719, 3419, 7027, 6006, 5085,
         3957, 2052, 1059, 7576, 6782, 7518, 6435, 5260, 3164, 1589, 2722]]), None)
-----
(tensor([[[-0.0416, -0.1237, -0.1337,  ...,  0.8697, -1.5003,  0.0063],
         [-0.4124, -0.3244, -0.4409,  ...,  1.1338, -1.5795,  0.7815],
         [-0.5775, -0.1962, -0.2509,  ...,  1.1128, -1.5003,  1.0509],
         ...,
         [ 0.1778,  0.4923,  0.4251,  ...,  0.6886, -4.7459, -1.1387],
         [ 0.2327,  0.5141,  0.7766,  ...,  0.4650, -4.7459, -1.8152],
         [ 0.3424,  0.6880,  0.3530,  ...,  0.5502, -5.3792, -1.7186]],

        [[-0.1617, -0.3260, -0.8413,  ...,  1.1270, -0.2880, -0.2004],
         [ 0.1719,  0.2432, -0.1631,  ...,  0.9445, -0.2776,  0.3082],
         [-0.2858, -0.4733, -1.3525,  ...,  1.2881, -0.2466, -0.0771],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]]), tensor([889, 494]), tensor([[3111, 7729,  149, 3312, 2721, 6146, 4325, 5040, 6100, 5360, 2722,   -1,
           -1,   -1,   -1,   -1,   -1,   -1],
        [3171, 7744, 4347, 7926, 2796,  566, 7576, 7828, 5818, 1068, 2895, 1999,
          375, 6523, 5261, 3941,  398, 2722]]), None)
-----
(tensor([[[ 0.0394,  0.1355,  0.1860,  ...,  0.7351, -1.0426,  1.1069],
         [ 0.1084, -0.0819,  0.6175,  ...,  1.0098, -0.6192,  0.7155],
         [ 0.4193,  0.6264,  1.0170,  ...,  1.0448, -0.7402,  0.5477],
         ...,
         [ 0.9535, -0.1660,  0.5504,  ...,  1.1729, -1.1031,  2.0576],
         [ 0.9047,  1.0809,  0.9531,  ...,  0.9632, -0.8208,  0.8832],
         [ 0.1913,  0.2056,  0.6846,  ...,  0.7228, -0.6999,  0.8553]],

        [[ 1.1559,  1.4428,  1.2066,  ..., -0.2648, -0.3481, -0.3544],
         [ 1.0371,  1.3831,  1.0387,  ..., -0.5977, -0.4609,  0.2370],
         [ 1.3541,  1.5764,  1.1934,  ..., -0.5859, -0.4609,  0.6708],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

        [[ 0.7558,  1.0690,  1.3729,  ..., -0.0059, -0.1404, -0.2140],
         [ 0.7950,  1.1457,  1.3854,  ...,  0.3027, -0.2301, -0.2918],
         [ 0.7602,  0.8801,  1.2379,  ...,  0.8854, -0.2189, -0.8525],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

        [[ 1.1641,  1.4775,  1.2410,  ..., -0.1812, -0.3484, -0.1921],
         [ 0.8714,  1.1607,  1.1715,  ..., -0.0823, -0.4353, -0.6391],
         [ 0.7500,  1.1197,  1.0381,  ...,  0.0364, -0.4932, -0.4380],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]]), tensor([455, 355, 178,  59]), tensor([[3419, 5400, 2553, 3428, 6365, 5260, 3312, 2720, 5038, 7672, 4021, 6435,
         7698, 4690, 2722,   -1,   -1,   -1],
        [4309, 6139, 7928, 2895,  752, 2103, 2761,   29, 2720, 3603,  420, 1675,
         2720, 4109,  648,  717,  583, 2722],
        [3398, 6453, 7744, 5647, 6147, 2900, 1635, 1638, 1685, 2722,   -1,   -1,
           -1,   -1,   -1,   -1,   -1,   -1],
        [4309, 7745, 4553, 5094, 3417,   13,   24,   16, 2722,   -1,   -1,   -1,
           -1,   -1,   -1,   -1,   -1,   -1]]), None)
-----
(tensor([[[ 0.8095,  1.0815,  0.8319,  ...,  0.5747,  3.8701, -0.2768],
         [ 0.8291,  1.0578,  0.9041,  ...,  0.6309,  3.8236, -3.7532],
         [ 0.1419,  0.1004,  0.2396,  ...,  0.7304,  3.6376, -4.9360],
         ...,
         [-0.0905, -0.4546,  0.1154,  ...,  0.8458, -2.2622, -0.3445],
         [ 0.1824,  0.3562,  0.2780,  ...,  0.9064, -2.2622, -0.0172],
         [ 0.2177,  0.2048,  0.0199,  ...,  1.1298, -2.1901, -0.1188]]]), tensor([2548]), tensor([[3494, 6504, 5964, 6409, 3972, 2165,  923, 4560, 2477, 2720, 7569, 7967,
         4468, 2904, 1714, 7920, 6183, 2715, 2718, 2862, 2719]]), None)

pad = 0
src_lang = 1
tgt_langs = i + 2 for i, _ in enumerate(tgt_langs)


## Di Gangi's implementation FBK-fairseq
task = tasks.setup_task(args)
dicts: OrderedDict
dicts['langs'], dicts['tgt_dict'] (created from dictionaries of all target langs), dicts['target'][lang] 

self.datasets[split]: RoundRobinZipDatasets

**Model settings** The first two CNNs in the encoder have 16 output channels, 3 × 3 kernel and stride (2, 2). 
The CNNs inside the 2D self-attention have 3 × 3 kernels, 4 output channels and stride 1. 

The output CNN of the 2D self-attention has 16 output channels. The following feed-forward layer has 512 output features, which is the same size as the Transformer layers. The hidden feed-forward layer size of Transformer is 1024. The decoder layers have also size 512 and hidden size 1024. Dropout is set to 0.1 in each layer. Each minibatch includes up to 8 sentences for each language and we update the gradient every 16 iterations. All the models are trained with the Adam [44] optimizer with an initial learning rate of 0.0003, then 4000 warmup steps during which it increases linearly up to a max value, and then decreases with the in- verse square root of the number of steps [6]. 

```python
encoder = TransformerEncoder(args,
                tgt_dict,
                audio_features=task.audio_features,
                language_embeddings=lang_embed_tokens,
                token_position=args.token_position
            )
decoder = TransformerDecoder(args,
        tgt_dict,
        decoder_embed_tokens,
        language_embeddings=lang_embed_tokens,
        token_position=args.token_position
    )
        
return TransformerModel(encoder, decoder, task.args.lang_pairs)
```

## Trainer

```json
>>> snapshot_dict['trainer']
{
    'updater/iterator:main/epoch': array(10), 
    'updater/iterator:main/current_position': array(17360), 
    'updater/iteration': array(230000), 
    'stop_trigger/previous_iteration': array(229999), 
    'stop_trigger/previous_epoch_detail': array(10.39216971), 
    'extension_triggers/validation/previous_iteration': array(230000), 
    'extension_triggers/validation/previous_epoch_detail': array(10.3921923), 
    'extension_triggers/PlotAttentionReport/previous_iteration': array(230000), 
    'extension_triggers/PlotAttentionReport/previous_epoch_detail': array(10.3921923), 
    'extensions/PlotReport/_plot_loss.png': array(
                                                '{"main/loss": [[8, 85.08099365234375], [9, 114.74853742066776], [10, 112.05137844629061]], 
                                                "validation/main/loss": [[8, 103.4844144748736], [9, 98.80690422540978], [10, 98.2255947402761]], 
                                                "main/loss_asr": [[8, 0.0], [9, 0.0], [10, 0.0]], 
                                                "validation/main/loss_asr": [[8, 0.0], [9, 0.0], [10, 0.0]], 
                                                "main/loss_st": [[8, 85.08099365234375], [9, 114.74853742066776], [10, 112.05137844629061]], 
                                                "validation/main/loss_st": [[8, 103.4844144748736], [9, 98.80690422540978], [10, 98.2255947402761]]}',
                                                dtype='<U493'), 
    'extension_triggers/PlotReport/previous_iteration': array(230000), 
    'extension_triggers/PlotReport/previous_epoch_detail': array(10.3921923), 
    'extensions/PlotReport_1/_plot_acc.png': array(
                                                '{"main/acc": [[8, 0.5961538461538461], [9, 0.530444692661169], [10, 0.5396039237575011]], 
                                                "validation/main/acc": [[8, 0.5595176170187053], [9, 0.5705362317847726], [10, 0.574461554421936]], 
                                                "main/acc_asr": [[8, 0.0], [9, 0.0], [10, 0.0]], 
                                                "validation/main/acc_asr": [[8, 0.0], [9, 0.0], [10, 0.0]]}',
                                                dtype='<U298'), 
    'extension_triggers/PlotReport_1/previous_iteration': array(230000), 
    'extension_triggers/PlotReport_1/previous_epoch_detail': array(10.3921923), 
    'extensions/PlotReport_2/_plot_bleu.png': array(
                                                '{"main/bleu": [[8, 0.0], [9, 0.0], [10, 0.0]], 
                                                "validation/main/bleu": [[8, 0.0], [9, 0.0], [10, 0.0]]}',
                                                dtype='<U103'), 
    'extension_triggers/PlotReport_2/previous_iteration': array(230000), 
    'extension_triggers/PlotReport_2/previous_epoch_detail': array(10.3921923), 
    'extension_triggers/snapshot_object/interval_trigger/previous_iteration': array(230000), 
    'extension_triggers/snapshot_object/interval_trigger/previous_epoch_detail': array(10.3921923), 
    'extension_triggers/snapshot_object/summary/_names': array('["validation/main/loss"]', dtype='<U24'), 
    'extension_triggers/snapshot_object/summary/_summaries/0/_x': array(193.82722164), 
    'extension_triggers/snapshot_object/summary/_summaries/0/_x2': array(18795.37402394), 
    'extension_triggers/snapshot_object/summary/_summaries/0/_n': array(2), 
    'extension_triggers/snapshot_object/best_value': array(98.22559474), 
    'extension_triggers/snapshot_object_1/interval_trigger/previous_iteration': array(230000), 
    'extension_triggers/snapshot_object_1/interval_trigger/previous_epoch_detail': array(10.3921923), 
    'extension_triggers/snapshot_object_1/summary/_names': array('["validation/main/acc"]', dtype='<U23'), 
    'extension_triggers/snapshot_object_1/summary/_summaries/0/_x': array(1.15929077), 
    'extension_triggers/snapshot_object_1/summary/_summaries/0/_x2': array(0.6720499), 
    'extension_triggers/snapshot_object_1/summary/_summaries/0/_n': array(2), 
    'extension_triggers/snapshot_object_1/best_value': array(0.57446155), 
    'extension_triggers/torch_snapshot/previous_iteration': array(230000), 
    'extension_triggers/torch_snapshot/previous_epoch_detail': array(10.3921923), 
    'extensions/LogReport/_trigger/previous_iteration': array(230000), 
    'extensions/LogReport/_trigger/previous_epoch_detail': array(10.3921923), 
    'extensions/LogReport/_summary/_names': array('[]', dtype='<U2'), 
    'extensions/LogReport/_log': array('
                                    [{
                                    "main/loss_asr": 0.0, 
                                    "main/loss_st": 85.08099365234375, 
                                    "main/acc_asr": 0.0, 
                                    "main/acc_mt": 0.0, 
                                    "main/acc": 0.5961538461538461, 
                                    "main/bleu": 0.0, 
                                    "main/loss": 85.08099365234375, 
                                    "validation/main/loss_asr": 0.0, 
                                    "validation/main/loss_st": 103.4844144748736, 
                                    "validation/main/acc_asr": 0.0, 
                                    "validation/main/acc_mt": 0.0, 
                                    "validation/main/acc": 0.5595176170187053, 
                                    "validation/main/bleu": 0.0, 
                                    "validation/main/loss": 103.4844144748736, 
                                    "lr": 0.00036327387107443524, 
                                    "epoch": 8, 
                                    "iteration": 185000, 
                                    "elapsed_time": 192200.72107164213
                                    }, 
                                    {
                                    "main/loss_asr": 0.0, 
                                    "main/loss_st": 115.32821753120422, 
                                    "main/acc_asr": 0.0, "main/acc_mt": 0.0, 
                                    "main/acc": 0.5274218032109924, 
                                    "main/bleu": 0.0, 
                                    "main/loss": 115.32821753120422, 
                                    "validation/main/loss_asr": 0.0, 
                                    "validation/main/loss_st": 98.15568211712414, 
                                    "validation/main/acc_asr": 0.0, 
                                    "validation/main/acc_mt": 0.0, 
                                    "validation/main/acc": 0.5697980634362989, 
                                    "validation/main/bleu": 0.0, 
                                    "validation/main/loss": 98.15568211712414, 
                                    "lr": 0.00035846208417275274, 
                                    "epoch": 8, 
                                    "iteration": 190000, 
                                    "elapsed_time": 197361.624626596
                                    }, 
                                    {
                                    "main/loss_asr": 0.0, 
                                    "main/loss_st": 114.71346376880408, 
                                    "main/acc_asr": 0.0, 
                                    "main/acc_mt": 0.0, 
                                    "main/acc": 0.5316505038717668, 
                                    "main/bleu": 0.0, 
                                    "main/loss": 114.71346376880408, 
                                    "validation/main/loss_asr": 0.0, 
                                    "validation/main/loss_st": 99.45812633369542, 
                                    "validation/main/acc_asr": 0.0, 
                                    "validation/main/acc_mt": 0.0, 
                                    "validation/main/acc": 0.5712744001332464, 
                                    "validation/main/bleu": 0.0, 
                                    "validation/main/loss": 99.45812633369542, 
                                    "lr": 0.0003538365731701862, 
                                    "epoch": 8, 
                                    "iteration": 195000, 
                                    "elapsed_time": 202520.33393891202
                                    }, 
                                    {"main/loss_asr": 0.0, "main/loss_st": 113.76040377669334, "main/acc_asr": 0.0, "main/acc_mt": 0.0, "main/acc": 0.5328818975880396, "main/bleu": 0.0, "main/loss": 113.76040377669334, "validation/main/loss_asr": 0.0, "validation/main/loss_st": 99.88844615598268, "validation/main/acc_asr": 0.0, "validation/main/acc_mt": 0.0, "validation/main/acc": 0.5640924193965503, "validation/main/bleu": 0.0, "validation/main/loss": 99.88844615598268, "lr": 0.0003493856214843422, "epoch": 9, "iteration": 200000, "elapsed_time": 207858.78996863612}, {"main/loss_asr": 0.0, "main/loss_st": 113.57487056673766, "main/acc_asr": 0.0, "main/acc_mt": 0.0, "main/acc": 0.5352681239642069, "main/bleu": 0.0, "main/loss": 113.57487056673766, "validation/main/loss_asr": 0.0, "validation/main/loss_st": 98.53392770018759, "validation/main/acc_asr": 0.0, "validation/main/acc_mt": 0.0, "validation/main/acc": 0.574487953123729, "validation/main/bleu": 0.0, "validation/main/loss": 98.53392770018759, "lr": 0.0003450985189838954, "epoch": 9, "iteration": 205000, "elapsed_time": 213043.07749263314}, {"main/loss_asr": 0.0, "main/loss_st": 112.77486765979529, "main/acc_asr": 0.0, "main/acc_mt": 0.0, "main/acc": 0.5381508377044603, "main/bleu": 0.0, "main/loss": 112.77486765979529, "validation/main/loss_asr": 0.0, "validation/main/loss_st": 98.4972378935995, "validation/main/acc_asr": 0.0, "validation/main/acc_mt": 0.0, "validation/main/acc": 0.5738486378307337, "validation/main/bleu": 0.0, "validation/main/loss": 98.4972378935995, "lr": 0.00034096545349373804, "epoch": 9, "iteration": 210000, "elapsed_time": 218273.30154809705}, {"main/loss_asr": 0.0, "main/loss_st": 110.66975374498368, "main/acc_asr": 0.0, "main/acc_mt": 0.0, "main/acc": 0.5412481981878513, "main/bleu": 0.0, "main/loss": 110.66975374498368, "validation/main/loss_asr": 0.0, "validation/main/loss_st": 97.12829271147523, "validation/main/acc_asr": 0.0, "validation/main/acc_mt": 0.0, "validation/main/acc": 0.5767581880303293, "validation/main/bleu": 0.0, "validation/main/loss": 97.12829271147523, "lr": 0.000336977416260745, "epoch": 9, "iteration": 215000, "elapsed_time": 223410.44185818406}, {"main/loss_asr": 0.0, "main/loss_st": 111.10800393486024, "main/acc_asr": 0.0, "main/acc_mt": 0.0, "main/acc": 0.5435022756949396, "main/bleu": 0.0, "main/loss": 111.10800393486024, "validation/main/loss_asr": 0.0, "validation/main/loss_st": 97.08006924013549, "validation/main/acc_asr": 0.0, "validation/main/acc_mt": 0.0, "validation/main/acc": 0.583120573728338, "validation/main/bleu": 0.0, "validation/main/loss": 97.08006924013549, "lr": 0.0003331261193056413, "epoch": 9, "iteration": 220000, "elapsed_time": 228619.56588784396}, {"main/loss_asr": 0.0, "main/loss_st": 110.26709885007143, "main/acc_asr": 0.0, "main/acc_mt": 0.0, "main/acc": 0.5458925646485462, "main/bleu": 0.0, "main/loss": 110.26709885007143, "validation/main/loss_asr": 0.0, "validation/main/loss_st": 94.58143379114851, "validation/main/acc_asr": 0.0, "validation/main/acc_mt": 0.0, "validation/main/acc": 0.5856600542810775, "validation/main/bleu": 0.0, "validation/main/loss": 94.58143379114851, "lr": 0.0003294039229342062, "epoch": 10, "iteration": 225000, "elapsed_time": 233966.75853144098}, {"main/loss_asr": 0.0, "main/loss_st": 110.2969423913002, "main/acc_asr": 0.0, "main/acc_mt": 0.0, "main/acc": 0.5462913072473132, "main/bleu": 0.0, "main/loss": 110.2969423913002, "validation/main/loss_asr": 0.0, "validation/main/loss_st": 99.24578784990915, "validation/main/acc_asr": 0.0, "validation/main/acc_mt": 0.0, "validation/main/acc": 0.5736307148618429, "validation/main/bleu": 0.0, "validation/main/loss": 99.24578784990915, "lr": 0.00032580377196417926, "epoch": 10, "iteration": 230000, "elapsed_time": 239245.96051169}]',
                                    dtype='<U5382'), 
    'extension_triggers/LogReport/previous_iteration': array(230000), 
    'extension_triggers/LogReport/previous_epoch_detail': array(10.3921923), 
    'extension_triggers/_observe_value/previous_iteration': array(230000), 
    'extension_triggers/_observe_value/previous_epoch_detail': array(10.3921923), 
    'extension_triggers/PrintReport/previous_iteration': array(230000), 
    'extension_triggers/PrintReport/previous_epoch_detail': array(10.3921923), 
    'extension_triggers/ProgressBar/previous_iteration': array(230000), 
    'extension_triggers/ProgressBar/previous_epoch_detail': array(10.3921923), 
    'extension_triggers/espnet_tensorboard_logger/previous_iteration': array(230000), 
    'extension_triggers/espnet_tensorboard_logger/previous_epoch_detail': array(10.3921923), 
    '_snapshot_elapsed_time': array(239246.10471111)}
```

**NOTE** Incorrect implementation of positional encoding in ESPNet e2e_st_transformer: input embed and positional embed are followed sequentially, not added together.

**Pretrained ASR models**
FILE_ID=1alt627cvDUhwGufPOC_-AfUVYkMEz0Ex (es)
FILE_ID=1aKRJOLW2TrAPB3UEW7RYFoFDGEm9dM4K (de)
FILE_ID=1nyMlgPEnIki-EN_QEsRFhAbITkUPzGzz (pt)
FILE_ID=1RhD9zgYx5AZkCRvsKb3WvBamkNnXc5pz (fr)
FILE_ID=1t1tDx62l_ozR2Jw8OAQGYtG5U7ico9yO (ro)
FILE_ID=1-XoSitmePZ127cxB_xryV88xQMgdIMf6 (ru)
FILE_ID=17NAo5CIwijjaTIcy_wbOWqcYq9JkIQD- (nl)
FILE_ID=1pXZk-1f8OTzjXZSJpuYIQ3VD_PBzImxo (it)

**Pretrained MT models**

FILE_ID=1jnS8aZh-FoKBy1qjX9tJ0wF8qK3weVBY (it)
FILE_ID=1K881dOzy13UDOr_VzteUm6zETmR_fkgw (nl)
FILE_ID=1ZNkmLVR6wlWTU9fmWZc5cLkdlF8LUHwO (ru)
FILE_ID=1x2k-N7DKXYi1WN9uB3qTlwIGxIiuZJNt (ro)
FILE_ID=1d9iqY-R0E6DzU1Af9KZQI3gzIuLGLfal (es)
FILE_ID=1lBnAbZCSR-y2gz1aWEdt_LxJm7KvBfZJ (fr)
FILE_ID=15hpGUyQTLKBLUxcdXdxnxD91f3X1bHWV (pt)
FILE_ID=1qQRu5m99PGR6XW5COgqdYAfYwbSGy39k (de)
gdown --id ${FILE_ID}

**TRAINING: 2 langs**
decoder_pre_multi_asr_lang_de_nl: iter 158k, 23m/1k iter => timelimit: 2.88h
encoder_pre_sum_multi_asr_lang_de_nl: iter 158k, 23m-24m/1k iter => timelimit: 3.0h
decoder_pre_1decoder_lang_de_nl: iter 163k, 20m-21m/1k iter => 165k: timelimit: 0.9h

xself-xsum-waitk0
xsource-xsum-waitk0
xsource-xconcat-x2st-waitk3

decoder_pre_xself_xsum_waitk0_lang_de_nl: 1780572
decoder_pre_xsource_xsum_waitk0_lang_de_nl: 1780632
decoder_pre_xsource_xconcat_x2st_waitk3_lang_de_nl: 1780635
decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl: 1781411 

**Pretraining**
pt_decoder_pre_1decoder_lang_de_nl

**TRAINING: 8 langs**
decoder_pre_1decoder: 1775633

**DECODING**
```
decoder_pre_multi_asr_lang_de_nl common_decode 40 (BLEU 17.42)
encoder_pre_sum_multi_asr_lang_de_nl common_decode 40 (BLEU 14.87)
decoder_pre_1decoder_lang_de_nl common_decode 77 (BLEU 20.53)

decoder_pre_multi_asr_lang_de_nl separate_decode 40 (en-de BLEU 17.75)
encoder_pre_sum_multi_asr_lang_de_nl separate_decode 40 (BLEU 14.97)
decoder_pre_1decoder_lang_de_nl separate_decode 77 (BLEU 20.81)

decoder_pre_xsource_xconcat_lang_de_nl common_decode 57 (BLEU 14.52)
decoder_pre_xself_xsum_lang_de_nl common_decode 66 (BLEU 13.99)
decoder_pre_xsource_xsum_lang_de_nl common_decode 61 (BLEU 13.74)
decoder_pre_xself_xconcat_lang_de_nl common_decode 61 (BLEU 14.33)
decoder_pre_xself_xsource_xconcat_lang_de_nl common_decode 52 (BLEU 13.80)
----- 
decoder_pre_multi_asr_lang_de_nl separate_decode 131 - beam 10: BLEU 22.56 - beam 1: BLEU 20.82
encoder_pre_sum_multi_asr_lang_de_nl separate_decode 136 - beam 10: BLEU 21.3 - beam 1: BLEU 19.68
decoder_pre_1decoder_lang_de_nl separate_decode 135 - beam 10: BLEU 22.09 - beam 1: BLEU 20.36

decoder_pre_xsource_xconcat_lang_de_nl common_decode 104 - beam 10: BLEU 15.93
decoder_pre_xself_xsum_lang_de_nl common_decode 105 - beam 10: BLEU 15.79
decoder_pre_xsource_xsum_lang_de_nl common_decode 108 - beam 10: BLEU 16.31 - beam 1: 12.90
decoder_pre_xself_xconcat_lang_de_nl common_decode 103 - beam 10: BLEU 16.0
decoder_pre_xself_xsource_xconcat_lang_de_nl common_decode 95 - beam 10: BLEU 15.88

decoder_pre_xsource_xconcat_lang_de_nl common_decode_weighting 104 - beam 10: BLEU 15.51
decoder_pre_xself_xsum_lang_de_nl common_decode_weighting 105 - beam 10: BLEU 15.09
decoder_pre_xsource_xsum_lang_de_nl common_decode_weighting 108 - beam 10: BLEU 15.84
decoder_pre_xself_xconcat_lang_de_nl common_decode_weighting 103 - beam 10: BLEU 15.13
decoder_pre_xself_xsource_xconcat_lang_de_nl common_decode_weighting 95 - beam 10: BLEU 15.57
```

##### FINAL RESULTS FOR 1 DECODER and 2 INDEPENDENT DECODERS
##### Use best validation model
sbatch submit.slurm decoder_pre_multi_asr_lang_de_nl separate_decode 0 tst-COMMON.en-de.de - beam 10: BLEU de 22.56 
sbatch submit.slurm decoder_pre_multi_asr_lang_de_nl separate_decode 0 tst-COMMON.en-nl.nl - beam 10: BLEU nl 26.51
sbatch submit.slurm decoder_pre_multi_asr_lang_de_nl separate_decode 0 tst-HE.en-de.de 1783374
sbatch submit.slurm decoder_pre_multi_asr_lang_de_nl separate_decode 0 tst-HE.en-nl.nl 1783375

sbatch submit.slurm encoder_pre_sum_multi_asr_lang_de_nl separate_decode 0 tst-COMMON.en-de.de - beam 10: BLEU de 21.28 
sbatch submit.slurm encoder_pre_sum_multi_asr_lang_de_nl separate_decode 0 tst-COMMON.en-nl.nl - beam 10 BLEU de 1.45 1783671
sbatch submit.slurm encoder_pre_sum_multi_asr_lang_de_nl separate_decode 0 tst-HE.en-de.de 1783376
sbatch submit.slurm encoder_pre_sum_multi_asr_lang_de_nl separate_decode 0 tst-HE.en-nl.nl 1783377

sbatch submit.slurm decoder_pre_1decoder_lang_de_nl separate_decode 0 tst-COMMON.en-de.de - beam 10: BLEU de 22.50
sbatch submit.slurm decoder_pre_1decoder_lang_de_nl separate_decode 0 tst-COMMON.en-nl.nl 1783282 - beam 10 BLEU nl 26.83
sbatch submit.slurm decoder_pre_1decoder_lang_de_nl separate_decode 0 tst-HE.en-de.de 1783378
sbatch submit.slurm decoder_pre_1decoder_lang_de_nl separate_decode 0 tst-HE.en-nl.nl 1783379

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate 8 tst-COMMON.en-de.de - beam 5 x beam_cross 3: BLEU de 20.78 
sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate 8 tst-COMMON.en-nl.nl 1783290
sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate 8 tst-HE.en-de.de 1783380
sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate 8 tst-HE.en-nl.nl 1783381

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate_b10_bx5 8 tst-COMMON.en-de.de - beam 10 x beam_cross 5: BLEU de 20.87
sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate_b10_bx5 8 tst-COMMON.en-nl.nl 1783292
sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate_b10_bx5 8 tst-HE.en-de.de 1783382
sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate_b10_bx5 8 tst-HE.en-nl.nl 1783383

sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate 17 tst-COMMON.en-de.de 1783394
sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate 17 tst-COMMON.en-nl.nl 1783395
sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate 17 tst-HE.en-de.de 1783396
sbatch submit.slurm decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl common_decode_separate 17 tst-HE.en-nl.nl 1783397

sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-COMMON.en-de.de 1783413
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-COMMON.en-es.es 1783414
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-COMMON.en-fr.fr 1783415
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-COMMON.en-it.it 1783416
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-COMMON.en-nl.nl 1783417
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-COMMON.en-pt.pt 1783418
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-COMMON.en-ro.ro 1783419
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-COMMON.en-ru.ru 1783420

sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-HE.en-de.de 1783421
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-HE.en-es.es 1783422
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-HE.en-fr.fr 1783423
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-HE.en-it.it 1783424
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-HE.en-nl.nl 1783425
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-HE.en-pt.pt 1783426
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-HE.en-ro.ro 1783427
sbatch submit.slurm decoder_pre_1decoder common_decode_separate 135 tst-HE.en-ru.ru 1783428

**debug**
tag=debug_decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl
decode=common_decode_separate
stage=5
ngpu=0
decode_config=./conf/tuning/${decode}
trans_model=model.acc.best
trans_set=tst-COMMON.en-de.de
bash run-cmd.sh $tag $stage $ngpu $decode_config $trans_model $trans_set


tag=debug_decoder_pre_xsource_xconcat_x2st_waitk3_pretrain_lang_de_nl
decode=common_decode
stage=4
ngpu=1
decode_config=./conf/tuning/${decode}
trans_model=model.acc.best
bash run-cmd.sh $tag $stage $ngpu $decode_config $trans_model