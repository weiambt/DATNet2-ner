# DATNet

这是这篇文章的：

“[用于低资源命名实体识别的双重对抗性神经迁移](https://www.aclweb.org/anthology/P19-1336/)”

dversarial Neural Transfer for Low-Resource Named Entity Recognition

---

## Requirements

- tensorflow==1.8.0 (>=1.8.0, <=1.13.1)
- tensorflow_hub==0.3.0
- numpy==1.14.6
- ujson==1.35
- emoji==0.5.1
- matplotlib=3.0.3
- tqdm==4.31.1

- python 3.x with package tensorflow-gpu (`1.13.1`), tensorflow_hub, ujson, emoji, matplotlib, tqdm, seqeval, scikit-learn

## Usage

You can download pre-processed datasets from [Box Drive (datasets)](https://app.box.com/s/3rgap12lnwr7mamkaks1lql12d46lat1), 
and save them to `./datasets/` folder.

Run baseline model on CoNLL-2003 English NER task using adversarial training (`--at true`). 
More parameters setting in [run_baseline.py](/run_baseline.py).
```shell script
python run_baseline.py --train true --elmo false --task conll03_en_ner --at true
```

Run DATNet-P model on CoNLL-2003 English NER (OntoNotes NER as source) using ELMo (`--elmo true`), adversarial training 
(`--at true`), share word embeddings (`--share_word true`). More parameters setting in [run_datnetp.py](/run_datnetp.py).
```shell script
python run_datnetp.py --src_task ontonotes_ner --tgt_task conll03_en_ner --elmo true --at ture \
                      --share_word true
```

Similarly, run DATNet-F model on CoNLL-2003 English NER (OntoNotes NER as source) using ELMo (`--elmo true`), adversarial 
training (`--at true`), share word embeddings (`--share_word true`). More parameters setting in 
[run_datnetf.py](/run_datnetf.py).
```shell script
python run_datnetf.py --src_task ontonotes_ner --tgt_task conll03_en_ner --elmo true --at ture \
                      --share_word true
```

**Note**: to obtain the main results of Table 2 in "Dual Adversarial Neural Transfer for Low-Resource Named Entity 
Recognition", you can download the DATNet codes (init version) and trained weights, which are available on 
[Box Drive (DATNet)](https://app.box.com/s/d7nuslxqccgtbct06vrzvtpbz7a3rtw8), and following the provided 
[instructions](https://app.box.com/s/toa8ncdp2hyk81qfeyqbsule2864zkky) to do evaluations. The pre-trained word 
embeddings are available [here](http://www.limteng.com/research/2018/05/14/pretrained-word-embeddings.html).

## Citation
If you feel this project helpful to your research, please cite our work.
```
@inproceedings{zhou2019dual,
    title = {Dual Adversarial Neural Transfer for Low-Resource Named Entity Recognition},
    author = {Zhou, Joey Tianyi and Zhang, Hao and Jin, Di and Zhu, Hongyuan and Fang, Meng and Goh, Rick Siow Mong and Kwok, Kenneth},
    booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year = {2019},
    address = {Florence, Italy},
    publisher = {Association for Computational Linguistics},
    url = {https://www.aclweb.org/anthology/P19-1336},
    doi = {10.18653/v1/P19-1336},
    pages = {3461--3471}
}
```
and
```
@article{8778733,
  author={J. T. {Zhou} and H. {Zhang} and D. {Jin} and X. {Peng}},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Dual Adversarial Transfer for Sequence Labeling}, 
  year={2021},
  volume={43},
  number={2},
  pages={434-446},
  doi={10.1109/TPAMI.2019.2931569}
}
```


## Usage

> All the following results is derived by transferring CoNLL-2003 English NER to the target datasets.

The pre-trained word embeddings are available [here](http://www.limteng.com/research/2018/05/14/pretrained-word-embeddings.html).

The best result on **WNUT-2016 English Twitter NER** is `F1=54.88`, restore and test the model by running: 

```bash
$ python3 run_datnetf.py --src_language english --tgt_language wnut2016 --norm_emb true --at true \
                         --share_word true --concat_rnn false --word_project true --train false
```

The best result on **WNUT-2017 English Twitter NER** is `F1=43.63`, restore and test the model by running: 

```bash
$ python3 run_datnetf.py --src_language english --tgt_language wnut2017 --norm_emb true --at true \
                         --share_word true --concat_rnn false --word_project true --train false
```

The best result on **CoNLL-2002 Dutch NER** is `F1=88.32`, restore and test the model by running: 

```bash
$ python3 run_datnetp.py --src_language english --tgt_language dutch --at true \
                         --norm_emb true --alpha 0.5 --concat_rnn false --train false
```

The best result on **CoNLL-2002 Spanish NER** is `F1=88.16`, restore and test the model by running: 

```bash
$ python3 run_datnetp.py --src_language english --tgt_language spanish --at true \
                         --norm_emb true --concat_rnn false --train false
```

Simply change 



This is **TensorFlow** implementation for paper (`* indicates equal contributions`):

- Joey Tianyi Zhou*, Hao Zhang*, Di Jin, Hongyuan Zhu, Meng Fang, Rick Siow Mong Goh and Kenneth Kwok, "[Dual 
Adversarial Neural Transfer for Low-Resource Named Entity Recognition](https://www.aclweb.org/anthology/P19-1336/)", The 
57th Annual Meeting of the Association for Computational Linguistics (ACL 2019, Long Paper, oral), Florence, Italy, 2019.
- Joey Tianyi Zhou*, Hao Zhang*, Di Jin, Xi Peng, "[Dual Adversarial Neural Transfer for Sequence 
Labeling](https://ieeexplore.ieee.org/document/8778733)", IEEE Transactions on Pattern Analysis and Machine Intelligence 
(IEEE TPAMI), 2019.

![overview](/figures/datnet.jpg)
