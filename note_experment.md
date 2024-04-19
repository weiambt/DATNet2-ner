
## 2024年4月18日 19:11:13
### 测试 datnetp
bsub -n 1 -q normal -o output2024-4-18_%J.txt python run_datnetp.py --src_task ontonotes_ner --tgt_task conll03_en_ner --elmo false --at True --share_word True

### 测试 datnetf

bsub -n 1 -q normal -o output2024-4-18_%J.txt python run_datnetf.py --src_task ontonotes_ner --tgt_task conll03_en_ner --elmo false --at True --share_word True

### 测试 datnetp 带elmo

修改了elmo的位置

bsub -n 1 -q normal -o output2024-4-18_%J.txt python run_datnetp.py --src_task ontonotes_ner --tgt_task conll03_en_ner --elmo true --at True --share_word True --model_name datnetp_test

报错，修改了elmo的url也不对，交到作业管理器就报错

## 