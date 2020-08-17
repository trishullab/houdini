# HOUDINI - Lifelong Learning
Code for running the experiments, described in our NeurIPS 2018 paper "HOUDINI: Lifelong Learning as Program Synthesis" (https://arxiv.org/abs/1804.00218)

## Requirements:

- Pytorch 0.3.1


## Datasets
The datasets are automatically downloaded when needed.

## Running

### Houdini
    For the counting sequences 1, 2, 3, and the long sequence:
    usage: python HOUDINI/Eval/CS_LS.py [-h] [--synthesizer {enumerative,evolutionary}]
                                 --taskseq {cs1,cs2,cs3,ls} [--dbg]
    
    For graph the sequences:
    usage: python HOUDINI/Eval/GraphSeq.py [-h] [--synthesizer {enumerative,evolutionary}]
                                 --taskseq {gs1, gs2} [--dbg]

    optional arguments:
      -h, --help            show this help message and exit
      --synthesizer {enumerative,evolutionary}
                            Synthesizer type. (default: enumerative)
      --taskseq {cs1,cs2,cs3,ls}
                            Task Sequence
      --dbg                 If set, the sequences run for a tiny amount of data



### Baselines

Counting Sequence 1, Standalone: `python Baselines/CountingSeqLongSeq.py cs1 sa`

Counting Sequence 1, low-level-transfer: `python Baselines/CountingSeqLongSeq.py cs1 wt`

Counting Sequence 2, Standalone: `python Baselines/CountingSeqLongSeq.py cs2 sa`

Counting Sequence 2, low-level-transfer: `python Baselines/CountingSeqLongSeq.py cs2 wt`

Counting Sequence 3, Standalone: `python Baselines/CountingSeqLongSeq.py cs3 sa`

Counting Sequence 3, low-level-transfer: `python Baselines/CountingSeqLongSeq.py cs3 wt`


Long Sequence, Standalone: `python Baselines/CountingSeqLongSeq.py ls sa`

Long Sequence, low-level-transfer: `python Baselines/CountingSeqLongSeq.py ls wt`
