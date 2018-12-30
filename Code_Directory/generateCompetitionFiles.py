from multiprocessing import cpu_count
from main import *
# usage -> python3 generateCompetitionFiles.py

BEAM_SIZE = 1

def create_taggers(model_file, inference_file):
    """
    create taggers list
    """
    sentences, _ = load_test(inference_file, comp=True)
    viterbi = pickle.load(open(model_file, 'rb'))
    tagger = parallel_viterbi(viterbi, sentences, BEAM_SIZE, cpu_count())
    return tagger, sentences


def create_file(inference_file, sentences, tagger):
    """
    create competition file
    """
    if inference_file == 'comp.words':
        f_name = 'comp_m1_203764618.wtag'
    else:
        f_name = 'comp_m2_203764618.wtag'
    fh = open(f_name, 'w')
    for sen_idx, sentence in enumerate(sentences):
        for idx, (word, tag) in enumerate(zip(sentence, tagger[sen_idx])):
            if idx == len(sentence) - 2:
                fh.write(word + '_' + tag)
            elif word != 'STOP' and word != '*':
                fh.write(word + '_' + tag + ' ')
        if sen_idx != len(sentences) - 1:
            fh.write('\n')
    fh.close()


if __name__ == '__main__':

    # create taggers
    comp_tagger, comp_sentences = create_taggers('./cache/model1.pickle', 'comp.words')
    comp2_tagger, comp2_sentences = create_taggers('./cache/model2.pickle', 'comp2.words')

    # create files
    create_file('comp.words', comp_sentences, comp_tagger)
    create_file('comp2.words', comp2_sentences, comp2_tagger)


