from datasets import load_dataset


if __name__ == '__main__':
    el_es = load_dataset('opus_books', 'el-en')
    es_el = load_dataset('opus_books', 'el-es')




    hellsinki_el_es = load_dataset('Helsinki-NLP/tatoeba_mt', 'ell-eng')
    hellsinki_es_el = load_dataset('Helsinki-NLP/tatoeba_mt', 'eng-ell')

    pass

