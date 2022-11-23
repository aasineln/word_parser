from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    LOC,
    NamesExtractor,
    DatesExtractor,
    MoneyExtractor,
    AddrExtractor,

    Doc
)

if __name__ == '__main__':
    segmenter = Segmenter()
    morph_vocab = MorphVocab()

    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_tagger = NewsNERTagger(emb)

    names_extractor = NamesExtractor(morph_vocab)
    dates_extractor = DatesExtractor(morph_vocab)
    money_extractor = MoneyExtractor(morph_vocab)
    addr_extractor = AddrExtractor(morph_vocab)
    text = 'Доброго дня из Санкт-Петербурга. Мэра города Москвы Собянина поздравили с юбилеем.'

    print(text)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)
    doc.segment(segmenter)

    for span in doc.spans:
        span.normalize(morph_vocab)

    names_dict = {_.text: _.normal for _ in doc.spans if _.text != _.normal}
    print(names_dict)