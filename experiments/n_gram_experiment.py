from keyphrase_extractor import KeyphraseExtractor
from utils import get_dataset, f1_macro


ds = get_dataset()
f1_avg = 0.0
for abstract, gold_keyphrases in ds:
    keyphrases = KeyphraseExtractor(abstract).extract_keyphrases()
    f1_avg += f1_macro(keyphrases, gold_keyphrases) / len(ds)
print(f1_avg)

