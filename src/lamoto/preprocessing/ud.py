from typing import List, Dict, Tuple

from tktkt.util.printing import gridify


class FilterAndCorrectUDtypes:

    def __init__(self, tag_mapping: Dict[str, int]):
        self._tag_mapping = tag_mapping

    def preprocess(self, words: List[str], heads: List[str], relations: List[str]) -> Tuple[List[str], List[int], List[int]]:
        """
        First take out all unlabelled words and their corresponding labels. Head indices are counted as in the
        sequence WITHOUT these words. There are two classes of them:
          - MWEs, in which case the words that follow are the decomposition of said word. For example, the MWE
            "doctor's" would produce the UD words ["doctor's", "doctor", "'s"] and all arcs referring to the index
            of "doctor's" or words after it actually refer to one word in the future (here it would be "doctor").
          - Repetition, like the "excited" in "Grace is more excited to see her than she is excited to see me."
        """
        assert len(words) == len(heads) == len(relations)

        filtered_words     = []
        filtered_heads     = []
        filtered_relations = []
        for word, head, relation in zip(words, heads, relations):
            if head != "None":  # For entries where this is true, it is an MWE when  example["upos"][i] == 13 and example["xpos"][i] is None and example["deprel"][i] == "_"
                filtered_words    .append(word)
                filtered_heads    .append(int(head))
                filtered_relations.append(self._tag_mapping[relation])
                # print(gridify([example["tokens"], example["xpos"], example["upos"], example["head"], example["deprel"]]))

        return filtered_words, filtered_heads, filtered_relations
