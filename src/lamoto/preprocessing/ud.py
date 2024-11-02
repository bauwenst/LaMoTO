from typing import List, Dict, Tuple, Union

from tktkt.util.printing import gridify


class FilterAndCorrectUDtypes:

    def __init__(self, dependency_tag_mapping: Dict[str, int]=None):
        self._tag_mapping = dependency_tag_mapping or dict()

    def preprocess(self, words: List[str], heads: List[Union[int,str]], relations: List[str]=None, pos_tags: List[Union[int,str]]=None) -> Tuple[List[str], List[int], List[int], List[int]]:
        """
        First take out all unlabelled words and their corresponding labels. There are two classes of them:
          - MWEs, in which case the words that follow are the decomposition of said word. For example, the MWE
            "doctor's" would produce the UD words ["doctor's", "doctor", "'s"] and all arcs referring to the index
            of "doctor's" or words after it actually refer to one word in the future (here it would be "doctor").
          - Repetition, like the "excited" in "Grace is more excited to see her than she is excited to see me."

        Head indices are counted as in the sequence WITHOUT these words, so for dependency parsing they must be removed.
        """
        if relations is None:
            relations = [None]*len(words)
        else:
            assert self._tag_mapping
        if pos_tags is None:
            pos_tags = [-100]*len(words)
        assert len(words) == len(heads) == len(relations) == len(pos_tags)

        filtered_words     = []
        filtered_heads     = []
        filtered_relations = []
        filtered_pos       = []
        for word, head, relation, pos in zip(words, heads, relations, pos_tags):
            if head != "None":  # For entries where this is true, it is an MWE when  example["upos"][i] == 13 and example["xpos"][i] is None and example["deprel"][i] == "_"
                filtered_words    .append(word)
                filtered_heads    .append(int(head))
                filtered_relations.append(self._tag_mapping[relation] if isinstance(relation, str) else -100)
                filtered_pos      .append(int(pos))
                # print(gridify([example["tokens"], example["xpos"], example["upos"], example["head"], example["deprel"]]))

        return filtered_words, filtered_heads, filtered_relations, filtered_pos
