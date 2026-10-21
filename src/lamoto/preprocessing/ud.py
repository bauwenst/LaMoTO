from typing import Union

from datasets import load_dataset, DatasetDict

from tktkt.util.printing import gridify


class UniversalDependenciesDataset:
    def __init__(self, name: str):
        self.name = name

    def load(self) -> DatasetDict:  # Takes care of remote code and renaming splits and so on.
        dataset = load_dataset("universal-dependencies/universal_dependencies", self.name)
        if "dev" in dataset:
            dataset["validation"] = dataset.pop("dev")
        return dataset


class FilterAndCorrectUDtypes:

    def __init__(self, pos_tag_mapping: dict[str, int]=None, dependency_tag_mapping: dict[str, int]=None):
        self._pos_mapping = pos_tag_mapping or dict()
        self._dp_mapping  = dependency_tag_mapping or dict()

    def preprocess(self, words: list[str], heads: list[Union[int,str]], relations: list[str]=None, pos_tags: list[Union[int,str]]=None) -> tuple[list[str], list[int], list[int], list[int]]:
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
            assert self._dp_mapping
        if pos_tags is None:
            pos_tags = [-100]*len(words)
        else:
            assert self._pos_mapping
        assert len(words) == len(heads) == len(relations) == len(pos_tags)

        filtered_words     = []
        filtered_heads     = []
        filtered_relations = []
        filtered_pos       = []
        for word, head, relation, pos in zip(words, heads, relations, pos_tags):
            if head != "None":  # For entries where this is true, it is an MWE when  example["upos"][i] == 13 and example["xpos"][i] is None and example["deprel"][i] == "_"
                filtered_words    .append(word)
                filtered_heads    .append(int(head))
                filtered_relations.append(self._dp_mapping[relation] if isinstance(relation, str) else -100)
                filtered_pos      .append(self._pos_mapping[pos]     if isinstance(pos, str) else -100)
                # print(gridify([example["tokens"], example["xpos"], example["upos"], example["head"], example["deprel"]]))

        return filtered_words, filtered_heads, filtered_relations, filtered_pos
