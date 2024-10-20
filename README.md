# LaMoTO
Language Modelling Tasks as Objects (LaMoTO) provides a framework for language model training (masked and causal, pretraining and finetuning) where the tasks, not just the models, are classes themselves.
It abstracts over the HuggingFace `transformers.Trainer` with one goal: reduce the entire model training process to a single
method call `task.train(hyperparameters)`.

## Usage
Let's say you want to train a RoBERTa-base model for dependency parsing (for which, by the way, there is no HuggingFace
class). This is how you would do that in LaMoTO, supported by the magic of [ArchIt](https://github.com/bauwenst/ArchIt):

```python
from archit.instantiation.basemodels import RobertaBaseModel
from archit.instantiation.heads import DependencyParsingHeadConfig, BaseModelExtendedConfig
from lamoto.tasks import DP, getDefaultHyperparameters

# Define task hyperparameters.
hp = getDefaultHyperparameters()
hp.MODEL_CONFIG_OR_CHECKPOINT = "roberta-base"
hp.archit_basemodel_class = RobertaBaseModel
hp.archit_head_config = DependencyParsingHeadConfig(
    head_dropout=0.33,
    extended_model_config=BaseModelExtendedConfig(
        layer_pooling=1
    )
)

# Instantiate language modelling task as object, and train model.
task = DP()
task.train(hyperparameters=hp)
```

## Features
- [x] Train models on >15 pre-training/fine-tuning tasks. See a list by importing `from lamoto.tasks`.
  - Model architectures come from [ArchIt](https://github.com/bauwenst/ArchIt), which means that as long as you have a
    `BaseModel` wrapper for your language model backbone, you can train it on any task, regardless of whether you wrote
    code defining the backbone-with-head architecture required for that task.
  - Custom (i.e. given) architectures are also supported.
- [x] Evaluate models with a superset of the metrics in HuggingFace's `evaluate`, with custom inference procedures (see e.g. strided pseudo-perplexity or bits-per-character).
- [x] Augment datasets before training or evaluating by somehow perturbing them.
- [x] Supports [TkTkT](https://github.com/bauwenst/TkTkT) tokenisers.
- [x] Weights-and-Biases integration.

## Installation
If you don't want to edit the source code yourself, run
```
pip install "lamoto[github] @ git+https://github.com/bauwenst/LaMoTO"
```
and if you do, instead run
```
git clone https://github.com/bauwenst/LaMoTO
cd LaMoTO
pip install -e .[github]
```
To be able to use the Weights-and-Biases integration, make sure you first run `wandb login` in a command-line terminal 
on the system you want to run on.

<!-- If you are me from the future: don't include the `[github]` tag, it will fuck up the editable installs for the packages I maintain. -->
