# LaMoTO
Language Modelling Tasks as Objects (LaMoTO) provides a framework for language model training (masked and causal, pretraining and finetuning) where the tasks, not just the models, are classes themselves.

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
hp.MODEL_CLASS = RobertaBaseModel
hp.HEAD_CONFIG = DependencyParsingHeadConfig(
    head_dropout=0.33,
    extended_model_config=BaseModelExtendedConfig(
        layer_pooling=1
    )
)

# Instantiate language modelling task as object, and train model.
task = DP()
task.train(hyperparameters=hp)
```
See all the supported pre-training and fine-tuning tasks under `lamoto.tasks`.

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
If you are me from the future: don't include the `[github]` tag, it will fuck up the editable installs for the packages I maintain.