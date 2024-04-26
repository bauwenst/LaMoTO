# LaMoTO
Language Modelling Tasks as Objects (LaMoTO) provides a framework for language model training (masked and causal, pretraining and finetuning) where the tasks, not just the models, are classes themselves.

## Installation
If you don't want to edit the source code yourself, run
```
pip install "lamoto[github] @ git+https://github.com/bauwenst/LaMoTO.git"
```
and if you do, instead run
```
git clone https://github.com/bauwenst/LaMoTO
cd LaMoTO
pip install -e .[github]
```
If you are me from the future: don't include the `[github]` tag, it will fuck up the editable installs for the packages I maintain.