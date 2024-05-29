"""
The goal is to take a ViT, swap out its linear projection layer for a
traditional tokeniser+embedding, and train it like an LM.

There are three ways to do this. Approach 2 and 3 are modelled below.
    1. Load an LM architecture from_pretrained a ViT checkpoint. Difficult (despite a ViT literally just being BERT with a
       different embedding layer), because the class names of none of the components match.
    2. Load an LM checkpoint, load a ViT checkpoint, and swap the encoders. You need a separate architectural
       class for each head if you want to save and load checkpoints of the result (something that mixes class names of
       the two models).
    3. Load a ViT checkpoint, load an LM checkpoint, substitute the embeddings and add on a head.
       This requires takes quite a bit of work because
            1. ViTMAEModel.forward() still expects pixel_values as keyword input (whilst we give input_ids), which will
               be passed to the embedder without keyword (hence it can expect input_ids once it has been substituted).
            2. ViTMAEModel.forward() expects the output of the embedder to be a 3-tuple, whilst for a normal LM embedder
               there is only one output value (the tensor of embeddings).
            3. There is no ...ForMaskedLM class for ViTMAE.
            4. The embeddings you add and the head you add have to be tied by PreTrainedModel.tie_weights() which
               ViT doesn't have supporting implementations for.
"""
from typing import Tuple, Any

import torch.nn as nn
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEPreTrainedModel, ViTMAEModel, ViTMAEEmbeddings, ViTMAEEncoder
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaEmbeddings, RobertaForMaskedLM, RobertaLMHead

from .augment_model import ModelAugmentation
from ..util.strings import getSubstringAfterLastSlash


def replaceRobertaEncoder(any_roberta_model: RobertaPreTrainedModel, any_vit_model: ViTMAEPreTrainedModel):
    """
    Any transformer model looks roughly like

        embeddings (a matrix that adapts the input format to hidden states)
        encoder    (a sequence of transformations from hidden states to hidden states)
        head       (a small transformation from hidden states to classes)

    In this case, what we want is to take RoBERTa and ViT and replace RoBERTa's encoder by the ViT's encoder, throwing
    away the entirety of RoBERTa's encoder and the ViT's embeddings and head.
    """
    roberta: RobertaModel = any_roberta_model.base_model
    vit: ViTMAEModel = any_vit_model.base_model

    roberta.encoder = vit.encoder


class InjectEncoder(ModelAugmentation):

    def __init__(self, other_model: PreTrainedModel):
        super().__init__("WithEncoderOf(" + getSubstringAfterLastSlash(other_model.name_or_path) + ")")
        self.encoder = other_model.base_model.encoder  # Should be among the lighter parts of the model.

    def augment(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model.base_model.encoder = self.encoder
        return model


######################################################################################################################
from transformers import AutoModel, AutoModelForMaskedLM, AutoConfig

class RobertaWithVitEncoderForMaskedLM(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.mlm = RobertaForMaskedLM(config)  # Takes care of tied embeddings and head weights.
        self.mlm.base_model.encoder = ViTMAEEncoder(config)  # Should not need more config hyperparameters than RoBERTa.

    def base_model(self) -> nn.Module:
        return self.mlm.base_model

    @staticmethod
    def from_checkpoints(lm_checkpoint: str, vit_checkpoint: str):
        language_model = AutoModelForMaskedLM.from_pretrained(lm_checkpoint)
        visual_model   = AutoModel.from_pretrained(vit_checkpoint)
        return RobertaWithVitEncoderForMaskedLM.from_models(language_model, visual_model)

    @staticmethod
    def from_models(language_model: RobertaForMaskedLM, visual_model: ViTMAEPreTrainedModel):
        final_model = RobertaWithVitEncoderForMaskedLM(language_model.config)
        final_model.mlm = language_model
        final_model.mlm.base_model.encoder = visual_model.base_model.encoder
        return final_model

    def forward(self, input_ids, token_type_ids, attention_mask, **kwargs):
        return self.mlm(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            **kwargs
        )


class ViTMAEWithRobertaEmbeddingsModel(ViTMAEPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.vit = ViTMAEModel(config)
        self.vit.base_model.embeddings = Embeddings_LanguageInputVitOutput(config)

    def base_model(self) -> nn.Module:
        return self.vit

    def forward(self, input_ids, **kwargs):
        return self.vit(
            pixel_values=input_ids,
            **kwargs
        )


class ViTMAEWithRobertaEmbeddingsForMaskedLM(ViTMAEPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.vit = ViTMAEWithRobertaEmbeddingsModel(config)
        self.head = RobertaLMHead(config)

    def base_model(self) -> nn.Module:
        return self.vit

    def forward(self, input_ids, **kwargs):
        hidden_states = self.core(input_ids, **kwargs)
        predictions = self.head(hidden_states)
        # TODO: Everything that would happen in a ForMaskedLM model's forward() method ...


class Embeddings_LanguageInputVitOutput(RobertaEmbeddings):
    """
    Has a .forward method that takes in input_ids (not pixel_values)
    but still an output that follows the ViT-MAE interface (see the ViTMAEEmbeddings class).
    """

    def forward(self, input_ids, token_type_ids, **kwargs) -> Tuple[Any,Any,Any]:
        embeddings = super()(input_ids, token_type_ids, **kwargs)
        return embeddings, None, None

########################################################################################################################

class Augmentation_VisionToLanguageEmbedding(ModelAugmentation):
    """
    Replaces the initial linear projection of a ViT by a tokeniser-based embedding.
    TODO: Possibly want to inherit the [CLS] token, but not sure if ViT-MAE actually trains it tho.
    """

    def __init__(self, replace_by_these_embeddings: RobertaEmbeddings=None, max_context_length: int=None):
        """
        :param replace_by_these_embeddings: Existing embeddings (a type embedding matrix |V| x H and a positional
                                            embedding matrix L x H) to use instead. Will obviously mismatch with the
                                            embedding space of the ViT, but may be better than random.
        """
        super().__init__(name="V2L")
        if replace_by_these_embeddings is None and max_context_length is None:
            raise ValueError("If no existing embeddings are given, a maximum context length must be specified to initialise positional embeddings.")

        self.weights = replace_by_these_embeddings
        self.n_positional_embeddings = max_context_length

    def augment(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        assert isinstance(model, ViTMAEPreTrainedModel)
        core: ViTMAEModel    = model.base_model
        assert isinstance(core, ViTMAEModel)
        config: ViTMAEConfig = model.config

        # Insert embedding matrix
        if self.weights is not None:
            new_embeddings = self.weights
        else:
            minimal_config_for_roberta_embeddings = RobertaConfig(
                vocab_size=len(tokenizer.get_vocab()),
                pad_token_id=tokenizer.pad_token_id,
                type_vocab_size=2,
                max_position_embeddings=self.n_positional_embeddings,  # Not sure if we can trust tokenizer.model_max_length. In ViT this is image_size // patch_size, but nowhere in the model is this relevant except the positional embeddings.

                hidden_size=config.hidden_size,
                hidden_dropout_prob=config.hidden_dropout_prob,
                layer_norm_eps=config.layer_norm_eps
            )
            new_embeddings = Embeddings_LanguageInputVitOutput(minimal_config_for_roberta_embeddings)

        core.embeddings = new_embeddings
        return model  # FIXME: This new model has no class, which is problematic because e.g. for weight tying, you need to reimplement get_input_embeddings to Roberta's implementaton, not ViT's.


class Augmentation_AddHeadForMaskedLM(ModelAugmentation):

    def __init__(self):
        super().__init__(name="MLM")

    def augment(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        # Instantiate a new (embeddings+encoder)+head model, and sub in the embeddings+encoder part.
        model.config.vocab_size = len(tokenizer.get_vocab())
        model_with_mlm_head = RobertaForMaskedLM(model.config)  # Takes care of constructing the right classifier object and tied weights.
        model_with_mlm_head.roberta = model.base_model  # `roberta` is any model that has as input an input_ids tensor and as output a tuple starting with one embedding per input_Id.

        # Now tie the newly initialised MLM head to the given embeddings.
        # Should work because:
        #   - tie_weights() ties the reference of get_input_embeddings() to get_output_embeddings().
        #   - RobertaForMaskedLM implements get_output_embeddings() to point to its head.
        #   - RobertaForMaskedLM does not implement get_input_embeddings(). The default implementation searches it on
        #     .base_model, which searches for .roberta, which we have reassigned to the given base model. Hence, as long
        #     as it implements get_input_embeddings(), they get tied.
        model_with_mlm_head.config.tie_word_embeddings = True
        model_with_mlm_head.tie_weights()
        return model_with_mlm_head
