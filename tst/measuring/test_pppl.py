import numpy as np
import torch
from transformers import AutoTokenizer, RobertaTokenizer, AutoModelForMaskedLM


def prototype_body(context_length: int=None, mask_specials=False):
    tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained("roberta-base")

    f = 0.25

    # Sizes
    L = tokenizer.model_max_length if context_length is None else context_length
    L_middle = 1
    L_right = int((L - L_middle) * f)
    L_left  = (L - L_middle) - L_right

    # Input
    tokens = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    tokens: torch.Tensor = torch.tensor(tokens)
    n = len(tokens)

    # Edge cases: smaller right side, or no right side and smaller left side.
    if n >= L_left + L_middle + L_right:
        # Normal case; there can be a top, middle and bottom.
        pass
    elif n >= L_left + L_middle:
        L_right = n - (L_left + L_middle)
    else:
        L_right  = 0
        L_middle = 0
        L_left   = n

    L = L_left + L_middle + L_right  # Smaller than the original L when n was smaller than the original L.

    # Contexts
    top    = tokens[:L].repeat(L_left, 1)  # Copies the first available window in the context.
    middle = tokens.unfold(dimension=0, size=L, step=1) if L_middle else torch.zeros(size=(0,L), dtype=torch.int64)  # Window shifting with step 1 of the same size as that available window.
    bottom = tokens[n-L:].repeat(L_right, 1)  # Tolerates L_right == 0. Also, n-L can never be negative since L <= n by the above.

    # Masks
    mask_top    = torch.eye(L_left, L)
    mask_middle = torch.zeros(n-L_left-L_right, L)
    if L_middle:
        mask_middle[:, L_left] = 1
    mask_bottom = torch.roll(torch.eye(L_right, L), shifts=-L_right, dims=1)

    # Concatenate, then filter out the rows belonging to special tokens
    full_ids  = torch.cat((top, middle, bottom))
    full_mask = torch.cat((mask_top, mask_middle, mask_bottom))

    if mask_specials:
        special_tokens_mask = torch.tensor(tokenizer.get_special_tokens_mask(tokens, already_has_special_tokens=True))
        full_ids  = full_ids[special_tokens_mask == 0, :]
        full_mask = full_mask[special_tokens_mask == 0, :]

        n -= special_tokens_mask.sum().item()

    # Finally, apply the constructed mask to the constructed IDs.
    input_ids = full_ids.masked_fill(full_mask == 1, tokenizer.mask_token_id)
    labels    = full_ids.masked_fill(full_mask != 1, -100)
    attention_mask = torch.ones_like(input_ids)

    with np.printoptions(linewidth=100):
        print(input_ids.numpy())
        print(labels.numpy())

    return input_ids, labels, attention_mask


def prototype_inference():
    model = AutoModelForMaskedLM.from_pretrained("roberta-base")
    input_ids, labels, attention_mask = prototype_body(model.config.max_position_embeddings, mask_specials=True)

    with torch.inference_mode():
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    print(loss)


def test_body():
    prototype_body(context_length=5)
    prototype_body(context_length=13)
    prototype_body(context_length=200)

    prototype_body(context_length=200, mask_specials=True)


def test_actual():
    from lamoto.measuring.pppl import pppl
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dataset = [{"text": "This is the first example here."}, {"text": "This is the second example of this dataset."}]

    print(pppl(
        AutoModelForMaskedLM.from_pretrained("roberta-base"),
        AutoTokenizer.from_pretrained("roberta-base"),
        dataset,
        rightward_fraction=0.5,
        tqdm_dataset_size=len(dataset)
    ))


if __name__ == "__main__":
    test_body()
    test_actual()