# LaMoTO design notes
## Why do we need a `sneakyLogitTransform` specifically for dependency parsing?
The issue to solve is that when you are evaluating in the HuggingFace trainer, all predictions and labels are
concatenated into one big tensor. The problem for DP arc predictions is that the amount of possible labels
CHANGES every batch, because the labels are positions inside the given sentence. Hence, you can't concatenate the
predictions for several batches because they don't have same amount of prediction classes.
You don't see this problem in training nor the first evaluation batch, because there is no batch interaction (yet) there.

As an example of the error you get:
```
   RuntimeError: The expanded size of the tensor (60) must match the existing size (58) at non-singleton dimension 2.
   Target sizes: [32, 58, 60].  Tensor sizes: [32, 58, 58]
```
The first batch had 60 positions, the second batch had 58.

Any solution for this has to basically force the Trainer to not accumulate logits, and instead commit them to the
evaluation metric immediately (which is how `supar` does it). Here's how you could do that:
- Write a custom Trainer that has an evaluation_loop that just doesn't accumulate.
   The problem with this approach is that Trainer actually has a bunch of useful acceleration code.
- Use the existing Trainer but with empty validation dataset and use some kind of callback to evaluate the model inside
   the callback instead of in Trainer's evaluation loop.
- Trainer has an argument preprocess_logits_for_metrics that is called like `logits = preprocess_logits_for_metrics(logits,labels)`
   before saving the logits. 

The first two approaches have their place, namely when you want metrics that aren't logit-based, like strided
PPL in causal LM. That's not the case for DP though. Here's how you could use the last method:
1. Instantiate the UAS/LAS metric
2. Capture it inside the preprocess_logits_for_metrics function and compute it immediately
3. Let preprocess_logits_for_metrics return empty tensors as logits and labels
4. Let computeMetrics return that metric's value.

Another approach:
1. Let preprocess_logits_for_metrics flatten the B x L x L logits into a B x L² tensor.
2. When the time comes to computeMetrics, have some way to identify the different lengths L and turn
  them back into squares inside a Metric.compute.

Yet another approach:
1. Compress the B x L x L logits into B x L x 1 class argmaxes, rather than letting the metric do this.
2. Let computeMetrics finish the process with UAS/LAS.

For DP, we do the first of these, except more elegantly, we capture `self` (the only time Python allows currying is in 
expressions `self.method`) and then we access the metric instance that is already present in `self` anyway.
