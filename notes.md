# Experiment Notes

## Run 1
Dataset: ficticious_nq
Model: Qwen3 8B

### Results
Structured Reasoning Injection (SRI) constantly slightly behind Passage Injection (PI) and Expert Messages. Usually,
the reason is that the model answers the overall question correctly, but ommits some relevant details that were in the
context but not specifically asked for. The other pipelines tend to include these details more often.
Also, SRI sometimes includes references to provided passages (i.e., "According to passage 2..."), which is not desired
and explicitly prompted against.
Worth mentioning is that if looking at thresholded accuracy (i.e., is the main answer correct or not), SRI is on par with PI and Expert Messages.
All of them are almost perfect with this metric, indicating that the dataset might be too easy. Not sure how to fix that yet.

### Next Steps
- Try to improve the SRI prompt to avoid references to passages.
- Try to improve the SRI prompt to encourage inclusion of all relevant details.
- Run with different models to see if the behavior is consistent.
- Try to use better dataset and evaluate only explicitly what is asked for, to avoid issues with extra details.

## Run 2
Dataset: ficticious_nq_rev (same but swapped question and answer generation order, questions first now)
Model: Qwen3 8B

Changed SRI prompt to include more relevant details, even if not explicitly asked for.

### Results
No significant change. SRI still slightly behind PI and Expert Messages.

## Run 3
Dataset: ficticious_nq_rev
Model: Qwen3 8B

Added more noise to test if SRI can better handle irrelevant information.
From 1.0 to 10.0 noise level. (noise to actual context ratio)

### Results
No significant changes in any pipeline. Order is still PI > Expert > SRI

### Next Steps
- Test smaller models to see if SRI has more impact there.

## Run 4
Dataset: ficticious_nq_rev
Model: Qwen3 1.7B

Moved to smaller model to see if SRI has more impact.

### Results
No significant changes in any pipeline. Order is still PI > Expert > SRI

## Run 5
Dataset: ficticious_nq_rev
Model: Qwen3 1.7B

Changed Correctness metric to only consider main answer, ignoring extra details.

### Results
Slightly higher scores across the board, but order is still PI > Expert > SRI.

### Next Steps
- Try directly comparing noise vs no noise in PI just to see if noise has any effect at all.
  Can even run extreme noise levels to see if it breaks anything.
- Try different metrics, like faithfulness or something else.
- Try to improve SRI prompt further or use preprocessing to filter irrelevant passages and then feed relevant ones 1:1 instead of summarizing them.
  Can also try more structured guidance, like "Follow these steps..."
- Try different datasets?
