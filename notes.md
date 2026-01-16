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

### Next Steps
- Try to improve the SRI prompt to avoid references to passages.
- Try to improve the SRI prompt to encourage inclusion of all relevant details.
- Run with different models to see if the behavior is consistent.
- Try to use better dataset and evaluate only explicitly what is asked for, to avoid issues with extra details.