from rag import rag_pipeline
from rag import llm, embeddings
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, Faithfulness, ResponseRelevancy, FactualCorrectness

eval_questions = [
    {
        "question": "How many casual leaves am I entitled to per year?",
        "reference": "Employees get 12 casual leaves per calendar year.",
    },
    {
        "question": "What is the notice period for resignation?",
        "reference": "The notice period is 60 days for confirmed employees.",
    }
]

samples = []

for item in eval_questions:
    result = rag_pipeline({"question": item["question"]})

    # split the joined string back into a list of chunks
    contexts = result["context"].split("\n\n")
    # if your rag_pipeline joins chunks with a different separator, change "\n\n" above to match it

    sample = SingleTurnSample(
        user_input=item["question"],
        retrieved_contexts=contexts,
        response=result["answer"],
        reference=item["reference"],
    )
    samples.append(sample)

dataset = EvaluationDataset(samples=samples)

evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

# Run the evaluation
result = evaluate(
    dataset=dataset,
    metrics=[
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
        FactualCorrectness(),
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

print(result)

df = result.to_pandas()
print(df)

df.to_csv("evaluation_results.csv", index=False)
