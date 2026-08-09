from langsmith import Client
from langchain_openrouter import ChatOpenRouter
from openai import OpenAI
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    HALLUCINATION_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT
)
from dotenv import load_dotenv
from regression_rag import make_pipeline
import re
import os

load_dotenv()

judge_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
JUDGE_MODEL = "openai/gpt-4o-mini"

correctness_judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    judge=judge_client,
    model=JUDGE_MODEL,
    feedback_key="correctness"
)

hallucination_judge = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    judge=judge_client,
    model=JUDGE_MODEL,
    feedback_key="hallucination",
)

retrieval_judge = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    judge=judge_client,
    model=JUDGE_MODEL,
    feedback_key="retrieval_relevance",
)

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    return correctness_judge(
        inputs=inputs["question"],
        outputs=outputs["answer"],
        reference_outputs=reference_outputs["answer"],
    )

def hallucination_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    return hallucination_judge(
        inputs=inputs["question"],
        outputs=outputs["answer"],
        context=outputs["context"],
        reference_outputs=reference_outputs["answer"],
    )

def retrieval_evaluator(inputs: dict, outputs: dict):
    return retrieval_judge(
        inputs=inputs["question"],
        context=outputs["context"],
    )

def main():
    client = Client()
    for version in ["v1_basic", "v2_strict"]:
        client.evaluate(
            make_pipeline(version),
            data="hr-policy-rag-eval",
            evaluators=[
                correctness_evaluator,
                hallucination_evaluator,
                retrieval_evaluator,
            ],
            experiment_prefix=f"hr-rag-{version}",
            max_concurrency=2,
            metadata={"prompt_version": version},
        )


if __name__ == "__main__":
    main()