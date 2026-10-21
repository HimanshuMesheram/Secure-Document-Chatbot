import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from core.config import validate_configuration
from core.llm import get_embeddings, get_llm


def load_dataset(path: Path) -> Dataset:
    """Load JSONL records and validate the required RAGAS fields."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            required = {
                "question",
                "answer",
                "contexts",
                "ground_truth",
            }

            if not required.issubset(record):
                raise ValueError(
                    f"Line {line_number} is missing required fields."
                )

            if not isinstance(record["contexts"], list):
                raise ValueError(
                    f"Line {line_number}: contexts must be a list."
                )

            records.append(record)

    if not records:
        raise ValueError("The evaluation dataset is empty.")

    return Dataset.from_list(records)


def main() -> None:
    validate_configuration()

    dataset_path = Path(__file__).with_name("dataset.jsonl")
    dataset = load_dataset(dataset_path)

    evaluator_llm = LangchainLLMWrapper(get_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    # These metrics use the configured evaluation model.
    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
    )

    output_path = Path(__file__).with_name("results.csv")
    results.to_pandas().to_csv(output_path, index=False)

    print("Evaluation completed.")
    print(results)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()