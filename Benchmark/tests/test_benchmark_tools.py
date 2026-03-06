from Benchmark.benchmark_tools.artifacts import build_benchmark_probes, summarize_retrieval_results
from Benchmark.benchmark_tools.models import ChunkArtifact, RetrievalCaseResult, RetrievedChunk


def test_build_benchmark_probes_returns_bounded_cases() -> None:
    artifacts = [
        ChunkArtifact(
            faiss_id=index,
            paper_id="paper-a",
            chunk_id=f"paper-a_chunk_{index:04d}",
            file_path=__file__,
            text=f"Chunk text number {index}. This is benchmark content for retrieval testing.",
        )
        for index in range(10)
    ]

    probes = build_benchmark_probes(artifacts, max_cases=4)

    assert len(probes) == 4
    assert probes[0].expected_chunk_id == "paper-a_chunk_0000"
    assert probes[-1].expected_chunk_id == "paper-a_chunk_0006"
    assert probes[0].query.startswith("Retrieve the chunk")


def test_summarize_retrieval_results_calculates_hit_rates() -> None:
    results = [
        RetrievalCaseResult(
            case_id="probe_001",
            query="question one",
            expected_chunk_id="chunk_a",
            expected_answer="answer a",
            reference_contexts=["context a"],
            retrieved_chunks=[
                RetrievedChunk(chunk_id="chunk_a", text="context a", score=0.9, rank=1),
                RetrievedChunk(chunk_id="chunk_b", text="context b", score=0.8, rank=2),
            ],
        ),
        RetrievalCaseResult(
            case_id="probe_002",
            query="question two",
            expected_chunk_id="chunk_c",
            expected_answer="answer c",
            reference_contexts=["context c"],
            retrieved_chunks=[
                RetrievedChunk(chunk_id="chunk_b", text="context b", score=0.7, rank=1),
                RetrievedChunk(chunk_id="chunk_c", text="context c", score=0.6, rank=2),
            ],
        ),
    ]

    summary = summarize_retrieval_results(results)

    assert summary["num_cases"] == 2
    assert summary["hit_at_1"] == 0.5
    assert summary["hit_at_3"] == 1.0
    assert summary["mrr"] == 0.75
    assert summary["average_top_score"] == 0.8
