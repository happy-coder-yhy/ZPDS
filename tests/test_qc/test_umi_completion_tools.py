from __future__ import annotations

from dataclasses import dataclass

from zpds_prepare.detectors.umi import (
    UmiCandidateView,
    UmiEpisodeCandidate,
    UmiEvidenceIndex,
    UmiGoldAnnotation,
    UmiLabeledSpan,
    UmiProvisionalMetric,
    adapt_umi_delivery,
    binary_classification_metrics,
    build_umi_source_contract_candidate,
    compare_independent_reviews,
    deterministic_config_hash,
    evaluate_threshold_candidates,
    evaluate_umi_hand_applicability,
    stratified_sample_episodes,
)
from zpds_prepare.detectors.umi.cli import build_parser


def _metric() -> UmiProvisionalMetric:
    return UmiProvisionalMetric(
        metric_name="umi.test",
        value=1,
        unit="count",
        applicability="applicable",
        severity="info",
        disposition="keep",
        reason_code="measurement_only_uncalibrated",
        start_ns=None,
        end_ns=None,
        evidence_uri="evidence.parquet",
        producer="test",
        version="1",
        config_hash=deterministic_config_hash({}),
        stream_id="robot0",
        source_session_id="umi-test",
    )


def test_hand_guard_cannot_emit_hand_absent() -> None:
    result = evaluate_umi_hand_applicability(
        build_umi_source_contract_candidate()
    )

    assert result.applicability == "not_applicable"
    assert result.run_human_hand_model is False
    assert result.emitted_reason_codes == ()
    assert "HAND_ABSENT" in result.forbidden_reason_codes


@dataclass
class _FakeSharedAdapter:
    def adapt_metric(self, metric: UmiProvisionalMetric) -> dict:
        return {"shared_metric": metric.metric_name}

    def adapt_view(self, view: UmiCandidateView) -> dict:
        return {"shared_view": view.name, "status": view.status}

    def adapt_revision(self, revision: dict) -> dict:
        return {"shared_revision": revision["schema_version"]}


def test_shared_adapter_boundary_converts_without_detector_dependency() -> None:
    metric = _metric()
    index = UmiEvidenceIndex(
        source_session_id="umi-test",
        producer="test",
        version="1",
        config_hash=metric.config_hash,
        artifacts={},
        metrics=(metric,),
    )
    view = UmiCandidateView(
        name="vio_ready_candidate",
        status="candidate_pass",
        applicability="applicable",
        disposition="keep",
        dependencies=("umi_vio_quality",),
    )

    delivery = adapt_umi_delivery(
        index,
        {view.name: view},
        {"schema_version": "umi-revision-candidate-v1"},
        _FakeSharedAdapter(),
    )

    assert delivery.metrics == ({"shared_metric": "umi.test"},)
    assert delivery.views[view.name]["shared_view"] == view.name
    assert delivery.revision == {
        "shared_revision": "umi-revision-candidate-v1"
    }


def test_stratified_sampler_is_deterministic_and_covers_strata() -> None:
    episodes = [
        UmiEpisodeCandidate(
            episode_id=f"episode-{index:02d}",
            task=f"task-{index % 4}",
            camera_group=f"camera-{index % 2}",
            duration_s=float(5 + (index % 3) * 20),
            outcome="success" if index % 2 else "failure",
            metadata={},
        )
        for index in range(40)
    ]

    first = stratified_sample_episodes(episodes, target_count=30)
    second = stratified_sample_episodes(list(reversed(episodes)), target_count=30)

    assert [episode.episode_id for episode in first] == [
        episode.episode_id for episode in second
    ]
    assert len(first) == 30
    assert len({episode.task for episode in first}) == 4


def test_review_comparison_requires_adjudication_for_disagreement() -> None:
    annotations = [
        UmiGoldAnnotation("e1", "r1", True, True),
        UmiGoldAnnotation("e1", "r2", True, True),
        UmiGoldAnnotation(
            "e2",
            "r1",
            True,
            False,
            spans=(UmiLabeledSpan("vio_gap", 10, 20),),
        ),
        UmiGoldAnnotation("e2", "r2", False, False),
        UmiGoldAnnotation("e3", "r1", True, None),
    ]

    report = compare_independent_reviews(annotations)

    assert report["agreement_count"] == 1
    assert report["agreement_rate"] == 0.5
    assert report["disagreements"][0]["episode_id"] == "e2"
    assert report["disagreements"][0]["adjudication_required"] is True
    assert report["incomplete_episode_ids"] == ["e3"]


def test_threshold_report_never_installs_candidate_automatically() -> None:
    report = evaluate_threshold_candidates(
        [1.0, 2.0, 8.0, 10.0],
        [True, True, False, False],
    )

    assert report["recommended_candidate"]["threshold"] == 2.0
    assert report["formal_threshold"] is False
    assert report["automatic_install"] is False
    assert report["requires_adjudicated_gold"] is True
    assert binary_classification_metrics(
        [True, True, False, False],
        [True, False, False, False],
    )["precision"] == 0.5


def test_cli_parser_requires_explicit_paths() -> None:
    args = build_parser().parse_args(
        [
            "--dataset",
            "source.mcap",
            "--output",
            "delivery",
            "--cache",
            "cache",
        ]
    )

    assert args.dataset.name == "source.mcap"
    assert args.output.name == "delivery"
    assert args.cache.name == "cache"
