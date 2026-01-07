"""
Consensus Scoring and Aggregation for EASD Evaluations

Combines results from multiple judge models into a single evaluation
with explicit consensus states and evidence validation.
"""

import statistics
from enum import Enum
from typing import Any, Optional

from .openrouter import JudgeResponse


# Critical drivers that must score ≥4 to pass DCT
CRITICAL_DRIVERS = [
    ("evidence", "data_validity"),
    ("assumptions", "validation"),
    ("scenarios", "scenario_depth"),
    ("decision", "conclusion_clarity"),
    ("decision", "decision_framing"),
]

# DCT threshold
DCT_REQUIRED_SCORE = 68
DCT_MAX_SCORE = 80


class ConsensusState(Enum):
    """
    Explicit consensus states for evaluation reliability.
    
    GREEN: 3/3 valid judge outputs
    AMBER: 2/3 valid (one failed parse or invalid schema)
    RED: <2 valid
    """
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


def get_nested_value(data: dict, *keys) -> Any:
    """Safely get a nested value from a dict."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return None
    return data


def extract_scores_from_evaluation(evaluation: dict) -> dict:
    """
    Extract all sub-criterion scores from an evaluation.
    
    Returns dict like:
    {
        "evidence": {"source_diversity": 4, "data_validity": 5, ...},
        "assumptions": {...},
        "scenarios": {...},
        "decision": {...}
    }
    """
    easd = get_nested_value(evaluation, "response_evaluation", "easd_scores")
    if not easd:
        return {}
    
    scores = {}
    for dimension in ["evidence", "assumptions", "scenarios", "decision"]:
        dim_data = easd.get(dimension, {})
        scores[dimension] = {}
        for key, value in dim_data.items():
            if key == "aggregate":
                continue
            if isinstance(value, dict) and "score" in value:
                scores[dimension][key] = value["score"]
    
    return scores


def determine_consensus_state(judge_responses: list[JudgeResponse]) -> tuple[ConsensusState, dict]:
    """
    Determine the explicit consensus state based on valid judge outputs.
    
    Returns:
        (ConsensusState, details_dict)
    """
    total = len(judge_responses)
    successful = sum(1 for r in judge_responses if r.success and r.evaluation)
    failed = [r for r in judge_responses if not r.success or not r.evaluation]
    
    details = {
        "total_judges": total,
        "valid_outputs": successful,
        "failed_judges": [
            {
                "model": r.model_alias,
                "error": r.error,
                "repair_attempted": getattr(r, 'repair_attempted', False)
            }
            for r in failed
        ]
    }
    
    if total == 0:
        return ConsensusState.RED, details
    
    if successful == total:
        return ConsensusState.GREEN, details
    elif successful >= 2:
        return ConsensusState.AMBER, details
    else:
        return ConsensusState.RED, details


def validate_evidence_pointers(
    evaluation: dict,
    extracted_data: Optional[dict] = None
) -> list[dict]:
    """
    Validate that evidence quotes exist in the referenced source paths.
    
    Returns list of validation issues found.
    """
    issues = []
    
    if extracted_data is None:
        # Can't validate without source data
        return [{"type": "warning", "message": "Source data not provided for evidence validation"}]
    
    easd = get_nested_value(evaluation, "response_evaluation", "easd_scores")
    if not easd:
        return issues
    
    # Build a searchable text index from extracted data
    source_texts = {
        "response.content": extracted_data.get("response", {}).get("content", ""),
        "request.decision_context": extracted_data.get("request", {}).get("decision_context", ""),
        "request.source_documents": extracted_data.get("request", {}).get("source_documents", ""),
        "request.summarized_documents": extracted_data.get("request", {}).get("summarized_documents", ""),
        "request.project_instructions": extracted_data.get("request", {}).get("project_instructions", ""),
        "request.other_chapters": extracted_data.get("request", {}).get("other_chapters", ""),
        "request.prereq_chapters": extracted_data.get("request", {}).get("prereq_chapters", ""),
    }
    
    # Validate each evidence pointer
    for dimension in ["evidence", "assumptions", "scenarios", "decision"]:
        dim_data = easd.get(dimension, {})
        for sub_criterion, sub_data in dim_data.items():
            if sub_criterion == "aggregate" or not isinstance(sub_data, dict):
                continue
            
            evidence_list = sub_data.get("evidence", [])
            for i, evidence in enumerate(evidence_list):
                if not isinstance(evidence, dict):
                    continue
                
                quote = evidence.get("quote", "")
                source_path = evidence.get("source_path", "")
                claim = evidence.get("claim", "")
                
                if not quote or not source_path:
                    issues.append({
                        "type": "missing_evidence",
                        "location": f"{dimension}.{sub_criterion}.evidence[{i}]",
                        "message": "Evidence missing quote or source_path"
                    })
                    continue
                
                # Check if quote exists in the specified source
                source_text = source_texts.get(source_path, "")
                
                # Also check if it's a shortened path
                for full_path, text in source_texts.items():
                    if source_path in full_path or full_path.endswith(source_path):
                        source_text = text
                        break
                
                if not source_text:
                    issues.append({
                        "type": "invalid_source_path",
                        "location": f"{dimension}.{sub_criterion}.evidence[{i}]",
                        "source_path": source_path,
                        "message": f"Source path not found in extracted data"
                    })
                elif quote not in source_text:
                    # Try fuzzy match - quote might be slightly reformatted
                    normalized_quote = " ".join(quote.split()).lower()
                    normalized_source = " ".join(source_text.split()).lower()
                    
                    if normalized_quote not in normalized_source:
                        issues.append({
                            "type": "quote_not_found",
                            "location": f"{dimension}.{sub_criterion}.evidence[{i}]",
                            "quote_preview": quote[:100] + "..." if len(quote) > 100 else quote,
                            "source_path": source_path,
                            "message": "Quote not found verbatim in source (possible hallucination)"
                        })
    
    return issues


def aggregate_scores(judge_responses: list[JudgeResponse]) -> dict:
    """
    Aggregate scores from multiple judges.
    
    Returns averaged scores with variance information.
    """
    # Collect all successful evaluations
    evaluations = [r.evaluation for r in judge_responses if r.success and r.evaluation]
    
    if not evaluations:
        return {}
    
    # Extract scores from each evaluation
    all_scores = [extract_scores_from_evaluation(e) for e in evaluations]
    
    # Aggregate by dimension and sub-criterion
    aggregated = {}
    variances = {"E": [], "A": [], "S": [], "D": []}
    dimension_map = {"evidence": "E", "assumptions": "A", "scenarios": "S", "decision": "D"}
    
    for dimension in ["evidence", "assumptions", "scenarios", "decision"]:
        aggregated[dimension] = {}
        
        # Get all sub-criteria for this dimension
        all_sub_criteria = set()
        for scores in all_scores:
            if dimension in scores:
                all_sub_criteria.update(scores[dimension].keys())
        
        for sub_criterion in all_sub_criteria:
            values = []
            for scores in all_scores:
                if dimension in scores and sub_criterion in scores[dimension]:
                    values.append(scores[dimension][sub_criterion])
            
            if values:
                avg = statistics.mean(values)
                aggregated[dimension][sub_criterion] = round(avg, 2)
                
                if len(values) > 1:
                    variance = statistics.stdev(values)
                    variances[dimension_map[dimension]].append(variance)
        
        # Calculate aggregate for dimension
        if aggregated[dimension]:
            aggregated[dimension]["aggregate"] = round(
                statistics.mean(aggregated[dimension].values()), 2
            )
    
    # Calculate overall variance per dimension
    variance_summary = {}
    for dim_key, var_list in variances.items():
        if var_list:
            variance_summary[dim_key] = round(statistics.mean(var_list), 2)
        else:
            variance_summary[dim_key] = 0.0
    
    return {
        "scores": aggregated,
        "variance": variance_summary
    }


def calculate_agreement_rate(judge_responses: list[JudgeResponse]) -> float:
    """
    Calculate the proportion of sub-criteria where all judges agreed within 1 point.
    """
    evaluations = [r.evaluation for r in judge_responses if r.success and r.evaluation]
    
    if len(evaluations) < 2:
        return 1.0  # Perfect agreement with single judge
    
    all_scores = [extract_scores_from_evaluation(e) for e in evaluations]
    
    agreed = 0
    total = 0
    
    for dimension in ["evidence", "assumptions", "scenarios", "decision"]:
        all_sub_criteria = set()
        for scores in all_scores:
            if dimension in scores:
                all_sub_criteria.update(scores[dimension].keys())
        
        for sub_criterion in all_sub_criteria:
            values = []
            for scores in all_scores:
                if dimension in scores and sub_criterion in scores[dimension]:
                    values.append(scores[dimension][sub_criterion])
            
            if len(values) >= 2:
                total += 1
                max_diff = max(values) - min(values)
                if max_diff <= 1:
                    agreed += 1
    
    return round(agreed / total, 2) if total > 0 else 1.0


def generate_flags(judge_responses: list[JudgeResponse], variance: dict) -> list[str]:
    """
    Generate warning flags for significant disagreements.
    """
    flags = []
    
    dimension_names = {"E": "Evidence", "A": "Assumptions", "S": "Scenarios", "D": "Decision"}
    
    for dim_key, var_value in variance.items():
        if var_value > 1.0:
            flags.append(
                f"{dimension_names[dim_key]} scores diverged significantly (σ={var_value}) - review recommended"
            )
    
    # Check for specific sub-criterion divergence
    evaluations = [r.evaluation for r in judge_responses if r.success and r.evaluation]
    all_scores = [extract_scores_from_evaluation(e) for e in evaluations]
    
    for dimension in ["evidence", "assumptions", "scenarios", "decision"]:
        all_sub_criteria = set()
        for scores in all_scores:
            if dimension in scores:
                all_sub_criteria.update(scores[dimension].keys())
        
        for sub_criterion in all_sub_criteria:
            values = []
            for scores in all_scores:
                if dimension in scores and sub_criterion in scores[dimension]:
                    values.append(scores[dimension][sub_criterion])
            
            if len(values) >= 2:
                max_diff = max(values) - min(values)
                if max_diff > 1:
                    flags.append(
                        f"{dimension}.{sub_criterion} scores diverged by {max_diff} points"
                    )
    
    return flags


def check_critical_drivers(aggregated_scores: dict) -> tuple[bool, list[str]]:
    """
    Check if all critical drivers meet the ≥4 threshold.
    
    Returns (passes, list of failures)
    """
    failures = []
    
    for dimension, sub_criterion in CRITICAL_DRIVERS:
        score = get_nested_value(aggregated_scores, "scores", dimension, sub_criterion)
        if score is None:
            failures.append(f"{dimension}.{sub_criterion}: missing")
        elif score < 4:
            failures.append(f"{dimension}.{sub_criterion}: {score} (requires ≥4)")
    
    return len(failures) == 0, failures


def calculate_total_score(aggregated_scores: dict) -> float:
    """
    Calculate total score from all sub-criteria (max 80).
    """
    total = 0
    scores = aggregated_scores.get("scores", {})
    
    for dimension in ["evidence", "assumptions", "scenarios", "decision"]:
        dim_scores = scores.get(dimension, {})
        for key, value in dim_scores.items():
            if key != "aggregate" and isinstance(value, (int, float)):
                total += value
    
    return round(total, 1)


def identify_residual_risks(
    aggregated_scores: dict,
    evaluations: list[dict]
) -> list[str]:
    """
    Identify residual risks from the DCT residual risk review.
    """
    risks = []
    scores = aggregated_scores.get("scores", {})
    
    # Check for weak status quo option (decision framing)
    decision_framing = get_nested_value(scores, "decision", "decision_framing")
    if decision_framing and decision_framing < 4:
        risks.append("Status quo or 'do nothing' option may be underdeveloped")
    
    # Check for sensitivity fragility
    sensitivity = get_nested_value(scores, "scenarios", "sensitivity_testing")
    if sensitivity and sensitivity < 3:
        risks.append("Sensitivity analysis shows potential fragility in recommendations")
    
    # Check for weak comparative framing
    tradeoff = get_nested_value(scores, "scenarios", "tradeoff_visibility")
    if tradeoff and tradeoff < 4:
        risks.append("Recommendation lacks strong comparative framing vs alternatives")
    
    # Check for underdeveloped costs/tradeoffs
    limitations = get_nested_value(scores, "assumptions", "limitations")
    if limitations and limitations < 3:
        risks.append("Tradeoffs or costs may be underdeveloped, obscuring execution risk")
    
    return risks


def check_dis_completeness(dis: dict) -> list[str]:
    """
    Check DIS for completeness and flag gaps.
    
    Returns list of gap warnings.
    """
    gaps = []
    required_fields = ["recommendation", "what_we_did", "what_we_tested", "risks_acknowledged", "why_this_path"]
    
    for field in required_fields:
        value = dis.get(field, "")
        if not value:
            gaps.append(f"DIS.{field}: Field is empty")
        elif isinstance(value, str) and value.startswith("[GAP]"):
            gaps.append(f"DIS.{field}: {value}")
        elif isinstance(value, str) and len(value) < 20:
            gaps.append(f"DIS.{field}: Field appears incomplete (very short)")
    
    return gaps


def aggregate_dis(evaluations: list[dict]) -> dict:
    """
    Aggregate DIS statements from multiple evaluations.
    Uses the first complete DIS found, with gap flagging.
    """
    # Try to find a complete DIS first
    for evaluation in evaluations:
        dis = get_nested_value(evaluation, "response_evaluation", "dis")
        if dis and all(dis.get(k) for k in ["recommendation", "what_we_did", "what_we_tested", "risks_acknowledged", "why_this_path"]):
            return dis
    
    # Fallback: construct from available pieces, preserving gap markers
    dis = {
        "recommendation": "",
        "what_we_did": "",
        "what_we_tested": "",
        "risks_acknowledged": "",
        "why_this_path": ""
    }
    
    for evaluation in evaluations:
        eval_dis = get_nested_value(evaluation, "response_evaluation", "dis") or {}
        for key in dis:
            if not dis[key] and eval_dis.get(key):
                dis[key] = eval_dis[key]
    
    # Mark any still-empty fields as gaps
    for key in dis:
        if not dis[key]:
            dis[key] = f"[GAP]: No {key.replace('_', ' ')} could be extracted from any judge response"
    
    return dis


def aggregate_request_evaluation(evaluations: list[dict]) -> dict:
    """
    Aggregate request evaluations from multiple judges.
    """
    scores = {
        "context_completeness": [],
        "evidence_availability": [],
        "decision_framing_quality": []
    }
    rationales = []
    
    for evaluation in evaluations:
        req_eval = evaluation.get("request_evaluation", {})
        for key in scores:
            if key in req_eval and isinstance(req_eval[key], (int, float)):
                scores[key].append(req_eval[key])
        if req_eval.get("rationale"):
            rationales.append(req_eval["rationale"])
    
    result = {}
    for key, values in scores.items():
        if values:
            result[key] = round(statistics.mean(values), 1)
        else:
            result[key] = 0
    
    result["rationale"] = rationales[0] if rationales else "No rationale provided"
    
    return result


def build_aggregated_evaluation(
    source_id: str,
    evaluated_at: str,
    judge_responses: list[JudgeResponse],
    extracted_data: Optional[dict] = None
) -> dict:
    """
    Build the final aggregated evaluation from all judge responses.
    
    Args:
        source_id: Identifier for the evaluation
        evaluated_at: ISO timestamp
        judge_responses: List of judge responses
        extracted_data: Optional source data for evidence validation
    """
    # Determine consensus state
    consensus_state, consensus_details = determine_consensus_state(judge_responses)
    
    # Get successful evaluations
    evaluations = [r.evaluation for r in judge_responses if r.success and r.evaluation]
    successful_models = [r.model_id for r in judge_responses if r.success]
    
    if consensus_state == ConsensusState.RED:
        return {
            "source_id": source_id,
            "evaluated_at": evaluated_at,
            "consensus_state": consensus_state.value,
            "consensus_details": consensus_details,
            "error": "Insufficient valid evaluations (RED consensus state - requires ≥2 judges)",
            "judge_responses": [
                {
                    "model": r.model_alias,
                    "success": r.success,
                    "error": r.error,
                    "repair_attempted": getattr(r, 'repair_attempted', False)
                }
                for r in judge_responses
            ]
        }
    
    # Aggregate scores
    aggregated = aggregate_scores(judge_responses)
    
    # Calculate metrics
    total_score = calculate_total_score(aggregated)
    agreement_rate = calculate_agreement_rate(judge_responses)
    flags = generate_flags(judge_responses, aggregated.get("variance", {}))
    critical_met, critical_failures = check_critical_drivers(aggregated)
    residual_risks = identify_residual_risks(aggregated, evaluations)
    
    # Build EASD scores with evidence from first evaluation
    easd_scores = build_easd_scores_with_evidence(aggregated, evaluations)
    
    # Aggregate DIS and check completeness
    dis = aggregate_dis(evaluations)
    dis_gaps = check_dis_completeness(dis)
    
    # Validate evidence pointers if source data provided
    evidence_validation = []
    if extracted_data:
        for evaluation in evaluations:
            evidence_validation = validate_evidence_pointers(evaluation, extracted_data)
            if evidence_validation:
                break  # Use first non-empty validation result
    
    # Count evidence issues by type
    hallucinated_evidence = [e for e in evidence_validation if e.get("type") == "quote_not_found"]
    
    # Add flag if hallucinated evidence found
    if hallucinated_evidence:
        flags.append(f"Evidence validation found {len(hallucinated_evidence)} quote(s) not matching source")
    
    # Add flag if DIS has gaps
    if dis_gaps:
        flags.append(f"DIS has {len(dis_gaps)} incomplete or missing field(s)")
    
    return {
        "source_id": source_id,
        "evaluated_at": evaluated_at,
        "consensus_state": consensus_state.value,
        "consensus_details": consensus_details,
        "request_evaluation": aggregate_request_evaluation(evaluations),
        "response_evaluation": {
            "dis": dis,
            "dis_gaps": dis_gaps,
            "easd_scores": easd_scores
        },
        "judge_consensus": {
            "models_used": successful_models,
            "consensus_state": consensus_state.value,
            "agreement_rate": agreement_rate,
            "score_variance": aggregated.get("variance", {}),
            "flags": flags
        },
        "evidence_validation": {
            "issues_found": len(evidence_validation),
            "hallucinated_quotes": len(hallucinated_evidence),
            "issues": evidence_validation[:10]  # Limit to first 10 issues
        },
        "dct_status": {
            "passes_threshold": total_score >= DCT_REQUIRED_SCORE and critical_met,
            "total_score": total_score,
            "required_score": DCT_REQUIRED_SCORE,
            "critical_drivers_met": critical_met,
            "critical_driver_failures": critical_failures if not critical_met else [],
            "residual_risks": residual_risks
        }
    }


def build_easd_scores_with_evidence(aggregated: dict, evaluations: list[dict]) -> dict:
    """
    Build EASD scores structure with evidence from evaluations.
    """
    scores = aggregated.get("scores", {})
    
    # Get evidence from first evaluation
    first_easd = get_nested_value(evaluations[0], "response_evaluation", "easd_scores") if evaluations else {}
    
    result = {}
    
    for dimension in ["evidence", "assumptions", "scenarios", "decision"]:
        dim_scores = scores.get(dimension, {})
        first_dim = first_easd.get(dimension, {}) if first_easd else {}
        
        result[dimension] = {}
        
        for sub_criterion, score in dim_scores.items():
            if sub_criterion == "aggregate":
                result[dimension]["aggregate"] = score
            else:
                # Get evidence and rationale from first evaluation
                first_sub = first_dim.get(sub_criterion, {})
                result[dimension][sub_criterion] = {
                    "score": score,
                    "rationale": first_sub.get("rationale", ""),
                    "score_gap_reason": first_sub.get("score_gap_reason", ""),
                    "evidence": first_sub.get("evidence", [])
                }
        
        # Ensure aggregate exists
        if "aggregate" not in result[dimension]:
            sub_scores = [v["score"] for k, v in result[dimension].items() 
                         if k != "aggregate" and isinstance(v, dict)]
            if sub_scores:
                result[dimension]["aggregate"] = round(statistics.mean(sub_scores), 2)
    
    # Calculate totals
    total = 0
    for dimension in result.values():
        for key, value in dimension.items():
            if key != "aggregate" and isinstance(value, dict):
                total += value.get("score", 0)
    
    result["total_score"] = round(total, 1)
    result["max_score"] = DCT_MAX_SCORE
    result["confidence_level"] = round(total / DCT_MAX_SCORE, 3)
    
    return result


def compute_repeatability_drift(
    run1_responses: list[JudgeResponse],
    run2_responses: list[JudgeResponse]
) -> dict:
    """
    Compare two runs of the same evaluation to detect score drift.
    
    High drift indicates prompt or rubric instability.
    
    Returns:
        Dict with per-dimension drift and overall stability assessment
    """
    # Extract scores from both runs
    scores1 = aggregate_scores(run1_responses)
    scores2 = aggregate_scores(run2_responses)
    
    if not scores1 or not scores2:
        return {"error": "Could not extract scores from one or both runs"}
    
    drift = {}
    dimension_map = {"evidence": "E", "assumptions": "A", "scenarios": "S", "decision": "D"}
    
    for dimension in ["evidence", "assumptions", "scenarios", "decision"]:
        dim_key = dimension_map[dimension]
        
        agg1 = scores1.get("scores", {}).get(dimension, {}).get("aggregate", 0)
        agg2 = scores2.get("scores", {}).get(dimension, {}).get("aggregate", 0)
        
        drift[dim_key] = abs(agg1 - agg2)
    
    # Calculate total score drift
    total1 = calculate_total_score(scores1)
    total2 = calculate_total_score(scores2)
    total_drift = abs(total1 - total2)
    
    # Assess stability
    max_drift = max(drift.values())
    stability = "stable" if max_drift <= 0.5 else "moderate" if max_drift <= 1.0 else "unstable"
    
    return {
        "dimension_drift": drift,
        "total_score_drift": total_drift,
        "run1_total": total1,
        "run2_total": total2,
        "stability_assessment": stability,
        "recommendation": "Prompt/rubric may need refinement" if stability == "unstable" else None
    }
