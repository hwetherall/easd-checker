#!/usr/bin/env python3
"""
EASD Synthesizer (Stage 3) - Meta-Analysis of Judge Evaluations

Synthesizes multiple judge evaluations into a single authoritative
assessment using Claude Opus 4.5 as the meta-analyst.

Key features:
- Cross-evaluation analysis with agreement/disagreement classification
- DIS synthesis with conflict resolution
- Final EASD scoring with confidence adjustments
- Meta-insights generation for actionable improvements
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lib.openrouter import call_synthesis_sync, JudgeResponse, SYNTHESIS_MODEL


def load_synthesis_prompt_template() -> str:
    """Load the synthesis prompt template."""
    prompt_path = Path(__file__).parent / "prompts" / "synthesis_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Synthesis prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def extract_judge_evaluations(evaluation: dict) -> list[dict]:
    """
    Extract individual judge evaluations from the aggregated evaluation.
    
    Since Stage 2 aggregates scores, we reconstruct judge-level data
    from the available information.
    """
    judge_consensus = evaluation.get("judge_consensus", {})
    models_used = judge_consensus.get("models_used", [])
    
    # The current aggregator merges scores, so we work with what's available
    # In production, you might want to store raw judge responses separately
    response_eval = evaluation.get("response_evaluation", {})
    easd_scores = response_eval.get("easd_scores", {})
    dis = response_eval.get("dis", {})
    
    # Create a representation of the aggregated evaluation
    # In a full implementation, this would be multiple separate evaluations
    judge_data = []
    
    for model in models_used:
        judge_data.append({
            "model": model,
            "dis": dis,
            "easd_scores": easd_scores
        })
    
    return judge_data


def format_scores_for_prompt(easd_scores: dict) -> str:
    """Format EASD scores as readable text for the prompt."""
    lines = []
    
    for dimension in ["evidence", "assumptions", "scenarios", "decision"]:
        dim_data = easd_scores.get(dimension, {})
        agg = dim_data.get("aggregate", 0)
        lines.append(f"\n**{dimension.upper()}** (aggregate: {agg})")
        
        for key, value in dim_data.items():
            if key == "aggregate":
                continue
            if isinstance(value, dict):
                score = value.get("score", "N/A")
                rationale = value.get("rationale", "No rationale provided")
                gap_reason = value.get("score_gap_reason", "")
                
                lines.append(f"  - {key}: {score}")
                lines.append(f"    Rationale: {rationale}")
                if gap_reason:
                    lines.append(f"    Gap reason: {gap_reason}")
    
    total = easd_scores.get("total_score", 0)
    confidence = easd_scores.get("confidence_level", 0)
    lines.append(f"\nTotal Score: {total}/80")
    lines.append(f"Confidence Level: {confidence:.1%}")
    
    return "\n".join(lines)


def format_dis_for_prompt(dis: dict) -> str:
    """Format DIS as readable text for the prompt."""
    lines = []
    
    fields = [
        ("recommendation", "Recommendation"),
        ("what_we_did", "What We Did"),
        ("what_we_tested", "What We Tested"),
        ("risks_acknowledged", "Risks Acknowledged"),
        ("why_this_path", "Why This Path")
    ]
    
    for field, label in fields:
        value = dis.get(field, "[Not provided]")
        lines.append(f"**{label}**: {value}")
    
    return "\n".join(lines)


def format_judge_evaluations(judge_data: list[dict]) -> str:
    """Format individual judge evaluations for the prompt."""
    if not judge_data:
        return "No individual judge evaluations available."
    
    sections = []
    
    for i, judge in enumerate(judge_data, 1):
        model = judge.get("model", f"Judge {i}")
        lines = [f"#### Judge: {model}"]
        
        # DIS
        dis = judge.get("dis", {})
        lines.append("\n**Decision Integrity Statement:**")
        lines.append(format_dis_for_prompt(dis))
        
        # Scores
        easd = judge.get("easd_scores", {})
        lines.append("\n**EASD Scores:**")
        lines.append(format_scores_for_prompt(easd))
        
        sections.append("\n".join(lines))
    
    return "\n\n---\n\n".join(sections)


def build_synthesis_prompt(template: str, evaluation: dict) -> str:
    """
    Build the synthesis prompt from template and evaluation data.
    
    Args:
        template: The synthesis prompt template
        evaluation: The Stage 2 evaluation JSON
    
    Returns:
        The filled prompt string
    """
    # Extract metadata
    source_id = evaluation.get("source_id", "unknown")
    evaluated_at = evaluation.get("evaluated_at", "unknown")
    consensus_state = evaluation.get("consensus_state", "unknown")
    
    judge_consensus = evaluation.get("judge_consensus", {})
    models_used = ", ".join(judge_consensus.get("models_used", []))
    agreement_rate = judge_consensus.get("agreement_rate", 0)
    score_variance = json.dumps(judge_consensus.get("score_variance", {}), indent=2)
    consensus_flags = "\n".join(f"- {f}" for f in judge_consensus.get("flags", [])) or "No flags"
    
    # Format scores and DIS
    response_eval = evaluation.get("response_evaluation", {})
    easd_scores = response_eval.get("easd_scores", {})
    dis = response_eval.get("dis", {})
    
    aggregated_scores = format_scores_for_prompt(easd_scores)
    aggregated_dis = format_dis_for_prompt(dis)
    
    # Evidence validation
    evidence_val = evaluation.get("evidence_validation", {})
    evidence_validation = json.dumps(evidence_val, indent=2)
    
    # DCT status
    dct_status = json.dumps(evaluation.get("dct_status", {}), indent=2)
    
    # Individual judge evaluations
    judge_data = extract_judge_evaluations(evaluation)
    judge_evaluations = format_judge_evaluations(judge_data)
    
    # Fill template
    prompt = template.replace("{{source_id}}", source_id)
    prompt = prompt.replace("{{evaluated_at}}", evaluated_at)
    prompt = prompt.replace("{{consensus_state}}", consensus_state)
    prompt = prompt.replace("{{models_used}}", models_used)
    prompt = prompt.replace("{{agreement_rate}}", f"{agreement_rate:.0%}")
    prompt = prompt.replace("{{aggregated_scores}}", aggregated_scores)
    prompt = prompt.replace("{{aggregated_dis}}", aggregated_dis)
    prompt = prompt.replace("{{score_variance}}", score_variance)
    prompt = prompt.replace("{{consensus_flags}}", consensus_flags)
    prompt = prompt.replace("{{evidence_validation}}", evidence_validation)
    prompt = prompt.replace("{{dct_status}}", dct_status)
    prompt = prompt.replace("{{judge_evaluations}}", judge_evaluations)
    
    return prompt


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


def get_confidence_indicator(confidence: str) -> str:
    """Get colored indicator for confidence level."""
    indicators = {
        "high": "🟢 HIGH",
        "medium": "🟡 MEDIUM",
        "low": "🔴 LOW"
    }
    return indicators.get(confidence, "⚪ UNKNOWN")


def print_synthesis_summary(synthesis: dict, verbose: bool = False):
    """Print a summary of the synthesis results."""
    print("\n" + "═" * 70)
    print(f"EASD Meta-Synthesis: {synthesis.get('source_id', 'Unknown')}")
    print("═" * 70)
    
    # Synthesis quality
    metadata = synthesis.get("synthesis_metadata", {})
    quality = metadata.get("synthesis_quality", {})
    overall_confidence = quality.get("overall_confidence", "unknown")
    
    print(f"\nSynthesis Quality: {get_confidence_indicator(overall_confidence)}")
    print(f"  Judges Synthesized: {', '.join(metadata.get('judges_synthesized', []))}")
    print(f"  Original Consensus: {metadata.get('original_consensus_state', 'N/A')}")
    
    # Uncertainty flags
    flags = quality.get("uncertainty_flags", [])
    if flags:
        print("  Uncertainty Flags:")
        for flag in flags:
            print(f"    ⚠ {flag}")
    
    # Synthesized DIS
    dis = synthesis.get("synthesized_dis", {})
    rec_confidence = dis.get("recommendation_confidence", "unknown")
    print("\n" + "─" * 70)
    print("Decision Integrity Statement")
    print("─" * 70)
    print(f"  Recommendation Confidence: {get_confidence_indicator(rec_confidence)}")
    
    rec = dis.get("recommendation", "")
    print(f"  Recommendation: {rec[:150]}{'...' if len(rec) > 150 else ''}")
    
    unresolved = dis.get("unresolved_risks", [])
    if unresolved:
        print("  Unresolved Risks:")
        for risk in unresolved[:3]:
            print(f"    • {risk}")
    
    conflicts = dis.get("recommendation_conflicts", [])
    if conflicts:
        print("  ⚠ Recommendation Conflicts:")
        for conflict in conflicts:
            print(f"    - {conflict.get('judge')}: {conflict.get('alternative_recommendation', '')[:60]}...")
    
    # Final EASD Scores
    final_scores = synthesis.get("final_easd_scores", {})
    print("\n" + "─" * 70)
    print("Final EASD Scores (1-5 scale)")
    print("─" * 70)
    
    dimensions = [
        ("Evidence (E)", "evidence"),
        ("Assumptions (A)", "assumptions"),
        ("Scenarios (S)", "scenarios"),
        ("Decision (D)", "decision")
    ]
    
    for label, key in dimensions:
        dim = final_scores.get(key, {})
        agg = dim.get("aggregate", 0)
        bar = "█" * int(agg) + "░" * (5 - int(agg))
        print(f"  {label:<20} {bar} {agg:.1f}")
        
        if verbose:
            for sub_key, sub_value in dim.items():
                if sub_key != "aggregate" and isinstance(sub_value, dict):
                    score = sub_value.get("score", 0)
                    adjusted = " (adjusted)" if sub_value.get("confidence_adjusted") else ""
                    print(f"    ├─ {sub_key:<18} {score}{adjusted}")
    
    # DCT Status
    passes = final_scores.get("passes_dct", False)
    total = final_scores.get("total_score", 0)
    status = "✓ PASSES" if passes else "✗ FAILS"
    print(f"\n  Decision Confidence Threshold: {status}")
    print(f"  Total Score: {total:.1f} / 80")
    
    critical = final_scores.get("critical_driver_status", {})
    if not critical.get("all_met"):
        failures = critical.get("failures", [])
        if failures:
            print("  Critical Driver Failures:")
            for f in failures:
                print(f"    ✗ {f}")
    
    # Meta-Insights
    insights = synthesis.get("meta_insights", {})
    print("\n" + "─" * 70)
    print("Meta-Insights")
    print("─" * 70)
    
    weakness = insights.get("biggest_analytical_weakness", "Not identified")
    print(f"  Biggest Weakness: {weakness[:100]}{'...' if len(weakness) > 100 else ''}")
    
    improvement = insights.get("confidence_improvement_action", "Not identified")
    print(f"  To Improve Confidence: {improvement[:100]}{'...' if len(improvement) > 100 else ''}")
    
    risk = insights.get("underweighted_risk", "Not identified")
    print(f"  Underweighted Risk: {risk[:100]}{'...' if len(risk) > 100 else ''}")
    
    print("\n" + "═" * 70)


def process_file(
    input_path: Path,
    output_dir: Path,
    verbose: bool = False,
    dry_run: bool = False
) -> Optional[dict]:
    """
    Process a single evaluation JSON file for synthesis.
    
    Args:
        input_path: Path to Stage 2 evaluation JSON file
        output_dir: Directory for output files
        verbose: Show detailed output
        dry_run: Preview without calling model
    
    Returns:
        The synthesis dict or None on failure.
    """
    # Load evaluation data
    try:
        evaluation = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_path}: {e}", file=sys.stderr)
        return None
    
    source_id = evaluation.get("source_id", input_path.stem.replace("-evaluation", ""))
    
    # Check consensus state
    consensus_state = evaluation.get("consensus_state", "unknown")
    if consensus_state == "red":
        print(f"Warning: Skipping {source_id} - RED consensus state (unreliable evaluation)", file=sys.stderr)
        return None
    
    if verbose:
        print(f"\nProcessing: {source_id}")
        print(f"  Input: {input_path}")
        print(f"  Original consensus: {consensus_state}")
    
    # Build prompt
    template = load_synthesis_prompt_template()
    prompt = build_synthesis_prompt(template, evaluation)
    
    if verbose:
        prompt_size = len(prompt.encode("utf-8"))
        print(f"  Prompt size: {format_size(prompt_size)}")
    
    if dry_run:
        print(f"  [DRY RUN] Would call synthesis model: {list(SYNTHESIS_MODEL.keys())[0]}")
        return None
    
    # Call synthesis model
    if verbose:
        print(f"  Calling Claude Opus for synthesis...")
    
    response = call_synthesis_sync(prompt)
    
    if not response.success:
        repair_note = " (repair attempted)" if response.repair_attempted else ""
        print(f"  ✗ Synthesis failed: {response.error}{repair_note}", file=sys.stderr)
        return None
    
    if verbose:
        repair_note = " (JSON repaired)" if response.repair_attempted else ""
        print(f"  ✓ Synthesis complete{repair_note}")
    
    synthesis = response.evaluation
    
    # Ensure required fields are present
    synthesized_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    if "source_id" not in synthesis:
        synthesis["source_id"] = source_id
    if "synthesized_at" not in synthesis:
        synthesis["synthesized_at"] = synthesized_at
    if "synthesis_model" not in synthesis:
        synthesis["synthesis_model"] = response.model_id
    
    # Ensure synthesis_metadata has source info
    if "synthesis_metadata" not in synthesis:
        synthesis["synthesis_metadata"] = {}
    
    metadata = synthesis["synthesis_metadata"]
    if "source_evaluation_id" not in metadata:
        metadata["source_evaluation_id"] = source_id
    if "source_evaluation_timestamp" not in metadata:
        metadata["source_evaluation_timestamp"] = evaluation.get("evaluated_at", "")
    if "original_consensus_state" not in metadata:
        metadata["original_consensus_state"] = consensus_state
    if "judges_synthesized" not in metadata:
        metadata["judges_synthesized"] = evaluation.get("judge_consensus", {}).get("models_used", [])
    
    # Save output
    output_path = output_dir / f"{source_id}-synthesis.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(synthesis, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"  → Wrote {output_path}")
    
    return synthesis


def find_evaluation_files(input_path: Path) -> list[Path]:
    """Find all evaluation JSON files in a directory."""
    if input_path.is_file():
        return [input_path]
    
    return sorted(input_path.glob("*-evaluation.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Synthesize multiple judge evaluations into an authoritative meta-analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file synthesis
  python synthesizer.py --input evaluations/ai-log-259643-evaluation.json

  # Batch mode (all evaluations in directory)
  python synthesizer.py --input evaluations/ --output synthesis/

  # Verbose output with sub-criterion breakdown
  python synthesizer.py --input evaluations/ai-log-259643-evaluation.json --verbose

  # Dry run (preview without calling model)
  python synthesizer.py --input evaluations/ --dry-run --verbose
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input file or directory containing evaluation JSON files"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("synthesis"),
        help="Output directory for synthesis results (default: ./synthesis)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress and sub-criterion scores"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without calling the model"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing synthesis summary"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Find files to process
    files = find_evaluation_files(args.input)
    
    if not files:
        print(f"Error: No evaluation JSON files found in {args.input}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"Found {len(files)} evaluation file(s) to synthesize")
    
    # Process files
    syntheses = []
    
    for file_path in files:
        synthesis = process_file(
            file_path,
            args.output,
            verbose=args.verbose,
            dry_run=args.dry_run
        )
        
        if synthesis:
            syntheses.append(synthesis)
            
            if not args.no_summary:
                print_synthesis_summary(synthesis, verbose=args.verbose)
    
    # Print batch summary
    if len(syntheses) > 1:
        print("\n" + "═" * 70)
        print("Batch Synthesis Summary")
        print("═" * 70)
        
        # Count by confidence
        high = sum(1 for s in syntheses 
                   if s.get("synthesis_metadata", {}).get("synthesis_quality", {}).get("overall_confidence") == "high")
        medium = sum(1 for s in syntheses 
                     if s.get("synthesis_metadata", {}).get("synthesis_quality", {}).get("overall_confidence") == "medium")
        low = sum(1 for s in syntheses 
                  if s.get("synthesis_metadata", {}).get("synthesis_quality", {}).get("overall_confidence") == "low")
        
        print(f"  Confidence: 🟢 {high} | 🟡 {medium} | 🔴 {low}")
        
        # Count DCT pass/fail
        passed = sum(1 for s in syntheses if s.get("final_easd_scores", {}).get("passes_dct"))
        total = len(syntheses)
        
        print(f"  Files synthesized: {total}")
        print(f"  Passed DCT: {passed}/{total} ({100*passed/total:.0f}%)")
        
        # Average score
        scores = [s.get("final_easd_scores", {}).get("total_score", 0) for s in syntheses]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"  Average score: {avg_score:.1f}/80")
        
        print("═" * 70)
    
    if not syntheses and not args.dry_run:
        print("\nNo syntheses completed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
