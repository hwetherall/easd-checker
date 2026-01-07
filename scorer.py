#!/usr/bin/env python3
"""
EASD Scorer - Multi-model evaluation of AI log quality

Evaluates extracted AI logs against the EASD framework using
consensus scoring from multiple judge models via OpenRouter.

Key features:
- Multi-judge consensus with Green/Amber/Red states
- Automatic JSON repair for malformed responses
- Evidence pointer validation
- Repeatability checking for rubric stability
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lib.openrouter import MODELS, call_judges_sync, JudgeResponse
from lib.aggregator import (
    build_aggregated_evaluation,
    ConsensusState,
    compute_repeatability_drift
)


def load_prompt_template() -> str:
    """Load the judge prompt template."""
    prompt_path = Path(__file__).parent / "prompts" / "judge_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Judge prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def build_prompt(template: str, extracted: dict, use_summarized: bool = True) -> str:
    """
    Build the evaluation prompt from template and extracted data.
    
    Args:
        template: The prompt template with placeholders
        extracted: The extracted JSON data
        use_summarized: If True, prefer summarized documents over full documents
    
    Returns:
        The filled prompt string
    """
    request = extracted.get("request", {})
    response = extracted.get("response", {})
    
    # Choose document source based on flag
    if use_summarized and request.get("summarized_documents"):
        source_docs = request.get("summarized_documents", "Not provided")
    else:
        source_docs = request.get("source_documents", "Not provided")
    
    # Fill in template variables
    prompt = template.replace("{{project_name}}", request.get("project_name", "Unknown Project"))
    prompt = prompt.replace("{{chapter_name}}", request.get("chapter_name", "Unknown Chapter"))
    prompt = prompt.replace("{{decision_context}}", request.get("decision_context", "Not provided"))
    prompt = prompt.replace("{{project_instructions}}", request.get("project_instructions", "Not provided"))
    prompt = prompt.replace("{{source_documents}}", source_docs)
    prompt = prompt.replace("{{other_chapters}}", request.get("other_chapters", "Not provided"))
    prompt = prompt.replace("{{prereq_chapters}}", request.get("prereq_chapters", "Not provided"))
    prompt = prompt.replace("{{response_content}}", response.get("content", "No response content"))
    
    return prompt


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


def get_consensus_indicator(state: str) -> str:
    """Get colored indicator for consensus state."""
    indicators = {
        "green": "🟢 GREEN",
        "amber": "🟡 AMBER",
        "red": "🔴 RED"
    }
    return indicators.get(state, "⚪ UNKNOWN")


def print_evaluation_summary(evaluation: dict, verbose: bool = False):
    """Print a summary of the evaluation results."""
    print("\n" + "═" * 70)
    print(f"EASD Evaluation: {evaluation.get('source_id', 'Unknown')}")
    print("═" * 70)
    
    # Consensus State (NEW - prominent display)
    consensus_state = evaluation.get("consensus_state", evaluation.get("judge_consensus", {}).get("consensus_state", "unknown"))
    consensus_details = evaluation.get("consensus_details", {})
    print(f"\nConsensus State: {get_consensus_indicator(consensus_state)}")
    print(f"  Valid Judges: {consensus_details.get('valid_outputs', 'N/A')}/{consensus_details.get('total_judges', 'N/A')}")
    
    # Show failed judges if any
    failed = consensus_details.get("failed_judges", [])
    if failed:
        print("  Failed Judges:")
        for f in failed:
            repair_note = " (repair attempted)" if f.get("repair_attempted") else ""
            print(f"    ✗ {f['model']}: {f['error']}{repair_note}")
    
    # Check for RED state
    if consensus_state == "red":
        print("\n  ⚠ INSUFFICIENT CONSENSUS - Results unreliable")
        if evaluation.get("error"):
            print(f"  Error: {evaluation['error']}")
        print("═" * 70)
        return
    
    # DCT Status
    dct = evaluation.get("dct_status", {})
    status = "✓ PASSES" if dct.get("passes_threshold") else "✗ FAILS"
    print(f"\nDecision Confidence Threshold: {status}")
    print(f"  Score: {dct.get('total_score', 0):.1f} / {dct.get('required_score', 68)} required")
    print(f"  Critical Drivers Met: {'Yes' if dct.get('critical_drivers_met') else 'No'}")
    
    if dct.get("critical_driver_failures"):
        print("  Failed drivers:")
        for failure in dct["critical_driver_failures"]:
            print(f"    - {failure}")
    
    # EASD Scores
    easd = evaluation.get("response_evaluation", {}).get("easd_scores", {})
    print("\n" + "─" * 70)
    print("EASD Scores (1-5 scale)")
    print("─" * 70)
    
    dimensions = [
        ("Evidence (E)", "evidence"),
        ("Assumptions (A)", "assumptions"),
        ("Scenarios (S)", "scenarios"),
        ("Decision (D)", "decision")
    ]
    
    for label, key in dimensions:
        dim = easd.get(key, {})
        agg = dim.get("aggregate", 0)
        bar = "█" * int(agg) + "░" * (5 - int(agg))
        print(f"  {label:<20} {bar} {agg:.1f}")
        
        if verbose:
            for sub_key, sub_value in dim.items():
                if sub_key != "aggregate" and isinstance(sub_value, dict):
                    score = sub_value.get("score", 0)
                    gap_reason = sub_value.get("score_gap_reason", "")
                    print(f"    ├─ {sub_key:<18} {score}")
                    if gap_reason and verbose:
                        print(f"    │  └─ Gap: {gap_reason[:60]}...")
    
    # Evidence Validation (NEW)
    evidence_val = evaluation.get("evidence_validation", {})
    if evidence_val.get("issues_found", 0) > 0:
        print("\n" + "─" * 70)
        print("Evidence Validation")
        print("─" * 70)
        hallucinated = evidence_val.get("hallucinated_quotes", 0)
        total_issues = evidence_val.get("issues_found", 0)
        
        if hallucinated > 0:
            print(f"  ⚠ {hallucinated} quote(s) not found in source (possible hallucination)")
        print(f"  Total issues: {total_issues}")
        
        if verbose and evidence_val.get("issues"):
            for issue in evidence_val["issues"][:5]:  # Show first 5
                print(f"    • {issue.get('location')}: {issue.get('message')}")
    
    # Judge Consensus
    consensus = evaluation.get("judge_consensus", {})
    print("\n" + "─" * 70)
    print("Judge Consensus")
    print("─" * 70)
    print(f"  Models: {', '.join(consensus.get('models_used', []))}")
    print(f"  Agreement Rate: {consensus.get('agreement_rate', 0):.0%}")
    
    variance = consensus.get("score_variance", {})
    if variance:
        var_str = ", ".join(f"{k}={v:.2f}" for k, v in variance.items())
        print(f"  Variance (σ): {var_str}")
    
    flags = consensus.get("flags", [])
    if flags:
        print("  Flags:")
        for flag in flags:
            print(f"    ⚠ {flag}")
    
    # DIS Gaps (NEW)
    dis_gaps = evaluation.get("response_evaluation", {}).get("dis_gaps", [])
    if dis_gaps:
        print("\n" + "─" * 70)
        print("DIS Completeness Issues")
        print("─" * 70)
        for gap in dis_gaps:
            print(f"  ⚠ {gap}")
    
    # Residual Risks
    risks = dct.get("residual_risks", [])
    if risks:
        print("\n" + "─" * 70)
        print("Residual Risks")
        print("─" * 70)
        for risk in risks:
            print(f"  • {risk}")
    
    # DIS Summary
    if verbose:
        dis = evaluation.get("response_evaluation", {}).get("dis", {})
        if dis.get("recommendation"):
            print("\n" + "─" * 70)
            print("Decision Integrity Statement")
            print("─" * 70)
            rec = dis.get('recommendation', '')
            print(f"  Recommendation: {rec[:200]}{'...' if len(rec) > 200 else ''}")
            
            why = dis.get('why_this_path', '')
            if why and not why.startswith("[GAP]"):
                print(f"  Why This Path: {why[:150]}{'...' if len(why) > 150 else ''}")
    
    print("\n" + "═" * 70)


def process_file(
    input_path: Path,
    output_dir: Path,
    models: Optional[list[str]] = None,
    verbose: bool = False,
    dry_run: bool = False,
    use_summarized: bool = True,
    repeatability_check: bool = False
) -> Optional[dict]:
    """
    Process a single extracted JSON file.
    
    Args:
        input_path: Path to extracted JSON file
        output_dir: Directory for output files
        models: List of model aliases to use
        verbose: Show detailed output
        dry_run: Preview without calling models
        use_summarized: Use summarized documents (default: True)
        repeatability_check: Run twice and compare for drift
    
    Returns:
        The evaluation dict or None on failure.
    """
    # Load extracted data
    try:
        extracted = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_path}: {e}", file=sys.stderr)
        return None
    
    source_id = extracted.get("source_id", input_path.stem)
    
    if verbose:
        print(f"\nProcessing: {source_id}")
        print(f"  Input: {input_path}")
        print(f"  Using: {'summarized' if use_summarized else 'full'} documents")
    
    # Build prompt
    template = load_prompt_template()
    prompt = build_prompt(template, extracted, use_summarized=use_summarized)
    
    if verbose:
        prompt_size = len(prompt.encode("utf-8"))
        print(f"  Prompt size: {format_size(prompt_size)}")
    
    if dry_run:
        print(f"  [DRY RUN] Would call models: {', '.join(models or list(MODELS.keys()))}")
        return None
    
    # Call judge models
    if verbose:
        print(f"  Calling judges: {', '.join(models or list(MODELS.keys()))}...")
    
    responses = call_judges_sync(prompt, models)
    
    # Report individual model results
    for resp in responses:
        status = "✓" if resp.success else "✗"
        if verbose:
            if resp.success:
                repair_note = " (JSON repaired)" if resp.repair_attempted else ""
                print(f"    {status} {resp.model_alias}: success{repair_note}")
            else:
                repair_note = " (repair failed)" if resp.repair_attempted else ""
                print(f"    {status} {resp.model_alias}: {resp.error}{repair_note}")
    
    # Repeatability check (optional)
    repeatability_result = None
    if repeatability_check:
        if verbose:
            print(f"  Running repeatability check (second call)...")
        responses2 = call_judges_sync(prompt, models)
        repeatability_result = compute_repeatability_drift(responses, responses2)
        
        if verbose:
            stability = repeatability_result.get("stability_assessment", "unknown")
            total_drift = repeatability_result.get("total_score_drift", 0)
            print(f"    Stability: {stability} (drift: {total_drift:.1f} points)")
    
    # Aggregate results with evidence validation
    evaluated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    evaluation = build_aggregated_evaluation(
        source_id, 
        evaluated_at, 
        responses,
        extracted_data=extracted  # Pass for evidence validation
    )
    
    # Add repeatability result if performed
    if repeatability_result:
        evaluation["repeatability_check"] = repeatability_result
    
    # Save output
    output_path = output_dir / f"{source_id}-evaluation.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"  → Wrote {output_path}")
    
    return evaluation


def find_extracted_files(input_path: Path) -> list[Path]:
    """Find all extracted JSON files in a directory."""
    if input_path.is_file():
        return [input_path]
    
    return sorted(input_path.glob("*-extracted.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate extracted AI logs against the EASD framework using multi-model consensus scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Consensus States:
  🟢 GREEN  - 3/3 judges returned valid output (high confidence)
  🟡 AMBER  - 2/3 judges valid (acceptable, one failure noted)
  🔴 RED    - <2 judges valid (unreliable, review required)

Examples:
  # Single file evaluation (uses summarized docs by default)
  python scorer.py --input extracted/ai-log-259643-extracted.json

  # Use full document context (larger prompt)
  python scorer.py --input extracted/ai-log-259643-extracted.json --full-context

  # Single model (faster, no consensus)
  python scorer.py --input extracted/ai-log-259643-extracted.json --model gemini

  # With repeatability check (detects prompt/rubric instability)
  python scorer.py --input extracted/ai-log-259643-extracted.json --repeatability-check

  # Batch mode (all files in directory)
  python scorer.py --input extracted/ --output evaluations/

  # Dry run (show what would be done)
  python scorer.py --input extracted/ --dry-run --verbose
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input file or directory containing extracted JSON files"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("evaluations"),
        help="Output directory for evaluation results (default: ./evaluations)"
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        action="append",
        help="Specific model(s) to use (can specify multiple). Default: all models"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress and sub-criterion scores"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without calling models"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing evaluation summary"
    )
    
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="Use full source documents instead of summarized versions (larger prompt)"
    )
    
    parser.add_argument(
        "--repeatability-check",
        action="store_true",
        help="Run evaluation twice and report score drift (tests prompt stability)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Find files to process
    files = find_extracted_files(args.input)
    
    if not files:
        print(f"Error: No extracted JSON files found in {args.input}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"Found {len(files)} file(s) to evaluate")
        print(f"Document mode: {'full context' if args.full_context else 'summarized (default)'}")
    
    # Process files
    evaluations = []
    
    for file_path in files:
        evaluation = process_file(
            file_path,
            args.output,
            models=args.model,
            verbose=args.verbose,
            dry_run=args.dry_run,
            use_summarized=not args.full_context,
            repeatability_check=args.repeatability_check
        )
        
        if evaluation:
            evaluations.append(evaluation)
            
            if not args.no_summary:
                print_evaluation_summary(evaluation, verbose=args.verbose)
    
    # Print batch summary
    if len(evaluations) > 1:
        print("\n" + "═" * 70)
        print("Batch Summary")
        print("═" * 70)
        
        # Count by consensus state
        green = sum(1 for e in evaluations if e.get("consensus_state") == "green")
        amber = sum(1 for e in evaluations if e.get("consensus_state") == "amber")
        red = sum(1 for e in evaluations if e.get("consensus_state") == "red")
        
        print(f"  Consensus: 🟢 {green} | 🟡 {amber} | 🔴 {red}")
        
        passed = sum(1 for e in evaluations if e.get("dct_status", {}).get("passes_threshold"))
        total = len(evaluations)
        
        print(f"  Files evaluated: {total}")
        print(f"  Passed DCT: {passed}/{total} ({100*passed/total:.0f}%)")
        
        # Average score (only from non-red evaluations)
        valid_scores = [
            e.get("dct_status", {}).get("total_score", 0) 
            for e in evaluations 
            if e.get("consensus_state") != "red"
        ]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            print(f"  Average score: {avg_score:.1f}/80")
        
        # Evidence validation summary
        total_hallucinated = sum(
            e.get("evidence_validation", {}).get("hallucinated_quotes", 0)
            for e in evaluations
        )
        if total_hallucinated > 0:
            print(f"  ⚠ Total hallucinated quotes: {total_hallucinated}")
        
        print("═" * 70)
    
    if not evaluations and not args.dry_run:
        print("\nNo evaluations completed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
