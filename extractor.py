#!/usr/bin/env python3
"""
Innovera AI Log Content Extractor

Extracts analyzable content from AI log files for EASD quality analysis.
Based on the extraction plan in JSON_EXTRACTION_PLAN.md
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class ExtractionResult:
    """Holds extraction results with metadata for reporting."""
    def __init__(self, source_id: str):
        self.source_id = source_id
        self.request_data: dict = {}
        self.response_content: Optional[str] = None
        self.original_request_size: int = 0
        self.original_response_size: int = 0
        self.extracted_size: int = 0
        self.warnings: list[str] = []
        self.success: bool = True


def extract_response_content(response_json: dict) -> tuple[Optional[str], bool]:
    """
    Extracts the AI-generated analysis text from response JSON.
    
    Returns:
        Tuple of (content, used_fallback)
    """
    # Primary path: steps[0].content[0].text
    steps = response_json.get('steps')
    if steps and len(steps) > 0:
        content = steps[0].get('content')
        if content and len(content) > 0:
            text = content[0].get('text')
            if text and text.strip():
                return text, False
    
    # Fallback path: messages[0].content[0].text
    messages = response_json.get('messages')
    if messages and len(messages) > 0:
        content = messages[0].get('content')
        if content and len(content) > 0:
            text = content[0].get('text')
            if text and text.strip():
                return text, True
    
    return None, False


def extract_request_content(
    request_json: dict,
    use_summarized: bool = False,
    skip_templates: bool = False
) -> dict:
    """
    Extracts relevant content from request JSON.
    
    Args:
        request_json: The parsed request JSON
        use_summarized: If True, use summarizedVaultDocumentsData instead of full
        skip_templates: If True, exclude template_content from output
    
    Returns:
        Structured dict of extracted content
    """
    result = {}
    
    # Main prompt from messages[0].content
    messages = request_json.get('messages')
    if messages and len(messages) > 0:
        prompt = messages[0].get('content')
        if prompt and str(prompt).strip():
            result['prompt'] = prompt
    
    # Navigate to promptGraphData
    annotations = request_json.get('requestAnnotations', {})
    prompt_graph = annotations.get('promptGraphData', {})
    
    # Context fields
    context = prompt_graph.get('context', {})
    
    project_name = context.get('projectName')
    if project_name and project_name.strip():
        result['project_name'] = project_name
    
    chapter_name = context.get('projectTemplateChapterName')
    if chapter_name and chapter_name.strip():
        result['chapter_name'] = chapter_name
    
    project_instructions = context.get('projectInstructions')
    if project_instructions and project_instructions.strip():
        result['project_instructions'] = project_instructions
    
    decision_context = context.get('projectDecisionContext')
    if decision_context and decision_context.strip():
        result['decision_context'] = decision_context
    
    # Function call results
    func_calls = prompt_graph.get('uniqueFunctionCalls', {})
    
    # Source documents - either full or summarized based on flag
    if use_summarized:
        summarized = func_calls.get('summarizedVaultDocumentsData', {})
        docs_result = summarized.get('result')
        if docs_result and docs_result.strip():
            result['source_documents'] = docs_result
    else:
        vault_data = func_calls.get('vaultDocumentsData', {})
        docs_result = vault_data.get('result')
        if docs_result and docs_result.strip():
            result['source_documents'] = docs_result
        
        # Also include summarized for reference
        summarized = func_calls.get('summarizedVaultDocumentsData', {})
        summarized_result = summarized.get('result')
        if summarized_result and summarized_result.strip():
            result['summarized_documents'] = summarized_result
    
    # Other chapters content
    other_chapters = func_calls.get('otherChaptersContent', {})
    other_result = other_chapters.get('result')
    if other_result and other_result.strip():
        result['other_chapters'] = other_result
    
    # Prerequisite chapters content
    prereq_chapters = func_calls.get('prereqChaptersContent', {})
    prereq_result = prereq_chapters.get('result')
    if prereq_result and prereq_result.strip():
        result['prereq_chapters'] = prereq_result
    
    # Business models
    biz_models = func_calls.get('businessModels', {})
    biz_result = biz_models.get('result')
    if biz_result and biz_result.strip():
        result['business_models'] = biz_result
    
    # Template content from nodes (if not skipped)
    if not skip_templates:
        nodes = prompt_graph.get('nodes', {})
        template_content = []
        
        for node_id, node_data in nodes.items():
            node_result = node_data.get('result', '')
            # Only include if content is substantive (>50 chars)
            if node_result and len(node_result.strip()) > 50:
                template_content.append({
                    'slug': node_data.get('slug', node_id),
                    'content': node_result
                })
        
        if template_content:
            result['template_content'] = template_content
    
    return result


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


def extract_id_from_filename(filename: str) -> Optional[str]:
    """Extract the log ID from a filename like ai-log-259643-requestParams.json"""
    match = re.search(r'ai-log-(\d+)', filename)
    return match.group(1) if match else None


def process_pair(
    request_path: Path,
    response_path: Path,
    use_summarized: bool = False,
    skip_templates: bool = False,
    verbose: bool = False
) -> ExtractionResult:
    """
    Process a request/response file pair.
    
    Returns:
        ExtractionResult with all extracted content and metadata
    """
    source_id = f"ai-log-{extract_id_from_filename(request_path.name)}"
    result = ExtractionResult(source_id)
    
    try:
        # Read and measure original sizes
        request_content = request_path.read_text(encoding='utf-8')
        response_content = response_path.read_text(encoding='utf-8')
        
        result.original_request_size = len(request_content.encode('utf-8'))
        result.original_response_size = len(response_content.encode('utf-8'))
        
        if verbose:
            print(f"  Reading {request_path.name} ({format_size(result.original_request_size)})")
            print(f"  Reading {response_path.name} ({format_size(result.original_response_size)})")
        
        # Parse JSON
        try:
            request_json = json.loads(request_content)
        except json.JSONDecodeError as e:
            result.warnings.append(f"Invalid JSON in request file: {e}")
            result.success = False
            return result
        
        try:
            response_json = json.loads(response_content)
        except json.JSONDecodeError as e:
            result.warnings.append(f"Invalid JSON in response file: {e}")
            result.success = False
            return result
        
        # Extract response content
        response_text, used_fallback = extract_response_content(response_json)
        if response_text:
            result.response_content = response_text
            if used_fallback and verbose:
                print("  ⚠ Used fallback path (messages) for response extraction")
                result.warnings.append("Used fallback extraction path for response")
        else:
            result.warnings.append("Could not extract response content from either path")
            if verbose:
                print("  ⚠ Could not extract response content")
        
        # Extract request content
        result.request_data = extract_request_content(
            request_json,
            use_summarized=use_summarized,
            skip_templates=skip_templates
        )
        
        if verbose:
            fields_found = list(result.request_data.keys())
            print(f"  Extracted request fields: {', '.join(fields_found)}")
        
    except Exception as e:
        result.warnings.append(f"Error processing files: {str(e)}")
        result.success = False
    
    return result


def build_output(result: ExtractionResult) -> dict:
    """Build the final output structure from extraction result."""
    output = {
        "source_id": result.source_id,
        "extracted_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "request": result.request_data,
        "response": {
            "content": result.response_content
        } if result.response_content else {},
        "metadata": {}
    }
    
    # Calculate extracted size
    extracted_json = json.dumps(output, ensure_ascii=False)
    extracted_size = len(extracted_json.encode('utf-8'))
    
    original_total = result.original_request_size + result.original_response_size
    reduction_pct = ((original_total - extracted_size) / original_total * 100) if original_total > 0 else 0
    
    output["metadata"] = {
        "original_request_size_bytes": result.original_request_size,
        "original_response_size_bytes": result.original_response_size,
        "extracted_size_bytes": extracted_size,
        "reduction_percentage": round(reduction_pct, 1)
    }
    
    if result.warnings:
        output["metadata"]["warnings"] = result.warnings
    
    result.extracted_size = extracted_size
    
    return output


def find_file_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Find matching request/response file pairs in a directory."""
    pairs = []
    request_files = list(input_dir.glob('ai-log-*-requestParams.json'))
    
    for req_file in request_files:
        log_id = extract_id_from_filename(req_file.name)
        if log_id:
            resp_file = input_dir / f'ai-log-{log_id}-response.json'
            if resp_file.exists():
                pairs.append((req_file, resp_file))
    
    return sorted(pairs, key=lambda p: p[0].name)


def print_summary_table(results: list[tuple[ExtractionResult, dict]]):
    """Print a summary table of all extractions."""
    print("\n" + "═" * 65)
    print("Extraction Complete")
    print("─" * 65)
    print(f"{'File ID':<20} {'Original':>12} {'Extracted':>12} {'Reduction':>10}")
    print("─" * 65)
    
    total_original = 0
    total_extracted = 0
    
    for result, output in results:
        original = result.original_request_size + result.original_response_size
        extracted = result.extracted_size
        reduction = output["metadata"]["reduction_percentage"]
        
        total_original += original
        total_extracted += extracted
        
        status = "" if result.success else " ⚠"
        print(f"{result.source_id:<20}{status} {format_size(original):>12} {format_size(extracted):>12} {reduction:>9.1f}%")
    
    print("─" * 65)
    
    if total_original > 0:
        total_reduction = (total_original - total_extracted) / total_original * 100
        print(f"{'Total':<20} {format_size(total_original):>12} {format_size(total_extracted):>12} {total_reduction:>9.1f}%")
    
    print("═" * 65)


def main():
    parser = argparse.ArgumentParser(
        description='Extract analyzable content from Innovera AI log files for EASD quality analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single pair
  python extractor.py --request ai-log-259643-requestParams.json --response ai-log-259643-response.json

  # Directory batch mode  
  python extractor.py --input ./logs --output ./extracted

  # Use summarized documents for smaller output
  python extractor.py --input ./logs --output ./extracted --summarized
        """
    )
    
    # Single file mode
    parser.add_argument('--request', '-r', type=Path, 
                        help='Path to request params JSON file')
    parser.add_argument('--response', '-s', type=Path,
                        help='Path to response JSON file')
    
    # Batch mode
    parser.add_argument('--input', '-i', type=Path,
                        help='Input directory containing log file pairs')
    parser.add_argument('--output', '-o', type=Path, default=Path('extracted'),
                        help='Output directory for extracted files (default: ./extracted)')
    
    # Options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed extraction progress')
    parser.add_argument('--summarized', action='store_true',
                        help='Use summarizedVaultDocumentsData instead of full vaultDocumentsData')
    parser.add_argument('--skip-templates', action='store_true',
                        help='Exclude template_content from output (further size reduction)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.request and args.response:
        # Single file mode
        if not args.request.exists():
            print(f"Error: Request file not found: {args.request}", file=sys.stderr)
            sys.exit(1)
        if not args.response.exists():
            print(f"Error: Response file not found: {args.response}", file=sys.stderr)
            sys.exit(1)
        
        pairs = [(args.request, args.response)]
        
    elif args.input:
        # Batch mode
        if not args.input.exists():
            print(f"Error: Input directory not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        
        pairs = find_file_pairs(args.input)
        if not pairs:
            print(f"Error: No matching file pairs found in {args.input}", file=sys.stderr)
            sys.exit(1)
        
        if args.verbose:
            print(f"Found {len(pairs)} file pair(s) in {args.input}")
    
    else:
        # Check if sample files exist in current directory or sample-jsons/
        sample_dirs = [Path('.'), Path('sample-jsons')]
        pairs = []
        
        for sample_dir in sample_dirs:
            if sample_dir.exists():
                found_pairs = find_file_pairs(sample_dir)
                if found_pairs:
                    pairs = found_pairs
                    if args.verbose:
                        print(f"Found {len(pairs)} file pair(s) in {sample_dir}")
                    break
        
        if not pairs:
            parser.print_help()
            print("\nError: Provide --request/--response or --input directory, or place files in current directory", 
                  file=sys.stderr)
            sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Process all pairs
    results = []
    
    for request_path, response_path in pairs:
        if args.verbose:
            print(f"\nProcessing: {request_path.name}")
        
        result = process_pair(
            request_path,
            response_path,
            use_summarized=args.summarized,
            skip_templates=args.skip_templates,
            verbose=args.verbose
        )
        
        if result.success or result.request_data or result.response_content:
            output = build_output(result)
            
            # Write output file
            output_filename = f"{result.source_id}-extracted.json"
            output_path = args.output / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            if args.verbose:
                print(f"  → Wrote {output_path}")
            
            results.append((result, output))
        else:
            print(f"  ✗ Skipped {result.source_id}: {'; '.join(result.warnings)}", file=sys.stderr)
    
    # Print summary
    if results:
        print_summary_table(results)
        
        # Show field summary for first file
        if len(results) == 1:
            _, output = results[0]
            print("\nExtracted Fields:")
            if output.get('request'):
                print(f"  Request: {', '.join(output['request'].keys())}")
            if output.get('response', {}).get('content'):
                content_len = len(output['response']['content'])
                print(f"  Response: content ({format_size(content_len)})")
    else:
        print("\nNo files were successfully processed.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
