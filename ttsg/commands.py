from __future__ import annotations

from argparse import Namespace

from .cache import CacheLayout
from .config import Paths, Settings
from .consolidation import consolidate
from .documents import discover_documents
from .export_docx import export_docx
from .extract import build_consensus, clean_with_llm, extract_target_segment
from .models import ConsensusDocument
from .ocr import build_provider, pdf_digest
from .preprocess import preprocess_pages, tile_pages
from .rasterize import rasterize_document


def run_ocr(args: Namespace) -> int:
    paths = Paths.from_root()
    paths.ensure_runtime_dirs()
    settings = Settings.from_env()
    cache = CacheLayout(paths)
    provider = build_provider(
        args.provider,
        mistral_api_key=settings.mistral_api_key,
        gemini_api_key=settings.gemini_api_key,
        replicate_api_token=settings.replicate_api_token,
        prefer_replicate=getattr(args, "prefer_replicate", False),
    )
    documents = discover_documents(paths)
    for document in documents:
        if cache.ocr_path(args.provider, document).exists() and not args.force:
            print(f"[skip] OCR cache exists for {args.provider} / {document.doc_id}")
            continue
        dpi = getattr(args, "dpi", 400)
        crop = not getattr(args, "no_crop", False)
        num_tiles = getattr(args, "tiles", 0)
        overlap = getattr(args, "overlap", 0.03)
        print(f"[ocr] {args.provider} -> {document.doc_id} (dpi={dpi}, crop={crop}, tiles={num_tiles}, overlap={overlap})")
        images = rasterize_document(cache, document, dpi=dpi, force=args.force)
        if crop:
            images = preprocess_pages(images, cache.preprocessed_dir(document, dpi=dpi), force=args.force)
        if num_tiles > 1 and hasattr(provider, "process_tiled"):
            tile_groups = tile_pages(
                images,
                cache.tiles_dir(document, dpi=dpi, num_tiles=num_tiles, overlap_pct=overlap),
                num_tiles=num_tiles,
                overlap_pct=overlap,
                force=args.force,
            )
            ocr_document = provider.process_tiled(document.doc_id, images, tile_groups, pdf_digest(document.pdf_path))
        else:
            ocr_document = provider.process(document.doc_id, images, pdf_digest(document.pdf_path))
        cache.save_ocr(args.provider, document, ocr_document)
    return 0


def run_extract_target(args: Namespace) -> int:
    paths = Paths.from_root()
    paths.ensure_runtime_dirs()
    settings = Settings.from_env()
    cache = CacheLayout(paths)
    documents = discover_documents(paths)
    no_clean = getattr(args, "no_clean", False)

    clean_usage_log: list[dict] = []
    for document in documents:
        segments = []
        for provider in args.providers:
            ocr_document = cache.load_ocr(provider, document)
            if not ocr_document:
                continue
            segment = extract_target_segment(ocr_document)
            cache.save_extraction(provider, document, segment)
            segments.append(segment)
        consensus = build_consensus(document.doc_id, segments)
        # LLM-based cleaning: strip content from other companies
        if not no_clean and consensus.normalized_text.strip() and settings.replicate_api_token:
            print(f"[clean] {document.doc_id}: cleaning with Gemini Flash via Replicate...")
            cleaned, clean_metrics = clean_with_llm(
                consensus.normalized_text,
                "Parla Enerji Yatırımları Anonim Şirketi",
                settings.replicate_api_token,
                gemini_api_key=settings.gemini_api_key,
            )
            clean_usage_log.append({"doc_id": document.doc_id, "stage": "clean", **clean_metrics})
            consensus = ConsensusDocument(
                doc_id=consensus.doc_id,
                normalized_text=cleaned,
                status=consensus.status,
                providers=consensus.providers,
                source_pages=consensus.source_pages,
                notes=[*consensus.notes, "llm_cleaned"],
            )
        cache.save_consensus(document, consensus)
        if consensus.normalized_text.strip():
            (paths.target_only / f"{document.doc_id}.md").write_text(consensus.normalized_text + "\n", encoding="utf-8")
        print(f"[extract-target] {document.doc_id}: {consensus.status}")

    # Persist cleaning LLM usage for benchmark
    if clean_usage_log:
        from .utils import write_json
        paths.final.mkdir(parents=True, exist_ok=True)
        write_json(paths.final / "llm_clean_usage.json", clean_usage_log)
        total_cost = sum(e.get("cost_usd", 0) for e in clean_usage_log)
        print(f"[extract-target] LLM cleaning cost: ${total_cost:.6f} ({len(clean_usage_log)} calls)")
    return 0


def run_consolidate(args: Namespace) -> int:
    paths = Paths.from_root()
    paths.ensure_runtime_dirs()
    settings = Settings.from_env()
    cache = CacheLayout(paths)
    result = consolidate(cache, replicate_token=settings.replicate_api_token, gemini_api_key=settings.gemini_api_key)
    cache.save_consolidation(result)
    print(f"[consolidate] board_members={len(result.board_members)} articles={len(result.articles)}")
    return 0


def run_export_docx(args: Namespace) -> int:
    paths = Paths.from_root()
    paths.ensure_runtime_dirs()
    cache = CacheLayout(paths)
    output_path = export_docx(cache)
    print(f"[export-docx] {output_path}")
    return 0


def run_pipeline(args: Namespace) -> int:
    providers = list(args.providers)
    for provider_name in providers:
        ocr_args = Namespace(
            provider=provider_name,
            force=args.force,
            prefer_replicate=args.prefer_replicate,
            dpi=args.dpi,
            no_crop=args.no_crop,
            tiles=getattr(args, "tiles", 0),
            overlap=getattr(args, "overlap", 0.03),
        )
        run_ocr(ocr_args)
    extract_args = Namespace(providers=providers, no_clean=getattr(args, "no_clean", False))
    run_extract_target(extract_args)
    run_consolidate(Namespace())
    run_export_docx(Namespace())
    return 0
