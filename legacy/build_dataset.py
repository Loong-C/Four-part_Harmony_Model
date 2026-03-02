import os
import json
import music21 as m21
from data_extractor import extract_structured_harmony_data
from vocab_manager import build_vocabularies, save_vocab


def _normalize_musicxml_paths(paths):
    """
    Normalize a mixed path list to unique .mxl/.xml string paths.
    """
    normalized = []
    seen = set()
    for raw_path in paths:
        path_str = str(raw_path)
        lower_path = path_str.lower()
        if not (lower_path.endswith(".mxl") or lower_path.endswith(".xml")):
            continue
        if path_str in seen:
            continue
        seen.add(path_str)
        normalized.append(path_str)
    return normalized


def _collect_bach_paths():
    """
    Collect Bach piece paths with a primary source and a compatibility fallback.
    Returns:
        tuple[list[str], str]: (paths, source_strategy)
    """
    diagnostics = []

    # Primary: composer lookup (works in music21 9.x)
    try:
        primary_paths = m21.corpus.getComposer("bach")
        normalized = _normalize_musicxml_paths(primary_paths)
        if normalized:
            return normalized, "getComposer('bach')"
        diagnostics.append("primary source returned 0 usable .mxl/.xml files")
    except Exception as exc:
        diagnostics.append(f"primary source failed: {exc}")

    # Fallback: general corpus scan + string filter
    try:
        fallback_paths = m21.corpus.getPaths(
            fileExtensions=("mxl", "xml"),
            name=("core", "local"),
        )
        bach_only = [p for p in fallback_paths if "bach" in str(p).lower()]
        normalized = _normalize_musicxml_paths(bach_only)
        if normalized:
            return normalized, "getPaths(...)+bach-filter"
        diagnostics.append("fallback source returned 0 usable .mxl/.xml files")
    except Exception as exc:
        diagnostics.append(f"fallback source failed: {exc}")

    detail = "; ".join(diagnostics) if diagnostics else "unknown reason"
    raise RuntimeError(
        "未找到任何可用的巴赫 MusicXML 曲目。"
        "请检查 music21 语料库安装是否完整。"
        f"诊断信息: {detail}"
    )


def create_full_dataset(save_dir="data"):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    print("正在搜寻巴赫曲库...")
    bach_paths, source_strategy = _collect_bach_paths()
    print(f"候选曲目总数: {len(bach_paths)} (来源: {source_strategy})")
    if not bach_paths:
        raise RuntimeError(
            f"候选曲目数为 0（来源: {source_strategy}）。"
            "请检查 music21 语料库是否可用。"
        )
    
    all_raw_data = []
    success_count = 0
    fail_count = 0
    
    # 遍历所有候选文件
    for path in bach_paths:
        file_name = os.path.basename(str(path))
        
        try:
            # 调用我们写好的提取函数！
            dataset = extract_structured_harmony_data(str(path))
            all_raw_data.extend(dataset) # 将提取的步骤合并到总池子中
            
            success_count += 1
            print(f"[{success_count}] 成功处理: {file_name} (提取了 {len(dataset)} 个时间步)")
            
        except Exception as e:
            # 真实数据通常会有脏数据（如非四声部、缺少调性等），我们用 try-except 优雅地跳过
            fail_count += 1
            print(f"[跳过] {file_name} - 原因: {e}")

    print("\n" + "="*40)
    print(f"数据提取完成！成功: {success_count} 首, 失败/跳过: {fail_count} 首")
    print(f"整个数据集共包含 {len(all_raw_data)} 个和声切片。")
    print("="*40)

    # 1. 保存原始的合并数据 (JSON 格式，方便人类随时检查)
    data_filepath = os.path.join(save_dir, 'bach_chorales_raw.json')
    print(f"正在保存全局数据到 {data_filepath} ...")
    with open(data_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_raw_data, f)

    # 2. 基于这几百首曲子，建立无所不包的“超级全局词表”
    print("正在构建并保存全局词表...")
    chord2id, dur2id, pitch2id = build_vocabularies(all_raw_data)
    
    save_vocab(chord2id, os.path.join(save_dir, 'chord_vocab.json'))
    save_vocab(dur2id, os.path.join(save_dir, 'duration_vocab.json'))
    save_vocab(pitch2id, os.path.join(save_dir, 'pitch_vocab.json'))
    
    print("所有操作完成！你的本地数据集已准备就绪。")

if __name__ == "__main__":
    create_full_dataset("data")
