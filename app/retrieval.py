"""第5回: ページ単位のベクトル検索。"""

import argparse
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from collections import defaultdict, deque
from pathlib import Path

from google import genai
from google.genai import types
from pydantic_ai.exceptions import ModelHTTPError
import sqlite_vec

from app.config import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL,
    RAG_DB, RAG_DOCUMENT_IDS, RAG_SOURCE, RAG_TOP_K, RAG_TREE, get_api_key,
    RAG_GRAPH, GRAPH_START_K, GRAPH_HOPS, GRAPH_MAX_RESULTS,
    TERM_HUB_MAX_PAGES, TERM_PAGES_PER_LECTURE,
)


def get_all_materials() -> str:
    """全資料を、講義・章・ページの情報付きで1つの文字列にまとめる。"""
    documents = json.loads(RAG_SOURCE.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"資料: {document['id']} / {document['title']}\n"
        f"章: {section['section']}\n"
        f"ページ: PDF p.{page['page']} / {page['title']}\n{page['content']}"
        for document in documents
        for section in document["content"]
        for page in section["content"]
    )

def load_pages(source: Path = RAG_SOURCE) -> list[dict]:
    """資料→章→ページのJSONを、対象資料のページ一覧に変換する。"""
    documents = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for document in documents:
        if document["id"] not in RAG_DOCUMENT_IDS:
            continue
        for section in document["content"]:
            for page in section["content"]:
                pages.append({"page_id": page["id"], "document_id": document["id"],
                              "lecture": int(document["id"][3:5]), "document_title": document["title"],
                              "section": section["section"], "title": page["title"],
                              "page": page["page"], "content": page["content"]})
    return pages


def browse_tree(node_id: str | None = None) -> dict:
    """直下のページは先頭文、指定したページ自身は全文を返す。"""
    tree = json.loads(RAG_TREE.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            info.update({key: value for key, value in pages[node["id"]].items() if key != "page_id"})
            if not full:
                info["content"] = re.split(r"(?<=[。！？])|\n", info["content"].strip(), maxsplit=1)[0]
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


def read_page(page_id: str) -> dict:
    """第1〜3回の本講義から、指定した1ページの本文と引用情報を返す。"""
    for page in load_pages(RAG_SOURCE):
        if page["page_id"] == page_id:
            return page
    raise ValueError("対象のページが見つかりません。")


def normalized_vector(values: list[float]) -> list[float]:
    """ベクトルの長さを1に揃える。"""
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("埋め込みがゼロベクトルです。")
    return [value / norm for value in values]


def embed_text(text: str, task_type: str) -> list[float]:
    """Geminiで文章を埋め込む。入力上限を超えた場合は切り捨てず停止する。"""
    with genai.Client(api_key=get_api_key()) as client:
        count = client.models.count_tokens(model=EMBEDDING_MODEL, contents=text).total_tokens
        if count is None or count > EMBEDDING_MAX_TOKENS:
            raise ValueError(f"埋め込み入力が上限を超えるか計測不能です: {count} tokens。")
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
        )
    embedding = response.embeddings[0]
    if embedding.statistics and embedding.statistics.truncated:
        raise ValueError("埋め込みAPIが本文を切り捨てました。")
    return normalized_vector(embedding.values)


@contextmanager
def open_index(path: Path):
    """既存の検索DBを読み取り専用で開き、終了時に閉じる。"""
    if not path.is_file():
        raise FileNotFoundError("検索DBがありません。")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        yield db
    finally:
        db.close()


def search(query: str) -> list[dict]:
    """質問を埋め込み、近いページを最大RAG_TOP_K件返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    with open_index(RAG_DB) as db:
        vector = embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute("""
            SELECT page_id, document_id, lecture, document_title, section, title, page, content,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (sqlite_vec.serialize_float32(vector), RAG_TOP_K))
        return [dict(row) for row in rows]


def search_graph(query: str) -> dict:
    """ベクトル検索を入口に、前後ページと共通用語で候補を広げる。"""
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]
             if node["kind"] == "term" or node["id"] in pages}
    adjacent = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] in ("next_page", "mentions") and a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)

    seeds = [hit for hit in search(query) if hit["page_id"] in pages][:min(GRAPH_START_K, GRAPH_MAX_RESULTS)]
    seed_ids = {seed["page_id"] for seed in seeds}
    results, candidates = [], {}
    for rank, seed in enumerate(seeds):
        root = seed["page_id"]
        results.append({**pages[root], "distance": seed["distance"], "hops": 0,
                        "path": [{"id": root, "title": pages[root]["title"]}]})
        queue = deque([(root, [root])])
        visited = {root}
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= GRAPH_HOPS:
                continue
            # 同じ深さなら、接続先が少ない用語からたどる。
            neighbors = sorted(adjacent[current], key=lambda n: (len(adjacent[n]), n))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                route = path + [neighbor]
                queue.append((neighbor, route))
                if neighbor not in pages or neighbor in seed_ids:
                    continue
                frequency = max((len(adjacent[n]) for n in route if nodes[n]["kind"] == "term"), default=0)
                priority = (len(route) - 1, frequency, rank, neighbor)
                candidate = {**pages[neighbor], "hops": len(route) - 1,
                             "path": [{"id": n, "title": nodes[n]["title"]} for n in route]}
                if neighbor not in candidates or priority < candidates[neighbor][0]:
                    candidates[neighbor] = (priority, candidate)
    expanded = [candidate for _, candidate in sorted(candidates.values())]
    return {"seeds": results, "expanded": expanded[:max(0, GRAPH_MAX_RESULTS - len(results))],
            "max_hops": GRAPH_HOPS}


def mentioned_lectures(query: str) -> set[int]:
    """「第1回と第3回」「第1〜3回」のような回の指定を取り出す。"""
    lectures = set()
    for start, end in re.findall(r"第(\d+)\s*[〜~～\-－]\s*(\d+)回", query):
        lectures.update(range(int(start), int(end) + 1))
    lectures.update(int(n) for n in re.findall(r"第(\d+)回", query))
    return lectures


def match_terms(query: str, terms: list[dict]) -> list[dict]:
    """用語の名前・別名が質問に含まれる用語を返す。英字の別名は単語単位で照合する。"""
    text = query.casefold()
    matched = []
    for term in terms:
        for name in {term["title"], *term.get("aliases", [])}:
            name = name.casefold()
            if name.isascii():
                found = re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text)
            else:
                found = len(name) >= 2 and name in text
            if found:
                matched.append(term)
                break
    return matched


def search_term_graph(query: str) -> dict:
    """質問に出る用語ノードを入口に加え、その用語が出るページを講義ごとに選ぶ。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    terms = [node for node in graph["nodes"] if node["kind"] == "term"]
    mentions, next_pages = defaultdict(set), defaultdict(set)
    for edge in graph["edges"]:
        for a, b in ((edge["source"], edge["target"]), (edge["target"], edge["source"])):
            if edge["relation"] == "mentions" and a.startswith("term:") and b in pages:
                mentions[a].add(b)
            elif edge["relation"] == "next_page" and a in pages and b in pages:
                next_pages[a].add(b)

    lectures = mentioned_lectures(query)
    in_scope = lambda page_id: not lectures or pages[page_id]["lecture"] in lectures
    matched = match_terms(query, terms)
    entry_terms = [t for t in matched if 0 < len(mentions[t["id"]]) <= TERM_HUB_MAX_PAGES]
    skipped_terms = [t["title"] for t in matched if len(mentions[t["id"]]) > TERM_HUB_MAX_PAGES]

    with open_index(RAG_DB) as db:
        vector = sqlite_vec.serialize_float32(embed_text(query.strip(), "RETRIEVAL_QUERY"))
        candidate_ids = sorted({p for t in entry_terms for p in mentions[t["id"]]})
        distances = {row["page_id"]: row["distance"] for row in db.execute(
            f"SELECT page_id, vec_distance_cosine(embedding, ?) AS distance FROM chunks "
            f"WHERE page_id IN ({','.join('?' * len(candidate_ids))})", (vector, *candidate_ids))
        } if candidate_ids else {}
        ranked = [dict(row) for row in db.execute("""
            SELECT page_id, vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (vector, RAG_TOP_K))]
    seed_hits = [hit for hit in ranked if hit["page_id"] in pages and in_scope(hit["page_id"])] \
        or [hit for hit in ranked if hit["page_id"] in pages]

    results, chosen = [], set()

    def add(page_id, **info):
        if page_id in chosen or len(results) >= GRAPH_MAX_RESULTS:
            return
        chosen.add(page_id)
        results.append({**pages[page_id], **info})

    for hit in seed_hits[:GRAPH_START_K]:
        add(hit["page_id"], distance=hit["distance"], via="vector")
    # 用語ごと・講義ごとに距離の近いページを選び、1つの講義に偏らないようにする。
    for term in entry_terms:
        by_lecture = defaultdict(list)
        for page_id in mentions[term["id"]]:
            if in_scope(page_id) and page_id in distances and page_id not in chosen:
                by_lecture[pages[page_id]["lecture"]].append(page_id)
        for lecture in sorted(by_lecture):
            for page_id in sorted(by_lecture[lecture], key=lambda p: (distances[p], p))[:TERM_PAGES_PER_LECTURE]:
                add(page_id, distance=distances[page_id], via=f"term:{term['title']}")
    if not entry_terms:
        for hit in seed_hits[:GRAPH_START_K]:
            for page_id in sorted(next_pages[hit["page_id"]]):
                add(page_id, via=f"next_page:{hit['page_id']}")

    return {"matched_terms": [t["title"] for t in entry_terms], "skipped_hub_terms": skipped_terms,
            "lectures": sorted(lectures), "results": results}



def run_full_context(agent, question: str, model, instructions: str, prompt_template: str):
    """全資料を一括投入し、検索ツールを使わずに回答する。"""
    prompt = prompt_template.format(materials=get_all_materials(), question=question)
    print(f"[full-context] 入力{len(prompt):,}文字")
    try:
        with agent.override(tools=[], instructions=instructions):
            return agent.run_sync(prompt, model=model)
    except ModelHTTPError as e:
        print(f"[full-context] APIエラー HTTP {e.status_code}: {e.body}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する")
    parser.add_argument("query", help="調べたい内容")
    args = parser.parse_args()
    for i, result in enumerate(search(args.query), 1):
        print(f"\n{'=' * 60}\n検索結果 {i}\n{'=' * 60}")
        print(f"講義: {result['document_title']}")
        print(f"章: {result['section']}")
        print(f"タイトル: {result['title']}")
        print(f"ページ: {result['page']}")
        print(f"\n{result['content']}")
