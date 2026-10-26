"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass(frozen=True)
class Measurement:
    """LLM の応答と、API 呼び出しにかかった時間・トークン数。"""

    text: str
    elapsed_seconds: float
    input_tokens: int | None
    output_tokens: int | None


def measure(question: str, model: str = MODEL) -> Measurement:
    """質問を送り、API 呼び出しの応答時間とトークン数を返す。"""
    client = genai.Client(api_key=get_api_key())
    start = time.perf_counter()
    response = client.models.generate_content(model=model, contents=question)
    elapsed_seconds = time.perf_counter() - start

    input_tokens = output_tokens = None
    if response.usage_metadata is not None:
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count

    return Measurement(
        text=response.text,
        elapsed_seconds=elapsed_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def print_measurement(measurement: Measurement) -> None:
    """計測結果を日本語で表示する。"""
    print(f"[応答時間] {measurement.elapsed_seconds:.3f} 秒")
    print(
        f"[トークン数] 入力: {measurement.input_tokens}, "
        f"出力: {measurement.output_tokens}"
    )


def ask(question: str) -> str:
    measurement = measure(question)
    print_measurement(measurement)
    return measurement.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
