import os
import re
import time
import logging
from typing import List, Dict, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
import networkx as nx
import matplotlib
matplotlib.use('Agg') # バックエンドをAggに設定 (サーバー環境等で推奨)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, RetryError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import gradio as gr
from dotenv import load_dotenv
import io
from PIL import Image

# --- Configuration & Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
MAX_LLM_RETRIES = 3
NODE_SUMMARY_LENGTH = 25 # ノードに表示する要約の最大文字数
RETRY_DELAY_SECONDS = 5 # 基本的なリトライ遅延秒数
MAX_SEARCH_ITERATIONS = 5 # 情報収集の最大繰り返し回数

# --- Japanese Font Setup for Matplotlib ---
def setup_japanese_font_for_matplotlib() -> Optional[str]:
    """
    システムにインストールされている利用可能な日本語フォントを探し、Matplotlibに設定を試みます。
    見つかったフォント名を返します。
    """
    font_candidates = [
        'Yu Gothic', 'Yu Mincho', 'MS Gothic', 'MS Mincho', 'Meiryo',  # Windows Standard
        'Hiragino Sans', 'Hiragino Mincho ProN',  # macOS Standard
        'IPAexGothic', 'IPAexMincho',  # Common Free Japanese Fonts
        'Noto Sans CJK JP', 'Noto Serif CJK JP'  # Google Noto Fonts
    ]

    # Matplotlibのフォントキャッシュを更新しようと試みる (非推奨APIのため注意)
    try:
        if hasattr(fm, '_rebuild'):
            fm._rebuild()
            logger.info("Matplotlib font cache rebuild initiated (if supported).")
    except Exception as e:
        logger.warning(f"Error attempting to rebuild font cache (fm._rebuild()): {e}. This might be normal for your Matplotlib version.")

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font_name in font_candidates:
        if font_name in available_fonts:
            try:
                font_prop = fm.FontProperties(family=font_name)
                fm.findfont(font_prop, fallback_to_default=False) # フォントが実際に利用可能か確認

                # 日本語フォントが見つかった場合、デフォルト設定を変更
                plt.rcParams['font.family'] = font_name
                current_sans_serif = [f for f in plt.rcParams.get('font.sans-serif', []) if f != 'DejaVu Sans']
                plt.rcParams['font.sans-serif'] = [font_name] + current_sans_serif

                logger.info(f"Found and set usable Japanese font: '{font_name}' for Matplotlib.")
                return font_name
            except Exception as e:
                logger.debug(f"Font '{font_name}' in ttflist but not usable or error during setting: {e}")
        else:
            logger.debug(f"Font '{font_name}' not found in Matplotlib's font list.")

    logger.warning(
        "No standard Japanese font found or usable by Matplotlib. "
        "Graph labels may not display Japanese characters correctly. "
        "Consider installing 'IPAexGothic' or 'Noto Sans CJK JP'."
    )
    return None

# モジュールロード時に日本語フォント設定を試行
INSTALLED_JAPANESE_FONT = setup_japanese_font_for_matplotlib()

# --- Data Classes ---
@dataclass
class ParsedStep:
    """LLMの思考プロセスの一ステップを表すデータクラス"""
    id: int
    type: str
    raw_content: str
    summarized_content: str = ""
    basis_ids: List[int] = field(default_factory=list)
    search_query: Optional[str] = None

@dataclass
class LLMThoughtProcess:
    """LLMからの応答全体を構造化したデータクラス"""
    raw_response: str
    steps: List[ParsedStep] = field(default_factory=list)
    final_answer: str = ""
    error_message: Optional[str] = None

# --- Core Classes ---
class GeminiHandler:
    """Gemini APIとの対話を管理するクラス"""
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-lite"):
        if not api_key:
            raise ValueError("Gemini API key is not set.")
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name,
                generation_config={"temperature": 0.3},
            )
            self.model_name = model_name
            logger.info(f"GeminiHandler initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize GeminiGenerativeModel: {e}")
            raise
        self.error_messages: List[str] = []

    def _get_base_prompt_template(self) -> str:
        """LLMへの基本プロンプトテンプレートを返す"""
        return """ユーザーの質問に対して、あなたの思考プロセスを段階的に説明しながら回答を生成してください。
思考プロセスは必ず `<think>` と `</think>` タグで囲んで出力してください。
各思考ステップは以下の形式で記述してください:
ステップ[番号]: [種別] 内容：[具体的な思考内容] (根拠: ステップX, ステップY)

利用可能なステップ種別リスト:
- 問題定義
- 仮説提示
- 情報収集 (内容には `[検索クエリ：ここに検索キーワード]` の形式で検索キーワードを記述)
- 情報分析
- 検証
- 中間結論
- 論点
- 反論
- 参照 (内容には参照元を記述)
- 最終結論候補

思考プロセスを記述した後、`<think>` タグの外に最終的な回答を記述してください。
思考ステップの内容は、後で要約されることを意識し、具体的かつ簡潔に記述してください。

以下は出力形式の厳格な例です：
質問：日本の首都はどこですか？
<think>
ステップ1: 問題定義 内容：ユーザーは日本の首都について質問している。
ステップ2: 情報収集 内容：[検索クエリ：日本の首都]
ステップ3: 情報分析 内容：検索結果によると、日本の首都は東京である。 (根拠: ステップ2)
ステップ4: 最終結論候補 内容：日本の首都は東京であると回答する。 (根拠: ステップ3)
</think>
日本の首都は東京です。

---
これまでの思考ステップの履歴 (もしあれば):
{previous_steps_str}
---
直前の検索結果 (必要な場合のみ参照):
{search_results_str}
---

上記の履歴と検索結果を踏まえ、思考を続けるか、最終的な回答を生成してください。
思考を続ける場合は、新しい`<think>`ブロックで、以前のステップ番号とは重複しないように続きのステップ番号を使用してください。

ユーザーの質問：{user_question}
"""

    def generate_response(self, user_question: str, previous_steps: Optional[List[ParsedStep]] = None, search_results: Optional[str] = None) -> Iterator[str]:
        """Geminiモデルにプロンプトを送信し、応答を生成します（ジェネレータ）。"""
        self.error_messages = [] # エラーリストをリセット
        previous_steps_str = self._format_previous_steps(previous_steps) if previous_steps else "なし"
        search_results_str = search_results if search_results else "なし"
        prompt = self._get_base_prompt_template().format(
            user_question=user_question,
            previous_steps_str=previous_steps_str,
            search_results_str=search_results_str
        )

        for attempt in range(MAX_LLM_RETRIES):
            try:
                logger.info(f"Sending request to Gemini (Attempt {attempt + 1}/{MAX_LLM_RETRIES}) Model: {self.model_name}")
                response = self.model.generate_content(prompt)
                # レスポンスのテキスト部分を抽出
                response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')) if response.candidates and response.candidates[0].content.parts else ""
                
                if not response_text.strip() and response.prompt_feedback:
                     # プロンプトフィードバックがある場合（例：安全フィルターなど）
                     safety_feedback = response.prompt_feedback
                     reason = safety_feedback.block_reason if safety_feedback.block_reason else "不明な理由"
                     error_msg = f"Geminiからの応答がブロックされました。理由: {reason}"
                     logger.warning(error_msg)
                     self.error_messages.append(error_msg)
                     yield f"ERROR:{error_msg}" # エラーを示すマーカーをyield
                     return # 処理を中断

                logger.info("Received response from Gemini.")

                if self._validate_response_format(response_text):
                    yield response_text # 正しい形式ならyieldして終了
                    return
                
                # フォーマットエラーの場合
                self.error_messages.append(f"Geminiの出力形式エラー。再試行中 ({attempt + 1}/{MAX_LLM_RETRIES})...")
                logger.warning(f"Gemini response format error (Attempt {attempt + 1}). Retrying...")
                yield f"Geminiの出力形式エラー。再試行中です ({attempt + 2}/{MAX_LLM_RETRIES})..."
                if attempt < MAX_LLM_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    self.error_messages.append("Gemini応答形式エラー。最大再試行回数に達しました。")
                    yield "ERROR:LLM_FORMAT_ERROR_MAX_RETRIES"
                    return

            except RetryError as e:
                error_msg = f"Gemini API retry error exceeded: {e}"
                logger.error(error_msg)
                self.error_messages.append(error_msg)
                yield f"ERROR:{error_msg}"
                return # リトライ回数を超えたら終了
            except GoogleAPIError as e:
                error_detail = self._handle_google_api_error(e, attempt)
                if error_detail: # エラーの詳細が返ってきた場合
                    yield f"ERROR:{error_detail}"
                    if "API_KEY_INVALID" in error_detail or "API_RATE_LIMIT" in error_detail:
                        return # 重大なエラーの場合は終了
                    if attempt < MAX_LLM_RETRIES - 1:
                         # レート制限などで待機が必要な場合
                         wait_time = RETRY_DELAY_SECONDS * (attempt + 1) * 1.5 # 少し長めに待つ
                         logger.info(f"Waiting for {wait_time} seconds before next retry.")
                         time.sleep(wait_time)
                    else:
                        self.error_messages.append(f"Gemini APIエラー。最大再試行回数に達しました。({error_detail})")
                        yield f"ERROR:API_ERROR_MAX_RETRIES_{error_detail}"
                        return
                else: # その他の予期せぬAPIエラー
                    error_msg = "予期せぬGemini APIエラーが発生しました。"
                    logger.error(error_msg, exc_info=True)
                    self.error_messages.append(error_msg)
                    yield f"ERROR:{error_msg}"
                    return
            except Exception as e:
                error_msg = f"Gemini API呼び出し中に予期せぬエラー: {e}"
                logger.error(error_msg, exc_info=True)
                self.error_messages.append(error_msg)
                yield f"ERROR:{error_msg}"
                return # 予期せぬエラーは即時終了

        # 全ての試行が終わっても成功しなかった場合
        if not self.error_messages:
             self.error_messages.append("不明なエラーにより、Geminiからの応答を取得できませんでした。")
             yield "ERROR:UNKNOWN_FAILURE_AFTER_RETRIES"

    def _handle_google_api_error(self, e: GoogleAPIError, attempt: int) -> Optional[str]:
        """Google APIエラーを処理し、エラーコード文字列またはNoneを返す"""
        error_str = str(e)
        if "API key not valid" in error_str or (hasattr(e, 'grpc_status_code') and e.grpc_status_code == 7):
            msg = "API_KEY_INVALID: Gemini APIキーが無効か、権限がありません。"
            self.error_messages.append(msg)
            return msg
        elif "Rate limit exceeded" in error_str or "QUOTA_EXCEEDED" in error_str:
            retry_after_seconds = RETRY_DELAY_SECONDS
            # レート制限時の具体的な待機時間を取得しようと試みる (APIによってはヘッダー等で通知される場合がある)
            # Gemini API の場合、error_info に含まれることがある
            if hasattr(e, 'error_info') and e.error_info and hasattr(e.error_info, 'quota_violations'):
                 for violation in e.error_info.quota_violations:
                    if hasattr(violation, 'retry_delay') and hasattr(violation.retry_delay, 'seconds'):
                        retry_after_seconds = max(retry_after_seconds, violation.retry_delay.seconds)
            
            msg = f"API_RATE_LIMIT: Gemini APIのレート制限に達しました。{retry_after_seconds}秒後に再試行してください。"
            logger.warning(msg)
            self.error_messages.append(msg)
            # Gradioへのフィードバック用にも同じメッセージを返す
            return f"レート制限超過。{retry_after_seconds}秒待機します。" # time.sleep は generate_response 内で行う
        else:
            msg = f"Gemini APIエラー: {error_str}"
            logger.error(msg, exc_info=True)
            self.error_messages.append(msg)
            return f"API_GENERAL_ERROR: {error_str}" # 一般的なAPIエラーとして報告

    def _format_previous_steps(self, steps: List[ParsedStep]) -> str:
        """過去の思考ステップをプロンプト用に整形します。"""
        if not steps:
            return "なし"
        return "\n".join([f"ステップ{s.id}: {s.type} 内容：{s.raw_content}" + (f" (根拠: {', '.join(map(str, s.basis_ids))})" if s.basis_ids else "") for s in steps])

    def _validate_response_format(self, response_text: str) -> bool:
        """LLMの応答が期待される形式であるか検証します。"""
        if not response_text.strip():
            logger.warning("Validation failed: Response text is empty.")
            return False
        if not ("<think>" in response_text and "</think>" in response_text):
            logger.warning("Validation failed: Missing <think> or </think> tags.")
            return False
        return True

    def summarize_text(self, text_to_summarize: str, max_length: int = NODE_SUMMARY_LENGTH) -> str:
        """与えられたテキストを要約します。エラー発生時は例外を送出。"""
        if not text_to_summarize or not text_to_summarize.strip():
            return "（空の内容）"
        try:
            prompt = f"以下のテキストを日本語で{max_length}文字以内を目安に、最も重要なポイントを捉えて要約してください。要約のみを出力してください:\n\n\"{text_to_summarize}\""
            # 要約用のモデルインスタンスを再作成（シングルトンパターンでも良いが、ここではシンプルに）
            summary_model = genai.GenerativeModel(self.model_name)
            response = summary_model.generate_content(prompt)
            summary = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip() if response.candidates and response.candidates[0].content.parts else ""

            if not summary:
                logger.warning(f"Summarization resulted in empty text for: '{text_to_summarize[:50]}...'")
                # 要約が空の場合、元のテキストの一部を返す
                return text_to_summarize[:max_length] + "..." if len(text_to_summarize) > max_length else text_to_summarize
            
            # 要約結果が長すぎる場合の切り詰め
            return summary if len(summary) <= max_length + 10 else summary[:max_length+5] + "..."
        except Exception as e:
            # summarize_textで発生したエラーは呼び出し元でハンドリングされるように再送出
            logger.error(f"Error during summarization: {e}", exc_info=True)
            raise RuntimeError(f"要約処理中にエラーが発生しました: {e}") from e

    def get_error_messages(self) -> List[str]:
        """GeminiHandler内で発生したエラーメッセージのリストを返します。"""
        return self.error_messages


class GoogleSearchHandler:
    """Google Custom Search APIを介して検索を実行するクラス"""
    def __init__(self, api_key: str, cse_id: str):
        self.service = None
        self.is_enabled = False
        if not api_key or not cse_id:
            logger.warning("Google API key or CSE ID is not set. Search functionality will be disabled.")
            return
        try:
            # developerKey は build 関数に渡す
            self.service = build("customsearch", "v1", developerKey=api_key)
            self.cse_id = cse_id
            self.is_enabled = True
            logger.info("Google Search Handler initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Google Search service: {e}")

    def search(self, query: str, num_results: int = 3) -> Optional[str]:
        """指定されたクエリでGoogle検索を実行し、結果を整形して返します。"""
        if not self.is_enabled or not self.service:
            logger.warning("Google Search service not initialized or disabled. Skipping search.")
            return "検索機能が無効です。APIキーまたはCSE IDを確認してください。"
        try:
            logger.info(f"Performing Google search for: '{query}'")
            # searchメソッドはdeveloperKeyを受け取らない (build時に指定)
            res = self.service.cse().list(q=query, cx=self.cse_id, num=num_results, lr='lang_ja').execute()

            if 'items' not in res or not res['items']:
                logger.info(f"No search results found for query: '{query}'")
                return "検索結果なし。"
            
            search_results_text = "検索結果:\n"
            for i, item in enumerate(res['items']):
                title = item.get('title', 'タイトルなし')
                snippet = item.get('snippet', 'スニペットなし').replace("\n", " ").strip()
                link = item.get('link', '#')
                search_results_text += f"{i+1}. タイトル: {title}\n  スニペット: {snippet}\n  URL: {link}\n\n"
            
            logger.info(f"Search successful for query: '{query}'")
            return search_results_text.strip()

        except HttpError as e:
            # HttpErrorのステータスコードと理由を取得
            status_code = e.resp.status
            reason = e._get_reason()
            logger.error(f"Google Search API HTTP error: {status_code} {reason} for query '{query}'")
            return f"検索APIエラー ({status_code}): {reason}"
        except Exception as e:
            logger.error(f"An unexpected error occurred during Google search for '{query}': {e}", exc_info=True)
            return "検索中に予期せぬエラーが発生しました。"


class ThoughtParser:
    """LLMの思考プロセスのテキストを解析し、構造化されたデータに変換するクラス"""
    def parse_llm_output(self, raw_llm_output: str) -> LLMThoughtProcess:
        """LLMの生の出力を解析し、思考ステップと最終回答を抽出します。"""
        result = LLMThoughtProcess(raw_response=raw_llm_output)
        
        # <think> タグの内容を抽出
        think_match = re.search(r"<think>(.*?)</think>", raw_llm_output, re.DOTALL)

        if not think_match:
            # <think> タグがない場合、全体を最終回答とみなし、エラーメッセージを設定
            result.error_message = "LLMの応答に<think>タグが見つかりません。"
            result.final_answer = raw_llm_output.strip()
            logger.warning(f"Parsing failed: No <think> tag found. Response: {raw_llm_output[:100]}...")
            return result

        think_content = think_match.group(1).strip()
        # <think> タグ以降の部分を最終回答とする
        result.final_answer = raw_llm_output[think_match.end():].strip()

        # 各ステップを解析するための正規表現
        # ステップ番号、種別、内容、根拠を抽出
        step_pattern = re.compile(
            # r"ステップ\s*(\d+)\s*:\s*([^内容]+?)\s*内容：(.*?)(?:\s*\(根拠:\s*(ステップ\s*\d+(?:,\s*ステップ\s*\d+)*)\s*\))?$"
            r"ステップ\s*(\d+)\s*:\s*([\w\s]+?)\s*内容：(.*?)(?:\s*\(根拠:\s*(?:ステップ\s*(\d+(?:,\s*ステップ\s*\d+)*))\s*\))?$",
            re.MULTILINE | re.IGNORECASE
        )

        parsed_steps_in_segment = []
        for match in step_pattern.finditer(think_content):
            try:
                step_id_from_llm = int(match.group(1))
                step_type = match.group(2).strip()
                raw_content = match.group(3).strip()
                basis_str = match.group(4) # 根拠部分 (e.g., "ステップ1, ステップ3")

                # 根拠IDの抽出
                basis_ids = []
                if basis_str:
                    basis_ids = [int(b_id) for b_id in re.findall(r'\d+', basis_str)]
                
                # 検索クエリの抽出
                search_query_match = re.search(r"\[検索クエリ：(.*?)\]", raw_content, re.IGNORECASE)
                search_query = search_query_match.group(1).strip() if search_query_match else None

                parsed_steps_in_segment.append(ParsedStep(
                    id=step_id_from_llm,
                    type=step_type,
                    raw_content=raw_content,
                    basis_ids=basis_ids,
                    search_query=search_query
                ))
            except (ValueError, IndexError, TypeError) as e:
                msg = f"ステップ解析エラー (マッチ: '{match.group(0)}'): {e}"
                logger.error(msg)
                # エラーメッセージにこの問題を記録
                result.error_message = (result.error_message or "") + "\n" + msg
            except Exception as e:
                msg = f"予期せぬステップ解析エラー (マッチ: '{match.group(0)}'): {e}"
                logger.error(msg, exc_info=True)
                result.error_message = (result.error_message or "") + "\n" + msg

        result.steps = parsed_steps_in_segment
        
        # <think>タグ内に内容があったが、有効なステップが解析できなかった場合
        if not result.steps and think_content:
            msg = "思考プロセス(<think>タグ内)が見つかりましたが、有効なステップを解析できませんでした。"
            logger.warning(msg + f" Content snippet: {think_content[:100]}...")
            result.error_message = (result.error_message or "") + "\n" + msg
        
        logger.info(f"Parsed LLM output: {len(result.steps)} steps, final answer snippet: '{result.final_answer[:50]}...'")
        return result


class GraphGenerator:
    """
    LLMの思考プロセスをグラフとして可視化するクラス
    - ノードは長方形で表示
    - カスタム階層レイアウトを使用
    - 右上に凡例を表示
    - 矢印の重なりを回避し、種類に応じてスタイルを変更
    """
    def __init__(self, llm_handler: 'GeminiHandler', installed_japanese_font: Optional[str]):
        self.llm_handler = llm_handler
        self.error_messages_for_graph: List[str] = []
        self.font_properties = None

        if installed_japanese_font:
            try:
                # FontPropertiesオブジェクトを作成
                self.font_properties = fm.FontProperties(family=installed_japanese_font)
                logger.info(f"GraphGenerator initialized. Using Matplotlib font: '{installed_japanese_font}'.")
            except Exception as e:
                error_msg = f"GraphGenerator: 設定済み日本語フォント '{installed_japanese_font}' の読み込みに失敗しました: {e}"
                logger.error(error_msg)
                self.error_messages_for_graph.append(error_msg)
        else:
            error_msg = "適切な日本語フォントが見つかりませんでした。グラフの文字が正しく表示されない可能性があります。"
            logger.warning(f"GraphGenerator: {error_msg}")
            self.error_messages_for_graph.append(error_msg)

    def _summarize_step_contents(self, steps: List['ParsedStep'], progress_fn=None):
        """各ステップの内容を要約する。進捗コールバック関数を受け取る。"""
        if not steps: return
        logger.info(f"Summarizing contents for {len(steps)} steps...")
        
        for i, step in enumerate(steps):
            # 要約がまだ行われていない場合のみ実行
            if not step.summarized_content:
                try:
                    step.summarized_content = self.llm_handler.summarize_text(step.raw_content, max_length=NODE_SUMMARY_LENGTH)
                    logger.debug(f"Summarized step {step.id}: '{step.summarized_content}'")
                except Exception as e:
                    error_msg = f"ステップS{step.id}の要約中にエラーが発生しました: {e}"
                    logger.error(error_msg)
                    self.error_messages_for_graph.append(error_msg)
                    # 要約失敗時のフォールバック内容を設定
                    fallback_summary = step.raw_content[:NODE_SUMMARY_LENGTH] + "..." if len(step.raw_content) > NODE_SUMMARY_LENGTH else step.raw_content
                    step.summarized_content = f"要約失敗: {fallback_summary}"
            
            # 進捗コールバックを実行
            if progress_fn:
                progress_fn((i + 1) / len(steps), f"ステップ {step.id} の内容を要約中 ({i+1}/{len(steps)})...")
            time.sleep(0.02) # 短い遅延を追加して進捗が見やすくする
        logger.info("Summarization complete.")
        
    def _custom_hierarchical_layout(self, G, root_node=0, vertical_gap=0.5, horizontal_gap=0.5):
        """カスタム階層レイアウトアルゴリズム"""
        if not G or root_node not in G:
            logger.warning(f"Graph is empty or root node {root_node} not found. Cannot create hierarchical layout.")
            return None
            
        levels = {root_node: 0} # 各ノードのレベルを格納
        nodes_at_level = {0: [root_node]} # レベルごとのノードリスト
        queue = [root_node]
        visited = {root_node}
        max_level = 0

        # BFSで階層構造を構築
        processed_nodes = 0
        while queue:
            processed_nodes += 1
            parent = queue.pop(0)
            parent_level = levels[parent]
            
            # 子ノードを取得し、sequentialエッジを優先するようにソート
            children = sorted(list(G.successors(parent)), key=lambda n: G.get_edge_data(parent, n).get('type') != 'sequential')
            
            for child in children:
                if child not in visited:
                    visited.add(child)
                    child_level = parent_level + 1
                    levels[child] = child_level
                    if child_level not in nodes_at_level:
                        nodes_at_level[child_level] = []
                    nodes_at_level[child_level].append(child)
                    queue.append(child)
                    max_level = max(max_level, child_level)
        
        # 未訪問のノードがあれば、最後のレベルに追加（グラフが連結でない場合）
        all_nodes = set(G.nodes())
        unvisited = all_nodes - visited
        if unvisited:
            unvisited_level = max_level + 1
            nodes_at_level[unvisited_level] = list(unvisited)
            for node in unvisited:
                levels[node] = unvisited_level
            logger.warning(f"{len(unvisited)} nodes were not reached from the root node and placed in level {unvisited_level}.")
        
        # 各レベルのノードを水平方向に配置
        pos = {}
        for level in range(max_level + 1):
            if level in nodes_at_level:
                nodes = sorted(nodes_at_level[level]) # ノードIDでソートして一貫性を保つ
                num_nodes = len(nodes)
                # レベル全体の幅を計算 (ノード数に応じて調整)
                level_width = max(1.0, (num_nodes - 1) * horizontal_gap) # 最低でも1.0の幅を確保
                x_start = -level_width / 2
                
                for i, node in enumerate(nodes):
                    x = x_start + i * horizontal_gap
                    y = -level * vertical_gap # Y座標はレベルに応じて下げる
                    pos[node] = (x, y)
                    logger.info(f"Node {node} at level {level}: pos=({x:.2f}, {y:.2f})") # デバッグ用ログ

        if len(pos) != G.number_of_nodes():
             logger.warning(f"Layout calculation mismatch: {len(pos)} positions calculated, but graph has {G.number_of_nodes()} nodes.")
        
        # 実際に配置されたノード数に基づいてスケーリング調整
        if pos:
             all_x = [p[0] for p in pos.values()]
             all_y = [p[1] for p in pos.values()]
             x_range = max(all_x) - min(all_x) if all_x else 1
             y_range = max(all_y) - min(all_y) if all_y else 1
             scale_x = 8 / x_range if x_range > 0 else 1
             scale_y = 6 / y_range if y_range > 0 else 1
             scale = min(scale_x, scale_y) * 0.8 # 全体的なスケールを調整
             
             scaled_pos = {node: (p[0] * scale, p[1] * scale) for node, p in pos.items()}
             logger.info(f"Applied scaling factor: {scale:.2f}")
             return scaled_pos
             
        return pos # スケール調整前の位置を返す

    def create_thinking_graph(self, user_question: str, all_steps: List['ParsedStep'], final_answer_text: str, progress_fn=None) -> Optional[plt.Figure]:
        """思考プロセスを可視化するネットワーク図を生成する"""
        self.error_messages_for_graph = [] # エラーリストを初期化
        # LLMハンドラからのエラーがあれば追加
        llm_errors = self.llm_handler.get_error_messages()
        if llm_errors: self.error_messages_for_graph.extend(llm_errors)

        # ステップ内容の要約を実行
        self._summarize_step_contents(all_steps, progress_fn)
        
        G = nx.DiGraph()
        QUESTION_NODE_ID = 0 # 質問ノードのID

        # --- ノードとエッジの追加 ---
        try:
            # 質問ノードの要約
            question_summary = self.llm_handler.summarize_text(user_question, max_length=NODE_SUMMARY_LENGTH + 15)
        except Exception as e:
            error_msg = f"質問の要約中にエラーが発生しました: {e}"
            logger.error(error_msg)
            self.error_messages_for_graph.append(error_msg)
            question_summary = user_question[:NODE_SUMMARY_LENGTH + 15] + "..."
        G.add_node(QUESTION_NODE_ID, label=f"質問:\n{question_summary}", type="question", color="skyblue")
        
        valid_step_ids_in_graph = {QUESTION_NODE_ID} # グラフに追加された有効なノードIDのセット
        
        # 思考ステップノードを追加
        if all_steps:
            # ノードIDの最大値を取得して、最終回答ノードのIDを決定
            max_existing_id = max([s.id for s in all_steps] + [QUESTION_NODE_ID])
            current_last_step_id = QUESTION_NODE_ID # 直前のノードID（初期値は質問ノード）

            # ステップをID順にソートして追加
            for step in sorted(all_steps, key=lambda s: s.id):
                node_graph_id = step.id
                # ノードIDが重複している場合はスキップ (通常は発生しないはずだが念のため)
                if node_graph_id in G: 
                    logger.warning(f"Duplicate node ID {node_graph_id} encountered. Skipping.")
                    continue 
                
                label_text = f"S{step.id}: {step.type}\n{step.summarized_content}"
                G.add_node(node_graph_id, label=label_text, type="ai_step", color="khaki")
                valid_step_ids_in_graph.add(node_graph_id)

                # 直前のノードからこのステップへの sequential エッジを追加
                if current_last_step_id in G:
                    G.add_edge(current_last_step_id, node_graph_id, type="sequential", style="solid", color="gray")
                current_last_step_id = node_graph_id # 現在のノードを次のステップの直前ノードとして更新

            # 最終回答ノードを追加
            try:
                answer_summary = self.llm_handler.summarize_text(final_answer_text, max_length=NODE_SUMMARY_LENGTH + 15)
            except Exception as e:
                error_msg = f"最終回答の要約中にエラーが発生しました: {e}"
                logger.error(error_msg)
                self.error_messages_for_graph.append(error_msg)
                answer_summary = final_answer_text[:NODE_SUMMARY_LENGTH + 15] + "..."
            
            final_answer_node_id = max_existing_id + 1
            G.add_node(final_answer_node_id, label=f"最終回答:\n{answer_summary}", type="final_answer", color="lightgreen")
            valid_step_ids_in_graph.add(final_answer_node_id)
            
            # 最後のステップノードから最終回答ノードへの sequential エッジを追加
            if current_last_step_id != QUESTION_NODE_ID and current_last_step_id in G:
                 G.add_edge(current_last_step_id, final_answer_node_id, type="sequential", style="solid", color="gray")
            # もしステップが一つもなく、質問から直接回答につながる場合
            elif QUESTION_NODE_ID in G and final_answer_node_id not in G.successors(QUESTION_NODE_ID):
                 G.add_edge(QUESTION_NODE_ID, final_answer_node_id, type="sequential", style="solid", color="gray")

            # basis エッジを追加
            for step in all_steps:
                target_node_id = step.id
                for basis_id in step.basis_ids:
                    # basis_id が有効で、target_node_id も有効、かつ異なるノードの場合のみエッジを追加
                    if basis_id in valid_step_ids_in_graph and target_node_id in valid_step_ids_in_graph and basis_id != target_node_id:
                        # 既に sequential エッジが存在しない場合のみ basis エッジを追加
                        if not G.has_edge(basis_id, target_node_id) or G.get_edge_data(basis_id, target_node_id).get('type') != 'sequential':
                            G.add_edge(basis_id, target_node_id, type="basis", style="dashed", color="purple")
        
        # ステップがなく、最終回答しかない場合
        elif final_answer_text and QUESTION_NODE_ID in G:
            try:
                answer_summary = self.llm_handler.summarize_text(final_answer_text, max_length=NODE_SUMMARY_LENGTH + 15)
            except Exception as e:
                error_msg = f"最終回答の要約中にエラーが発生しました: {e}"
                logger.error(error_msg)
                self.error_messages_for_graph.append(error_msg)
                answer_summary = final_answer_text[:NODE_SUMMARY_LENGTH + 15] + "..."
            final_answer_node_id = QUESTION_NODE_ID + 1 # 質問ノードのID + 1
            G.add_node(final_answer_node_id, label=f"最終回答:\n{answer_summary}", type="final_answer", color="lightgreen")
            G.add_edge(QUESTION_NODE_ID, final_answer_node_id, type="sequential", style="solid", color="gray")
        
        # 有効なノードが存在しない場合はグラフ生成を中止
        if not G.nodes():
            self.error_messages_for_graph.append("有効な思考プロセスまたは回答が得られませんでした。グラフは生成できません。")
            logger.error("No nodes found to generate the graph.")
            return None

        # --- レイアウト計算 ---
        pos = None
        layout_attempts = 0
        max_layout_attempts = 3
        
        while layout_attempts < max_layout_attempts:
            try:
                if layout_attempts == 0:
                    logger.info("Attempting custom hierarchical layout.")
                    pos = self._custom_hierarchical_layout(G, root_node=QUESTION_NODE_ID, vertical_gap=0.6, horizontal_gap=0.8)
                    if pos is None: raise ValueError("Custom layout returned None.")
                elif layout_attempts == 1:
                     logger.info("Custom hierarchical layout failed or returned None. Falling back to spring layout.")
                     pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
                elif layout_attempts == 2:
                     logger.info("Spring layout failed. Falling back to random layout.")
                     pos = nx.random_layout(G, seed=42)
                else:
                     # すべてのレイアウト試行が失敗した場合
                     raise RuntimeError("All layout attempts failed.")
                
                # レイアウトが成功したかチェック (posが空でないことを確認)
                if pos and len(pos) == G.number_of_nodes():
                    logger.info(f"Layout successful using method {layout_attempts+1}.")
                    break # 成功したらループを抜ける
                else:
                    raise ValueError(f"Layout calculation resulted in invalid positions (expected {G.number_of_nodes()} nodes, got {len(pos)}).")

            except Exception as e:
                logger.warning(f"Layout attempt {layout_attempts + 1} failed: {e}")
                self.error_messages_for_graph.append(f"グラフレイアウトの試行 {layout_attempts + 1} に失敗しました: {e}")
                layout_attempts += 1
                pos = None # 失敗したら pos をリセット

        # レイアウト計算に完全に失敗した場合
        if pos is None:
            self.error_messages_for_graph.append("グラフのレイアウト計算に失敗しました。")
            logger.error("Failed to calculate graph layout after all attempts.")
            plt.close(plt.figure()) # 開いている図を閉じる
            return None
            
        # --- Matplotlib描画設定 ---
        fig, ax = plt.subplots(figsize=(min(20, max(12, G.number_of_nodes() * 2.0)), min(15, max(8, G.number_of_nodes() * 1.0))))
        
        labels = {n: d['label'] for n, d in G.nodes(data=True)}
        node_colors = {n: d['color'] for n, d in G.nodes(data=True)}

        # ノードを描画 (長方形)
        node_width, node_height = 0.4, 0.2 # ノードのサイズ比率 (軸スケール基準)
        node_positions_adjusted = {}
        for node, (x, y) in pos.items():
            width = node_width
            height = node_height
            # レベルに応じた幅調整（オプション）
            # level = levels.get(node, 0) 
            
            rect = plt.Rectangle((x - width / 2, y - height / 2), width, height, 
                                 facecolor=node_colors[node], alpha=0.95, transform=ax.transData)
            ax.add_patch(rect)
            node_positions_adjusted[node] = (x, y - height / 2) # ラベル配置のため、ノード下部の中心点を計算

        # ラベルを描画 (ノードの中央下部に配置)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax, 
                                font_weight='bold', verticalalignment='top', # topにすることでノードの下部に揃える
                                fontproperties=self.font_properties) # 日本語フォント適用

        # エッジをFancyArrowPatchで描画
        drawn_edges = set() # 同じ方向への重複エッジ描画を防ぐ
        for u, v, data in G.edges(data=True):
            if (u, v) in drawn_edges: continue
                
            start_pos = pos[u]
            end_pos = pos[v]
            edge_type = data.get('type', 'sequential')
            
            # 矢印スタイル定義
            arrow_style_params = {'head_length': 0.15, 'head_width': 0.08, 'tail_width': 0.01} # スケール基準のサイズ
            connectionstyle = 'arc3,rad=0.1' # 少しカーブさせる
            
            if edge_type == 'sequential':
                color = 'black'
                style = 'solid'
                # 矢印のサイズを少し大きく
                arrow_style_params.update({'head_length': 0.2, 'head_width': 0.1})
                connectionstyle = 'arc3,rad=0.05' # 少し緩やかなカーブ
            elif edge_type == 'basis':
                color = 'purple'
                style = 'dashed'
                # 矢印のサイズを少し小さく
                arrow_style_params.update({'head_length': 0.12, 'head_width': 0.06})
                connectionstyle = 'arc3,rad=0.15' # 少し強めのカーブ
            else:
                continue # 未知のタイプは描画しない

            # FancyArrowPatch を作成
            arrow = mpatches.FancyArrowPatch(
                start_pos, end_pos,
                arrowstyle=mpatches.ArrowStyle.Simple(**arrow_style_params), # ArrowStyle.Simpleを使用
                color=color,
                linestyle=style,
                shrinkA=node_width/2 * 1.1, # ノードの幅の半分より少し大きく縮める
                shrinkB=node_height/2 * 1.1, # ノードの高さの半分より少し大きく縮める
                mutation_scale=150, # 矢印全体のスケール調整係数 (figsizeに依存しないように)
                connectionstyle=connectionstyle,
                pathpatch_kwds={'lw': 1.5 if style=='solid' else 1.0} # linewidth指定 (FancyArrowPatch自体には直接指定できないため)
            )
            ax.add_patch(arrow)
            drawn_edges.add((u, v))

        # --- 凡例の設定 ---
        node_handles = [
            mpatches.Patch(color='skyblue', label='質問'),
            mpatches.Patch(color='khaki', label='AIの思考ステップ'),
            mpatches.Patch(color='lightgreen', label='最終回答')
        ]
        edge_handles = [
            mlines.Line2D([], [], color='black', linestyle='solid', label='時系列の流れ'),
            mlines.Line2D([], [], color='purple', linestyle='dashed', label='根拠・参照')
        ]
        all_handles = node_handles + edge_handles
        # 凡例に日本語フォントを適用
        ax.legend(handles=all_handles, loc='upper right', fontsize='medium', title="凡例", prop=self.font_properties)
        
        # --- 図全体の調整 ---
        ax.set_title("AI思考連鎖の可視化 (Gemini)", fontsize=16, fontweight='bold', fontproperties=self.font_properties)
        ax.autoscale() # 自動スケーリング
        ax.margins(0.15) # 余白を追加
        plt.axis('off') # 軸を非表示
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # タイトルとの重なりを避けるための調整

        logger.info("Graph visualization created successfully.")
        return fig

    def get_error_messages_for_graph(self) -> List[str]:
        """グラフ生成中に発生したエラーメッセージのリストを返す"""
        return self.error_messages_for_graph

    def get_error_messages_html(self) -> str:
        """グラフ生成エラーメッセージをHTML形式で返す"""
        if not self.error_messages_for_graph: return ""
        # 重複を除去し、ソートして表示
        unique_messages = sorted(list(set(self.error_messages_for_graph)))
        # HTMLエスケープ処理を追加（クロスサイトスクリプティング対策）
        escaped_messages = [msg.replace('&', '&').replace('<', '<').replace('>', '>') for msg in unique_messages]
        
        html = "<div style='color: red; border: 1px solid red; padding: 10px; margin-top: 10px; background-color: #ffeeee;'>" \
               "<strong>グラフ生成に関する注意・エラー:</strong><ul>"
        for msg in escaped_messages:
            html += f"<li>{msg}</li>"
        html += "</ul></div>"
        return html

class AISystem:
    """AI思考システムの中核クラス。LLMと検索機能を統合し、思考プロセスを管理します。"""
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.google_search_api_key = os.getenv("GOOGLE_API_KEY") # 変数名を修正
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")

        if not self.gemini_api_key:
            logger.error("CRITICAL: GEMINI_API_KEY environment variable not set. Gemini functionality will be disabled.")
            # Gradio起動時にエラー表示させるため、例外送出はしないでおく
        
        try:
            # GeminiHandlerの初期化 (APIキーがない場合でもインスタンス化は試みる)
            self.llm_handler = GeminiHandler(api_key=self.gemini_api_key)
        except ValueError as e:
            logger.error(f"Initialization failed for GeminiHandler: {e}")
            # エラーメッセージは後でGradio側で表示
            self.llm_handler = None # エラーを示すためにNoneを設定
        
        # GoogleSearchHandlerの初期化 (APIキーやCSE IDがない場合は無効になる)
        self.search_handler = GoogleSearchHandler(api_key=self.google_search_api_key, cse_id=self.google_cse_id)
        
        # ThoughtParserは常に初期化
        self.parser = ThoughtParser()
        
        # GraphGeneratorの初期化 (LLMハンドラが必要)
        if self.llm_handler:
            self.graph_generator = GraphGenerator(llm_handler=self.llm_handler, installed_japanese_font=INSTALLED_JAPANESE_FONT)
        else:
            self.graph_generator = None # LLMハンドラがない場合はNone
            logger.error("GraphGenerator could not be initialized because GeminiHandler failed.")

        logger.info("AISystem initialized.")

    def _append_and_renumber_steps(self, existing_steps: List[ParsedStep], new_steps_from_llm: List[ParsedStep]) -> List[ParsedStep]:
        """既存のステップリストに、重複を除外して新しいステップを追加し、IDを連番に振り直します。"""
        updated_all_steps = list(existing_steps)
        # 既存ステップの (タイプ, 内容) のペアで重複をチェック
        existing_signatures = {(step.type, step.raw_content) for step in existing_steps}
        
        current_max_id = max(s.id for s in existing_steps) if existing_steps else 0

        for new_step in new_steps_from_llm:
            # 新しいステップが既に存在するかチェック
            if (new_step.type, new_step.raw_content) in existing_signatures:
                logger.info(f"Skipping duplicate step detected (content match): Type='{new_step.type}', Content='{new_step.raw_content[:30]}...'")
                continue # 重複はスキップ
            
            # 新しいステップに連番のIDを割り当て
            current_max_id += 1
            new_step.id = current_max_id
            updated_all_steps.append(new_step)
            # 追加したステップのシグネチャもセットに追加
            existing_signatures.add((new_step.type, new_step.raw_content))
            
        logger.info(f"Appended {len(new_steps_from_llm) - (len(new_steps_from_llm) - len(existing_signatures))} new steps. Total steps: {len(updated_all_steps)}")
        return updated_all_steps

    def process_question_iterations(self, user_question: str, progress_fn) -> Tuple[str, Optional[Image.Image], str]:
        """
        ユーザーの質問に対するAIの思考プロセスを反復的に実行し、最終回答、思考グラフ、エラーメッセージを生成します。
        Gradioの進捗表示に対応するため、ジェネレータとして動作します。
        """
        if not self.llm_handler:
             return ("Gemini APIキーが無効なため、処理を開始できません。", None, "<p style='color:red; font-weight:bold;'>Gemini APIキーが無効です。設定を確認してください。</p>")
        if not self.graph_generator:
             return ("システムエラーのため、処理を開始できません。", None, "<p style='color:red; font-weight:bold;'>システム内部エラーが発生しました。</p>")

        # 初期進捗設定
        progress_fn(0, desc="AI(Gemini)が思考を開始しました...")
        
        accumulated_steps: List[ParsedStep] = []
        accumulated_final_answer: str = ""
        current_raw_llm_output_segment = ""
        search_iteration_count = 0
        overall_error_messages: List[str] = [] # 処理全体のエラーメッセージを収集

        # --- 1. 最初のLLM応答取得 ---
        llm_response_generator = self.llm_handler.generate_response(user_question)
        try:
            for update_message in llm_response_generator:
                if "ERROR:" in update_message:
                    current_raw_llm_output_segment = update_message
                    break # エラーが発生したらループを抜ける
                # update_messageが完了メッセージ（...で終わらない）の場合、最終的な応答として保存
                if not update_message.endswith("..."):
                    current_raw_llm_output_segment = update_message
                progress_fn(0.1, desc=update_message) # 進捗を更新
            
            # LLMハンドラのエラーメッセージを取得して統合
            overall_error_messages.extend(self.llm_handler.get_error_messages())

            # エラーチェック
            if "ERROR:" in current_raw_llm_output_segment:
                error_msg = f"AI処理エラー（初回応答）: {current_raw_llm_output_segment.replace('ERROR:', '').strip()}"
                logger.error(error_msg)
                overall_error_messages.append(error_msg)
                # エラーが発生した場合、ここで処理を終了
                return ("処理中にエラーが発生しました。", None, self._format_overall_error_messages(overall_error_messages))

            # LLM応答の解析
            parsed_segment_obj = self.parser.parse_llm_output(current_raw_llm_output_segment)
            # ステップを追加（IDの振り直しを含む）
            accumulated_steps = self._append_and_renumber_steps(accumulated_steps, parsed_segment_obj.steps)
            accumulated_final_answer = parsed_segment_obj.final_answer
            # パーサーからのエラーがあれば追加
            if parsed_segment_obj.error_message:
                overall_error_messages.append(f"パーサー通知: {parsed_segment_obj.error_message}")
            
            # 現在のステップリストを保存（次のループで使用）
            last_parsed_segment_steps = parsed_segment_obj.steps

        except Exception as e:
            error_msg = f"初回応答処理中に予期せぬエラーが発生しました: {e}"
            logger.error(error_msg, exc_info=True)
            overall_error_messages.append(error_msg)
            return ("処理中に予期せぬエラーが発生しました。", None, self._format_overall_error_messages(overall_error_messages))

        # --- 2. 検索とLLMの反復処理 ---
        # 直前のステップに検索クエリが含まれている場合、かつ最大反復回数未満の場合にループ
        while any(s.type == "情報収集" and s.search_query for s in last_parsed_segment_steps) and \
              search_iteration_count < MAX_SEARCH_ITERATIONS:
            
            # 情報収集ステップを探す (最初に見つかったものを利用)
            search_step = next((s for s in last_parsed_segment_steps if s.type == "情報収集" and s.search_query), None)
            if not search_step:
                break # 検索対象ステップがなければループを抜ける

            search_iteration_count += 1
            query = search_step.search_query
            progress_percentage = min(0.2 + search_iteration_count * 0.1, 0.8) # 進捗率を計算
            progress_fn(progress_percentage, desc=f"'{query[:30]}...'を検索中 ({search_iteration_count}/{MAX_SEARCH_ITERATIONS})...")
            
            # Google検索を実行
            s_results = self.search_handler.search(query) or "情報取得失敗。"
            if "エラー" in s_results: # 検索エラーが発生した場合
                 overall_error_messages.append(f"検索エラー: {s_results}")
                 logger.warning(f"Google search failed for query '{query}'. Result: {s_results}")
                 # エラーがあっても続行するかどうか？ -> 今回は続行してLLMに判断させる
            
            progress_fn(progress_percentage + 0.05, desc="検索結果を元にAI(Gemini)が再考中...")

            # 検索結果を渡してLLMに再応答生成を依頼
            llm_response_generator = self.llm_handler.generate_response(user_question, accumulated_steps, s_results)
            current_raw_llm_output_segment = ""
            try:
                for update_message in llm_response_generator:
                    if "ERROR:" in update_message:
                        current_raw_llm_output_segment = update_message
                        break
                    if not update_message.endswith("..."):
                         current_raw_llm_output_segment = update_message
                    progress_fn(progress_percentage + 0.1, desc=update_message) # より詳細な進捗表示
                
                overall_error_messages.extend(self.llm_handler.get_error_messages())

                # エラーチェック
                if "ERROR:" in current_raw_llm_output_segment:
                    error_msg = f"AI処理エラー（検索後）: {current_raw_llm_output_segment.replace('ERROR:', '').strip()}"
                    logger.error(error_msg)
                    overall_error_messages.append(error_msg)
                    # エラーが発生したらループを抜ける (失敗時の最終回答を返す)
                    break 

                # LLM応答の解析
                parsed_segment_obj = self.parser.parse_llm_output(current_raw_llm_output_segment)
                # ステップを追加
                accumulated_steps = self._append_and_renumber_steps(accumulated_steps, parsed_segment_obj.steps)
                # 最終回答が更新されたら保存
                if parsed_segment_obj.final_answer.strip():
                    accumulated_final_answer = parsed_segment_obj.final_answer
                # パーサーからのエラーがあれば追加
                if parsed_segment_obj.error_message:
                    overall_error_messages.append(f"パーサー通知（検索後）: {parsed_segment_obj.error_message}")
                
                last_parsed_segment_steps = parsed_segment_obj.steps
                
                # 新しい応答に検索クエリが含まれていなければループを終了
                if not any(s.type == "情報収集" and s.search_query for s in last_parsed_segment_steps):
                    break
            
            except Exception as e:
                error_msg = f"検索後の応答処理中に予期せぬエラーが発生しました: {e}"
                logger.error(error_msg, exc_info=True)
                overall_error_messages.append(error_msg)
                # エラーが発生した場合、ループを中断し、現在の状態を返す
                break 

        # --- 3. 最終回答の調整 ---
        # LLMからの最終回答がない場合、または空の場合のフォールバック
        if not accumulated_final_answer.strip() and not ("ERROR:" in current_raw_llm_output_segment):
            if accumulated_steps:
                accumulated_final_answer = "思考プロセスに基づき検討しましたが、最終的な回答は明示的に出力されませんでした。上記グラフを参照してください。"
            else:
                accumulated_final_answer = "AIからの有効な応答が得られませんでした。"

        # --- 4. グラフ生成 ---
        progress_fn(0.8, desc="思考グラフを生成中です...")
        
        # グラフ生成関数に進捗コールバックを渡す
        summary_prog_lambda = lambda p, d: progress_fn(0.8 + p * 0.15, desc=d)
        
        graph_image_pil = None
        fig = None
        try:
            if self.graph_generator:
                 fig = self.graph_generator.create_thinking_graph(user_question, accumulated_steps, accumulated_final_answer, summary_prog_lambda)
                 # グラフ生成後のエラーメッセージを追加
                 overall_error_messages.extend(self.graph_generator.get_error_messages_for_graph())

                 if fig:
                     # MatplotlibのFigureオブジェクトをPIL Imageに変換
                     buf = io.BytesIO()
                     fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                     plt.close(fig) # 図を閉じてメモリを解放
                     buf.seek(0)
                     graph_image_pil = Image.open(buf)
                     logger.info("Successfully generated graph image.")
            else:
                 overall_error_messages.append("グラフ生成モジュールが初期化されていないため、グラフは生成されませんでした。")
                 logger.error("GraphGenerator is None, cannot create graph.")

        except Exception as e:
            error_msg = f"グラフ生成または画像変換中にエラーが発生しました: {e}"
            logger.error(error_msg, exc_info=True)
            overall_error_messages.append(error_msg)
            if fig: plt.close(fig) # エラー発生時も図を閉じる

        # --- 5. 結果の返却 ---
        progress_fn(1.0, desc="処理完了")
        final_answer_html = accumulated_final_answer # Markdown形式で表示されることを期待
        graph_errors_html = self._format_overall_error_messages(overall_error_messages)

        return final_answer_html, graph_image_pil, graph_errors_html

    def _format_overall_error_messages(self, messages: List[str]) -> str:
        """全てのエラーメッセージをHTML形式で整形します。"""
        if not messages:
            return ""
        # 重複を除去し、ソートして表示
        unique_messages = sorted(list(set(messages)))
        # HTMLエスケープ処理
        escaped_messages = [msg.replace('&', '&').replace('<', '<').replace('>', '>') for msg in unique_messages]
        
        html_content = "<div style='color: red; border: 1px solid red; padding: 10px; margin-top: 10px; background-color: #ffeeee;'>" \
                       "<strong>処理に関する注意・エラー:</strong><ul>"
        for msg in escaped_messages:
            html_content += f"<li>{msg}</li>"
        html_content += "</ul></div>"
        return html_content


def create_gradio_interface(system: AISystem):
    """Gradioインターフェースを作成し、設定します。"""
    # Gradioのテーマ設定
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue, 
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.gray
    )
    
    with gr.Blocks(theme=theme, title="AI思考連鎖可視化システム") as demo:
        gr.Markdown("# 🧠 AI思考連鎖可視化システム (Gemini版)")
        gr.Markdown("ユーザーの質問に対し、AI(Gemini)が思考プロセスを段階的に示しながら回答します。思考の連鎖はネットワーク図として可視化されます。")

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="質問を入力してください:", 
                    placeholder="例: 日本のAI技術の最新トレンドとその課題は何ですか？", 
                    lines=3, 
                    show_copy_button=True,
                    container=False # コンポーネントのみ表示
                )
                submit_button = gr.Button("質問を送信する", variant="primary", scale=0) # scale=0 で幅を小さく
            with gr.Column(scale=1):
                # 設定情報を表示するMarkdownブロック
                settings_md = f"""
                ### 設定情報
                - **LLM Model**: `{system.llm_handler.model_name if system.llm_handler else '未設定'}`
                - **Google検索**: `{'有効' if system.search_handler.is_enabled else '無効'}`
                - **最大検索回数**: `{MAX_SEARCH_ITERATIONS}`
                - **LLM再試行回数**: `{MAX_LLM_RETRIES}`
                - **使用フォント(グラフ)**: `{INSTALLED_JAPANESE_FONT if INSTALLED_JAPANESE_FONT else '(日本語フォントが見つかりません)'}`
                """
                gr.Markdown(settings_md)

        gr.Markdown("---")
        gr.Markdown("## 🤖 AIの回答と思考プロセス")
        
        with gr.Row():
            with gr.Column(scale=1):
                answer_output = gr.Markdown(label="AIの最終回答")
                # エラー/通知メッセージ表示用コンポーネント
                graph_errors_output = gr.HTML(label="処理に関する通知・エラー") 
            with gr.Column(scale=2):
                # グラフ表示用コンポーネント
                graph_output = gr.Image(
                    label="思考プロセスネットワーク図", 
                    type="pil", 
                    interactive=False, # 表示のみ
                    show_download_button=True, # ダウンロードボタンを表示
                    height=400 # 画像の高さを固定
                )

        # --- イベントハンドラの設定 ---
        # submit_buttonがクリックされたときの処理
        def submit_fn_wrapper_for_gradio(question, progress=gr.Progress(track_tqdm=True)):
            """Gradioからの入力を受け取り、AISystemの処理を実行し、結果を返すラッパー関数"""
            if not question.strip():
                # 質問が空の場合はエラーメッセージを返す
                return ("質問が入力されていません。", None, "<p style='color:orange;'>質問を入力してください。</p>")
            
            # AISystemの処理を実行し、結果を返す
            # process_question_iterationsはジェネレータを返すため、ループ処理が必要
            final_result_tuple = (None, None, None)
            try:
                # process_question_iterations は yield するので、最後の結果を取得
                for result_tuple in system.process_question_iterations(question, progress):
                    final_result_tuple = result_tuple
                
                # 最終的な結果を返す
                return final_result_tuple
            
            except Exception as e:
                # 万が一 process_question_iterations 内で未捕捉の例外が発生した場合
                logger.error(f"Unhandled exception in submit_fn_wrapper_for_gradio: {e}", exc_info=True)
                error_html = "<div style='color: red; border: 1px solid red; padding: 10px;'>予期せぬエラーが発生しました。詳細はログを確認してください。</div>"
                return ("システムエラーが発生しました。", None, error_html)

        submit_button.click(
            fn=submit_fn_wrapper_for_gradio, 
            inputs=[question_input], 
            outputs=[answer_output, graph_output, graph_errors_output]
        )
        
        # 例示質問
        gr.Examples(
            examples=[
                ["日本のAI技術の最新トレンドとその社会への影響について教えてください。"],
                ["太陽光発電のメリット・デメリットを整理し、今後の展望を予測してください。"],
                ["気候変動が海洋生態系に与える影響について、具体的な事例を挙げて説明してください。"]
            ], 
            inputs=[question_input], 
            label="質問例"
        )
        
        gr.Markdown("---")
        gr.Markdown("© 2025 ユニークAIシステム構築道場 G9 (プロトタイプ - Gemini版). 回答生成には時間がかかることがあります。")
        
    return demo

if __name__ == "__main__":
    # アプリケーションのエントリーポイント
    
    # 環境変数チェック (起動時に確認)
    if not os.getenv("GEMINI_API_KEY"):
        print("\n" + "="*80)
        print(" WARNING: GEMINI_API_KEY environment variable not set.")
        print(" Gemini API functionality will be disabled.")
        print(" Please set the GEMINI_API_KEY environment variable.")
        print("="*80 + "\n")
        # この場合でも、Gradioインターフェースは表示する（エラーメッセージが出る）

    # AISystemインスタンスを作成
    ai_system_instance = AISystem()
    
    # Gradioインターフェースを作成
    gradio_app_interface = create_gradio_interface(ai_system_instance)
    
    # Gradioアプリケーションを起動
    try:
        logger.info("Launching Gradio interface...")
        # server_name="0.0.0.0" にすると外部からアクセス可能になる
        # server_port を指定してポート番号を固定
        gradio_app_interface.launch(server_name="localhost", server_port=7860) 
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {e}", exc_info=True)
        print(f"Error launching Gradio interface: {e}")