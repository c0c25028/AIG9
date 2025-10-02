import os
import re
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
NODE_SUMMARY_LENGTH = 25
RETRY_DELAY_SECONDS = 5 # 基本的なリトライ遅延
MAX_SEARCH_ITERATIONS = 5

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

    # Matplotlibのフォントキャッシュを再構築 (起動時に一度だけ)
    # Note: fm.fontManager.findfont()はキャッシュを更新する副作用を持つため、
    # _rebuild()の明示的な呼び出しはMatplotlibのバージョンや環境によっては不要、あるいは問題を引き起こす可能性がある。
    # しかし、問題が解決しない場合は試す価値がある。
    try:
        # _rebuild() の呼び出しはバージョン依存性があるので、存在チェックまたはtry-exceptで囲む
        # 最新のMatplotlibでは非推奨または存在しない場合があるため、注意
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
                # findfontで実際のファイルパスを確認。fallback_to_default=Falseは重要で、
                # フォントが見つからない場合にデフォルトフォントにフォールバックさせないことで、
                # True Fontが見つかったかどうかのチェックを厳密にする。
                fm.findfont(font_prop, fallback_to_default=False)
                
                # 日本語フォントが見つかった場合、デフォルトのsans-serifからDejaVu Sansを削除し、
                # 見つかった日本語フォントを最優先にする。
                plt.rcParams['font.family'] = font_name
                # Sans-serifリストからDejaVu Sansを削除し、見つかったフォントを最初に追加
                current_sans_serif = [f for f in plt.rcParams.get('font.sans-serif', []) if f != 'DejaVu Sans']
                plt.rcParams['font.sans-serif'] = [font_name] + current_sans_serif
                
                logger.info(f"Found and set usable Japanese font: '{font_name}' globally for Matplotlib. Removed DejaVu Sans from default sans-serif list.")
                return font_name
            except Exception as e:
                logger.debug(f"Font '{font_name}' in ttflist but not usable or error during setting: {e}")
        else:
            logger.debug(f"Font '{font_name}' not in ttflist.")

    logger.warning(
        "No standard Japanese font found or usable by Matplotlib. "
        "Graph labels may not display Japanese characters correctly. "
        "Please install a Japanese font like 'IPAexGothic' or 'Noto Sans CJK JP'."
    )
    # 日本語フォントが見つからない場合は、デフォルトのsans-serifを維持
    # plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Verdana'] + plt.rcParams.get('font.sans-serif', [])
    return None

# Attempt to set a Japanese font globally when the module loads
INSTALLED_JAPANESE_FONT = setup_japanese_font_for_matplotlib()


@dataclass
class ParsedStep:
    id: int
    type: str
    raw_content: str
    summarized_content: str = ""
    basis_ids: List[int] = field(default_factory=list)
    search_query: Optional[str] = None

@dataclass
class LLMThoughtProcess:
    raw_response: str
    steps: List[ParsedStep] = field(default_factory=list)
    final_answer: str = ""
    error_message: Optional[str] = None


class GeminiHandler:
    """Gemini APIとの対話を管理するクラス"""
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-lite"):
        if not api_key:
            raise ValueError("Gemini API key is not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"temperature": 0.3},
        )
        self.model_name = model_name
        self.error_messages: List[str] = []

    def _get_base_prompt_template(self) -> str:
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
ステップ2: 情報収集 内容：[検索クリ：日本の首都]
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

    def generate_response(self, user_question: str, previous_steps: Optional[List[ParsedStep]] = None, search_results: Optional[str] = None):
        """Geminiモデルにプロンプトを送信し、応答を生成します。"""
        self.error_messages = [] # Reset errors
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
                response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')) if response.candidates else ""
                logger.info("Received response from Gemini.")

                if self._validate_response_format(response_text):
                    yield response_text
                    return
                
                # フォーマットエラーの場合はリトライ
                self.error_messages.append(f"Geminiの出力形式エラーが発生しました。再試行中 ({attempt + 1}/{MAX_LLM_RETRIES})...")
                logger.warning(f"Gemini response format error (Attempt {attempt + 1}). Retrying...")
                if attempt < MAX_LLM_RETRIES - 1:
                    yield f"Geminiの出力形式エラー。再試行中です ({attempt + 2}/{MAX_LLM_RETRIES})..."
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    self.error_messages.append("Gemini応答形式エラー。最大再試行回数に達しました。")
                    yield "ERROR: LLM_FORMAT_ERROR_MAX_RETRIES"
                    return
            except RetryError as e:
                logger.error(f"Gemini API retry error exceeded: {e}")
                self.error_messages.append(f"Gemini APIリトライエラー超過。再試行中 ({attempt + 1}/{MAX_LLM_RETRIES})...")
                if attempt < MAX_LLM_RETRIES - 1:
                    yield f"Gemini APIエラー（リトライ超過）。再試行中です ({attempt + 2}/{MAX_LLM_RETRIES})..."
                    time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                else:
                    self.error_messages.append("Gemini APIエラー（リトライ超過）。最大再試行回数に達しました。")
                    yield "ERROR: API_RETRY_ERROR_MAX_RETRIES"
                    return
            except GoogleAPIError as e:
                logger.error(f"Gemini API error: {e}")
                error_detail = ""
                if "API key not valid" in str(e) or (hasattr(e, 'grpc_status_code') and e.grpc_status_code == 7):
                    error_detail = "APIキーが無効か、権限がありません。"
                    self.error_messages.append(error_detail)
                    yield "ERROR: API_KEY_INVALID"
                    return
                elif "Rate limit exceeded" in str(e):
                    # クォータエラーの場合、より長い遅延を考慮
                    retry_after_seconds = 0
                    if hasattr(e, 'error_info') and e.error_info and e.error_info.quota_violations:
                        for violation in e.error_info.quota_violations:
                            if hasattr(violation, 'retry_delay') and hasattr(violation.retry_delay, 'seconds'):
                                retry_after_seconds = max(retry_after_seconds, violation.retry_delay.seconds)
                    
                    if retry_after_seconds > 0:
                        error_detail = f"Gemini APIのレート制限に達しました。{retry_after_seconds}秒後に再試行します。"
                        self.error_messages.append(error_detail)
                        logger.warning(error_detail)
                        yield f"Gemini APIのレート制限に達しました。時間をおいて再試行してください ({attempt + 2}/{MAX_LLM_RETRIES})..."
                        if attempt < MAX_LLM_RETRIES - 1:
                            time.sleep(retry_after_seconds + 5) 
                        else:
                            yield "ERROR: API_RATE_LIMIT"
                            return
                    else: 
                        error_detail = "Gemini APIのレート制限に達しました。時間をおいて再試行してください。"
                        self.error_messages.append(error_detail)
                        yield f"Gemini APIのレート制限に達しました。時間をおいて再試行してください ({attempt + 2}/{MAX_LLM_RETRIES})..."
                        if attempt < MAX_LLM_RETRIES - 1:
                            time.sleep(RETRY_DELAY_SECONDS * (attempt + 2)) 
                        else:
                            yield "ERROR: API_RATE_LIMIT"
                            return
                else:
                    error_detail = "不明なAPIエラーが発生しました。"
                    self.error_messages.append(f"Gemini APIエラー: {error_detail}")
                    if attempt < MAX_LLM_RETRIES - 1:
                        yield f"Gemini APIエラーが発生しました。再試行中です ({attempt + 2}/{MAX_LLM_RETRIES})..."
                        time.sleep(RETRY_DELAY_SECONDS)
                    else:
                        yield "ERROR: API_ERROR_MAX_RETRIES"
                        return
            except Exception as e:
                logger.error(f"Unexpected error during Gemini API call: {e}")
                self.error_messages.append(f"予期せぬエラーが発生しました。再試行中 ({attempt + 1}/{MAX_LLM_RETRIES})...")
                if attempt < MAX_LLM_RETRIES - 1:
                    yield f"予期せぬエラーが発生しました。再試行中です ({attempt + 2}/{MAX_LLM_RETRIES})..."
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    self.error_messages.append("予期せぬエラー。最大再試行回数に達しました。")
                    yield "ERROR: UNEXPECTED_ERROR_MAX_RETRIES"
                    return
        self.error_messages.append("不明なエラーにより、応答を取得できませんでした。")
        yield "ERROR: UNKNOWN_FAILURE_AFTER_RETRIES"

    def _format_previous_steps(self, steps: List[ParsedStep]) -> str:
        """過去の思考ステップをプロンプト用に整形します。"""
        if not steps:
            return "なし"
        return "\n".join([f"ステップ{s.id}: {s.type} 内容：{s.raw_content}" + (f" (根拠: {', '.join(map(str, s.basis_ids))})" if s.basis_ids else "") for s in steps])

    def _validate_response_format(self, response_text: str) -> bool:
        """LLMの応答が期待される形式であるか検証します。"""
        has_think_tags = "<think>" in response_text and "</think>" in response_text
        if not response_text.strip():
            logger.warning("Validation failed: Response text is empty.")
            return False
        if not has_think_tags:
            logger.warning("Validation failed: Missing <think> or </think> tags.")
            return False
        return True

    def summarize_text(self, text_to_summarize: str, max_length: int = NODE_SUMMARY_LENGTH) -> str:
        """与えられたテキストを要約します。"""
        if not text_to_summarize or not text_to_summarize.strip():
            return "（空の内容）"
        try:
            prompt = f"以下のテキストを日本語で{max_length}文字以内を目安に、最も重要なポイントを捉えて要約してください。要約のみを出力してください。:\n\n\"{text_to_summarize}\""
            summary_model = genai.GenerativeModel(self.model_name)
            response = summary_model.generate_content(prompt)
            summary = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip() if response.candidates else ""

            if not summary:
                logger.warning(f"Summarization resulted in empty text for: '{text_to_summarize[:50]}...'")
                return text_to_summarize[:max_length] + "..." if len(text_to_summarize) > max_length else text_to_summarize
            return summary if len(summary) <= max_length + 10 else summary[:max_length+5] + "..."
        except Exception as e:
            # summarize_text内で発生したエラーは呼び出し元でハンドリングされるよう、ここでは再raiseまたはエラーメッセージを伝達
            # GraphGeneratorでこのエラーをキャッチし、グラフのエラーメッセージに含める
            raise e # エラーを再スローして呼び出し元でキャッチさせる

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
            self.service = build("customsearch", "v1", developerKey=api_key)
            self.cse_id = cse_id
            self.is_enabled = True
            logger.info("Google Search Handler initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Google Search service: {e}")

    def search(self, query: str, num_results: int = 3) -> Optional[str]:
        """指定されたクエリでGoogle検索を実行し、結果を整形して返します。"""
        if not self.is_enabled or not self.service:
            logger.warning("Google Search service not initialized or disabled. Skipping search.")
            return "検索機能が無効です。APIキーまたはCSE IDを確認してください。"
        try:
            logger.info(f"Performing Google search for: '{query}'")
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
            logger.error(f"Google Search API HTTP error: {e.resp.status} {e._get_reason()} for query '{query}'")
            return f"検索APIエラー: {e._get_reason()}"
        except Exception as e:
            logger.error(f"An unexpected error occurred during Google search for '{query}': {e}")
            return "検索中に予期せぬエラーが発生しました。"


class ThoughtParser:
    """LLMの思考プロセスのテキストを解析し、構造化されたデータに変換するクラス"""
    def parse_llm_output(self, raw_llm_output: str) -> LLMThoughtProcess:
        """LLMの生の出力を解析し、思考ステップと最終回答を抽出します。"""
        result = LLMThoughtProcess(raw_response=raw_llm_output)
        think_match = re.search(r"<think>(.*?)</think>", raw_llm_output, re.DOTALL)

        if not think_match:
            result.error_message = "LLMの応答に<think>タグが見つかりません。"
            result.final_answer = raw_llm_output.strip()
            return result

        think_content = think_match.group(1).strip()
        result.final_answer = raw_llm_output[think_match.end():].strip()

        step_pattern = re.compile(
            r"ステップ\s*(\d+)\s*:\s*([^内容]+?)\s*内容：(.*?)(?:\s*\(根拠:\s*(ステップ\s*\d+(?:,\s*ステップ\s*\d+)*)\s*\))?$",
            re.MULTILINE | re.IGNORECASE
        )
        parsed_steps_in_segment = []
        for match in step_pattern.finditer(think_content):
            try:
                step_id_from_llm = int(match.group(1))
                step_type = match.group(2).strip()
                raw_content = match.group(3).strip()
                basis_str = match.group(4)
                basis_ids = [int(b_id) for b_id in re.findall(r'ステップ\s*(\d+)', basis_str, re.IGNORECASE)] if basis_str else []
                search_query_match = re.search(r"\[検索クエリ：(.*?)\]", raw_content)
                search_query = search_query_match.group(1).strip() if search_query_match else None

                parsed_steps_in_segment.append(ParsedStep(
                    id=step_id_from_llm, type=step_type, raw_content=raw_content,
                    basis_ids=basis_ids, search_query=search_query
                ))
            except Exception as e:
                msg = f"ステップ解析エラー: {match.group(0)} ({e})"
                logger.error(msg)
                result.error_message = (result.error_message or "") + "\n" + msg

        result.steps = parsed_steps_in_segment
        if not result.steps and think_content:
            msg = "思考プロセス(<think>タグ内)が見つかりましたが、有効なステップを解析できませんでした。"
            logger.warning(msg + f" Content: {think_content[:100]}...")
            result.error_message = (result.error_message or "") + "\n" + msg
        return result


# (他のクラスやimport文は元のまま)
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

class GraphGenerator:
    """
    LLMの思考プロセスをグラフとして可視化するクラス (最終・安定版)
    - ノードを横長の長方形で表示
    - カスタム関数による階層的レイアウトを使用
    - 右上に凡例を表示
    - 矢印を強調し、ノードとの重なりを回避 (バージョン非依存の方法で実装)
    """
    def __init__(self, llm_handler: 'GeminiHandler', installed_japanese_font: Optional[str]):
        # ... (変更なし)
        self.llm_handler = llm_handler
        self.error_messages_for_graph: List[str] = []
        self.font_properties = None

        if installed_japanese_font:
            try:
                self.font_properties = fm.FontProperties(family=installed_japanese_font)
                logger.info(f"GraphGenerator: Using Matplotlib-configured font: '{installed_japanese_font}'.")
            except Exception as e:
                logger.error(f"GraphGenerator: Failed to load font property for '{installed_japanese_font}': {e}")
                self.error_messages_for_graph.append(f"設定済み日本語フォント '{installed_japanese_font}' の読み込みに失敗しました。")
        else:
            self.error_messages_for_graph.append("適切な日本語フォントが見つかりませんでした。グラフの文字が正しく表示されない可能性があります。")
            logger.warning("GraphGenerator: No specific font set. Relying on Matplotlib's default font mechanisms.")

    def _summarize_step_contents(self, steps: List['ParsedStep'], progress_fn=None):
        # ... (変更なし)
        logger.info(f"Summarizing contents for {len(steps)} steps...")
        if not steps: return
        for i, step in enumerate(steps):
            if progress_fn:
                progress_fn((i + 1) / len(steps), f"ステップ {step.id} の内容を要約中 ({i+1}/{len(steps)})...")
            if not step.summarized_content:
                try:
                    step.summarized_content = self.llm_handler.summarize_text(step.raw_content, max_length=NODE_SUMMARY_LENGTH)
                except Exception as e:
                    error_msg = f"ステップS{step.id}の要約中にエラーが発生しました: {e}"
                    logger.error(error_msg)
                    self.error_messages_for_graph.append(error_msg)
                    step.summarized_content = step.raw_content[:NODE_SUMMARY_LENGTH] + "..." if len(step.raw_content) > NODE_SUMMARY_LENGTH else step.raw_content
            time.sleep(0.05)
        logger.info("Summarization complete.")
        
    def _custom_hierarchical_layout(self, G, root_node=0, vertical_gap=0.5, horizontal_gap=0.5):
        # ... (変更なし)
        if not root_node in G:
            logger.warning(f"Root node {root_node} not in graph. Cannot create hierarchical layout.")
            return None
        levels = {root_node: 0}
        nodes_at_level = {0: [root_node]}
        queue = [root_node]
        visited = {root_node}
        max_level = 0
        while queue:
            parent = queue.pop(0)
            parent_level = levels[parent]
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
        all_nodes = set(G.nodes())
        unvisited = all_nodes - visited
        if unvisited:
            unvisited_level = max_level + 1
            nodes_at_level[unvisited_level] = list(unvisited)
            for node in unvisited:
                levels[node] = unvisited_level
        pos = {}
        for level, nodes in nodes_at_level.items():
            num_nodes = len(nodes)
            level_width = (num_nodes - 1) * horizontal_gap
            x_start = -level_width / 2
            sorted_nodes = sorted(nodes)
            for i, node in enumerate(sorted_nodes):
                x = x_start + i * horizontal_gap
                y = -level * vertical_gap
                pos[node] = (x, y)
        return pos

    def create_thinking_graph(self, user_question: str, all_steps: List['ParsedStep'], final_answer_text: str, progress_fn=None) -> Optional[plt.Figure]:
        # ... (グラフ構築部分は変更なし) ...
        self.error_messages_for_graph = []
        llm_errors = self.llm_handler.get_error_messages()
        if llm_errors: self.error_messages_for_graph.extend(llm_errors)
        self._summarize_step_contents(all_steps, progress_fn)
        G = nx.DiGraph()
        QUESTION_NODE_ID = 0
        try:
            question_summary = self.llm_handler.summarize_text(user_question, max_length=NODE_SUMMARY_LENGTH + 15)
        except Exception as e:
            error_msg = f"質問の要約中にエラーが発生しました: {e}"; logger.error(error_msg); self.error_messages_for_graph.append(error_msg)
            question_summary = user_question[:NODE_SUMMARY_LENGTH + 15] + "..."
        G.add_node(QUESTION_NODE_ID, label=f"質問:\n{question_summary}", type="question", color="skyblue")
        valid_step_ids_in_graph = {QUESTION_NODE_ID}
        current_prev_node_id = QUESTION_NODE_ID
        if all_steps:
            max_existing_id = max(s.id for s in all_steps) if all_steps else 0
            for step in sorted(all_steps, key=lambda s: s.id):
                node_graph_id = step.id
                if node_graph_id in G: continue
                label_text = f"S{step.id}: {step.type}\n{step.summarized_content}"
                G.add_node(node_graph_id, label=label_text, type="ai_step", color="khaki")
                valid_step_ids_in_graph.add(node_graph_id)
                if current_prev_node_id in G: G.add_edge(current_prev_node_id, node_graph_id, type="sequential", style="solid", color="gray")
                current_prev_node_id = node_graph_id
            try:
                answer_summary = self.llm_handler.summarize_text(final_answer_text, max_length=NODE_SUMMARY_LENGTH + 15)
            except Exception as e:
                error_msg = f"最終回答の要約中にエラーが発生しました: {e}"; logger.error(error_msg); self.error_messages_for_graph.append(error_msg)
                answer_summary = final_answer_text[:NODE_SUMMARY_LENGTH + 15] + "..."
            final_answer_node_id = max_existing_id + 1
            G.add_node(final_answer_node_id, label=f"最終回答:\n{answer_summary}", type="final_answer", color="lightgreen")
            valid_step_ids_in_graph.add(final_answer_node_id)
            if current_prev_node_id != QUESTION_NODE_ID and current_prev_node_id in G: G.add_edge(current_prev_node_id, final_answer_node_id, type="sequential", style="solid", color="gray")
            elif QUESTION_NODE_ID in G: G.add_edge(QUESTION_NODE_ID, final_answer_node_id, type="sequential", style="solid", color="gray")
            for step in all_steps:
                target_node_id = step.id
                for basis_id in step.basis_ids:
                    if basis_id in valid_step_ids_in_graph and target_node_id in valid_step_ids_in_graph and basis_id != target_node_id:
                        if not G.has_edge(basis_id, target_node_id) or G.get_edge_data(basis_id, target_node_id).get('type') != 'sequential': G.add_edge(basis_id, target_node_id, type="basis", style="dashed", color="purple")
        elif final_answer_text and QUESTION_NODE_ID in G:
            try:
                answer_summary = self.llm_handler.summarize_text(final_answer_text, max_length=NODE_SUMMARY_LENGTH + 15)
            except Exception as e:
                error_msg = f"最終回答の要約中にエラーが発生しました: {e}"; logger.error(error_msg); self.error_messages_for_graph.append(error_msg)
                answer_summary = final_answer_text[:NODE_SUMMARY_LENGTH + 15] + "..."
            final_answer_node_id = QUESTION_NODE_ID + 1
            G.add_node(final_answer_node_id, label=f"最終回答:\n{answer_summary}", type="final_answer", color="lightgreen")
            G.add_edge(QUESTION_NODE_ID, final_answer_node_id, type="sequential", style="solid", color="gray")
        else:
            self.error_messages_for_graph.append("有効な思考プロセスまたは回答が得られませんでした。グラフは生成できません。"); return None
        if not G.nodes():
            self.error_messages_for_graph.append("グラフを生成するためのノードがありません。"); return None

        fig, ax = plt.subplots(figsize=(min(20, max(12, G.number_of_nodes() * 2.0)), min(15, max(8, G.number_of_nodes() * 1.0))))
        
        pos = None
        try:
            logger.info("Using custom hierarchical layout.")
            pos = self._custom_hierarchical_layout(G, root_node=QUESTION_NODE_ID)
            if pos is None: raise ValueError("Custom layout returned None.")
        except Exception as e:
            logger.warning(f"Custom hierarchical layout failed ('{e}'). Falling back to spring layout.")
            try:
                pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
            except Exception as e2:
                logger.error(f"Fallback spring_layout also failed: {e2}. Using random_layout.")
                pos = nx.random_layout(G, seed=42)

        if pos is None:
            self.error_messages_for_graph.append("グラフのレイアウト計算に失敗しました。"); plt.close(fig); return None

        labels = {n: d['label'] for n, d in G.nodes(data=True)}
        
        # ★★★★★ ここからが変更箇所 ★★★★★
        # nx.draw_networkx_edgesを完全にやめ、FancyArrowPatchで矢印を手動描画する

        # 1. エッジを一本ずつ描画
        for u, v, data in G.edges(data=True):
            start_pos = pos[u]
            end_pos = pos[v]
            
            # 線の種類に応じてスタイルを設定
            if data.get('type') == 'sequential':
                arrow_style = mpatches.ArrowStyle.Simple(head_length=15, head_width=10, tail_width=2.5)
                arrow = mpatches.FancyArrowPatch(
                    start_pos, end_pos,
                    arrowstyle=arrow_style,
                    color="black",
                    shrinkA=0, shrinkB=20, # 終点を20ポイント縮める
                    mutation_scale=1,
                    connectionstyle='arc3,rad=0.05'
                )
            elif data.get('type') == 'basis':
                arrow_style = mpatches.ArrowStyle.Simple(head_length=10, head_width=6, tail_width=0.8)
                arrow = mpatches.FancyArrowPatch(
                    start_pos, end_pos,
                    arrowstyle=arrow_style,
                    color="purple",
                    linestyle='dashed',
                    shrinkA=0, shrinkB=20, # 終点を20ポイント縮める
                    mutation_scale=1,
                    connectionstyle='arc3,rad=0.15'
                )
            else:
                continue # 未知のタイプは描画しない
            
            ax.add_patch(arrow)

        # 2. ノードを描画 (変更なし)
        node_width, node_height = 0.4, 0.2
        for node in G.nodes():
            x_center, y_center = pos[node]
            node_color = G.nodes[node]['color']
            x_corner, y_corner = x_center - node_width / 2, y_center - node_height / 2
            rect = plt.Rectangle((x_corner, y_corner), node_width, node_height, facecolor=node_color, alpha=0.95, transform=ax.transData)
            ax.add_patch(rect)
        
        # 3. ラベルを描画 (変更なし)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax, font_weight='bold', verticalalignment='center')
        
        # 4. 凡例を描画 (変更なし)
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
        ax.legend(handles=all_handles, loc='upper right', fontsize='small', title="凡例", prop=self.font_properties)
        
        # ★★★★★ 変更箇所ここまで ★★★★★

        ax.set_title("AI思考連鎖の可視化 (Gemini)", fontsize=16, fontweight='bold', fontproperties=self.font_properties if self.font_properties else None)
        
        ax.autoscale()
        ax.margins(0.1) 
        plt.axis('off')
        plt.tight_layout(pad=1.0)
        return fig

    def get_error_messages_for_graph(self) -> List[str]:
        # ... (変更なし)
        return self.error_messages_for_graph

    def get_error_messages_html(self) -> str:
        # ... (変更なし)
        if not self.error_messages_for_graph: return ""
        escaped_messages = [msg.replace("&", "&").replace("<", "<").replace(">", ">") for msg in self.error_messages_for_graph]
        html = "<div style='color: red; border: 1px solid red; padding: 10px; margin-top: 10px;'>" \
               "<strong>グラフに関する注意:</strong><ul>"
        for msg in escaped_messages:
            html += f"<li>{msg}</li>"
        html += "</ul></div>"
        return html


class AISystem:
    """AI思考システムの中核を担うクラス。LLMと検索機能を統合し、思考プロセスを管理します。"""
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        # 変数名のタイポ修正 Google Search_api_key -> Google Search_api_key
        self.Google_Search_api_key = os.getenv("GOOGLE_API_KEY") 
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")

        if not self.gemini_api_key:
            logger.error("CRITICAL: GEMINI_API_KEY environment variable not set.")

        self.llm_handler = GeminiHandler(api_key=self.gemini_api_key)
        self.search_handler = GoogleSearchHandler(api_key=self.Google_Search_api_key, cse_id=self.google_cse_id)
        self.parser = ThoughtParser()
        self.graph_generator = GraphGenerator(llm_handler=self.llm_handler, installed_japanese_font=INSTALLED_JAPANESE_FONT)

    def _append_and_renumber_steps(self, existing_steps: List[ParsedStep], new_steps_from_llm: List[ParsedStep]) -> List[ParsedStep]:
        """既存のステップに新しいステップを追加し、重複を排除しつつIDを振り直します。"""
        updated_all_steps = list(existing_steps)
        existing_signatures = {(step.type, step.raw_content) for step in existing_steps}
        
        current_max_id = max(s.id for s in existing_steps) if existing_steps else 0

        for new_step in new_steps_from_llm:
            if (new_step.type, new_step.raw_content) in existing_signatures:
                logger.info(f"Skipping duplicate step (content match): Type='{new_step.type}', Content='{new_step.raw_content[:30]}...'")
                continue
            
            current_max_id += 1
            new_step.id = current_max_id
            updated_all_steps.append(new_step)
            existing_signatures.add((new_step.type, new_step.raw_content))
            
        return updated_all_steps

    def process_question_iterations(self, user_question: str, progress_fn):
        """
        ユーザーの質問に対するAIの思考プロセスを反復的に実行し、
        最終回答、思考グラフ、エラーメッセージを生成します。
        """
        if not self.gemini_api_key:
            yield ("Gemini APIキーが設定されていません。", None, "<p style='color:red; font-weight:bold;'>Gemini APIキーが設定されていません。</p>")
            return

        progress_fn(0, desc="AI(Gemini)が思考を開始しました...")
        accumulated_steps: List[ParsedStep] = []
        accumulated_final_answer: str = ""
        current_raw_llm_output_segment = ""
        search_iteration_count = 0
        overall_error_messages: List[str] = [] # 全体のエラーメッセージを収集

        # 1. 最初のLLM応答
        llm_response_generator = self.llm_handler.generate_response(user_question)
        for update_message in llm_response_generator:
            if "ERROR:" in update_message:
                current_raw_llm_output_segment = update_message
                break
            if update_message.endswith("..."):
                progress_fn(0.1, desc=update_message)
            else:
                current_raw_llm_output_segment = update_message

        overall_error_messages.extend(self.llm_handler.get_error_messages())

        if "ERROR:" in current_raw_llm_output_segment:
            error_msg = f"AI処理エラー: {current_raw_llm_output_segment.replace('ERROR: ', '')}"
            yield (error_msg, None, f"<p style='color:red; font-weight:bold;'>{error_msg}</p>"); return

        parsed_segment_obj = self.parser.parse_llm_output(current_raw_llm_output_segment)
        accumulated_steps = self._append_and_renumber_steps(accumulated_steps, parsed_segment_obj.steps)
        accumulated_final_answer = parsed_segment_obj.final_answer
        if parsed_segment_obj.error_message:
            overall_error_messages.append(f"パーサーからの通知: {parsed_segment_obj.error_message}")

        last_parsed_segment_steps = parsed_segment_obj.steps

        # 2. 検索とLLMの反復処理
        while any(s.type == "情報収集" and s.search_query for s in last_parsed_segment_steps) and \
              search_iteration_count < MAX_SEARCH_ITERATIONS:
            search_step = next((s for s in last_parsed_segment_steps if s.type == "情報収集" and s.search_query), None)
            if not search_step:
                break

            search_iteration_count += 1
            query = search_step.search_query
            prog_frac = min(0.3 + search_iteration_count * 0.1, 0.9)
            progress_fn(prog_frac, desc=f"'{query[:30]}...'検索中({search_iteration_count}/{MAX_SEARCH_ITERATIONS})...")
            
            s_results = self.search_handler.search(query) or "情報取得失敗。"
            
            prog_frac = min(0.4 + search_iteration_count * 0.1, 0.9)
            progress_fn(prog_frac, desc="検索結果を元にAI(Gemini)が再考中...")

            llm_response_generator = self.llm_handler.generate_response(user_question, accumulated_steps, s_results)
            current_raw_llm_output_segment = ""
            for update_message in llm_response_generator:
                if "ERROR:" in update_message:
                    current_raw_llm_output_segment = update_message
                    break
                if update_message.endswith("..."):
                    prog_frac_inner = min(0.5 + search_iteration_count * 0.1, 0.9)
                    progress_fn(prog_frac_inner, desc=update_message)
                else:
                    current_raw_llm_output_segment = update_message
            
            overall_error_messages.extend(self.llm_handler.get_error_messages())

            if "ERROR:" in current_raw_llm_output_segment:
                error_msg = f"AI処理エラー(検索後): {current_raw_llm_output_segment.replace('ERROR: ', '')}"
                overall_error_messages.append(error_msg)
                break

            parsed_segment_obj = self.parser.parse_llm_output(current_raw_llm_output_segment)
            accumulated_steps = self._append_and_renumber_steps(accumulated_steps, parsed_segment_obj.steps)
            if parsed_segment_obj.final_answer.strip():
                accumulated_final_answer = parsed_segment_obj.final_answer
            if parsed_segment_obj.error_message:
                overall_error_messages.append(f"パーサー通知(検索後): {parsed_segment_obj.error_message}")
            
            last_parsed_segment_steps = parsed_segment_obj.steps
            if not any(s.type == "情報収集" and s.search_query for s in last_parsed_segment_steps):
                break
        
        # 最終回答の調整
        if not accumulated_final_answer.strip() and accumulated_steps and not ("ERROR:" in current_raw_llm_output_segment):
            accumulated_final_answer = "思考プロセスに基づき検討しましたが、最終回答の明示的出力はありませんでした。グラフを参照してください。"
        elif not accumulated_final_answer.strip() and not accumulated_steps and not ("ERROR:" in current_raw_llm_output_segment):
            accumulated_final_answer = "AIからの有効な応答がありませんでした。"

        progress_fn(0.8, desc="思考グラフを生成中です...")
        summary_prog_lambda = lambda p, d: progress_fn(0.8 + p * 0.15, desc=d)
        
        fig = self.graph_generator.create_thinking_graph(user_question, accumulated_steps, accumulated_final_answer, summary_prog_lambda)
        
        graph_image_pil = None
        if fig:
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                buf.seek(0)
                graph_image_pil = Image.open(buf)
            except Exception as e:
                logger.error(f"Error converting graph to PIL Image or saving: {e}")
                overall_error_messages.append("警告: グラフ画像の生成または表示に失敗しました。")
        
        overall_error_messages.extend(self.graph_generator.get_error_messages_for_graph())
        graph_errors_html = self._format_overall_error_messages(overall_error_messages)

        progress_fn(1.0, desc="完了")
        yield (accumulated_final_answer, graph_image_pil, graph_errors_html)

    def _format_overall_error_messages(self, messages: List[str]) -> str:
        """全てのエラーメッセージをHTML形式で整形します。"""
        if not messages:
            return ""
        unique_messages = sorted(list(set(messages)))
        html_content = "<div style='color: red; border: 1px solid red; padding: 10px; margin-top: 10px;'>"\
                       "<strong>処理に関する注意・エラー:</strong><ul>"
        for msg in unique_messages:
            html_content += f"<li>{msg}</li>"
        html_content += "</ul></div>"
        return html_content


def create_gradio_interface(system: AISystem):
    """Gradio UIを作成します。"""
    with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)) as demo:
        gr.Markdown("# 🧠 AI思考連鎖可視化システム (Gemini版)")
        gr.Markdown("ユーザーの質問に対し、AI(Gemini)が思考プロセスを段階的に示しながら回答します。思考の連鎖はネットワーク図として可視化されます。")

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(label="質問を入力してください:", placeholder="例: 日本のAI技術の最新トレンドとその課題は何ですか？", lines=3, show_copy_button=True)
                submit_button = gr.Button("質問を送信する", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown(f"### 設定（参考）\n"
                            f"- LLM Model: `{system.llm_handler.model_name}`\n"
                            f"- Google検索: `{'有効' if system.search_handler.is_enabled else '無効 (APIキー/CSE ID未設定)'}`\n"
                            f"- 最大検索回数: `{MAX_SEARCH_ITERATIONS}`\n"
                            f"- LLM再試行回数: `{MAX_LLM_RETRIES}`\n"
                            f"- 使用フォント(グラフ): `{INSTALLED_JAPANESE_FONT if INSTALLED_JAPANESE_FONT else '(日本語フォントが見つかりません)'}`")
        gr.Markdown("---")
        gr.Markdown("## 🤖 AIの回答と思考プロセス")
        with gr.Row():
            with gr.Column(scale=1):
                answer_output = gr.Markdown(label="AIの最終回答")
                graph_errors_output = gr.HTML(label="処理に関する通知・エラー")
            with gr.Column(scale=2):
                graph_output = gr.Image(label="思考プロセスネットワーク図", type="pil", interactive=False, show_download_button=True)

        def submit_fn_wrapper_for_gradio(question, progress=gr.Progress(track_tqdm=True)):
            if not question.strip():
                return ("質問が入力されていません。", None, "<p style='color:orange;'>質問を入力してください。</p>")
            
            final_result_tuple = (None, None, None)
            for result_tuple in system.process_question_iterations(question, progress):
                final_result_tuple = result_tuple
            return final_result_tuple

        submit_button.click(fn=submit_fn_wrapper_for_gradio, inputs=[question_input], outputs=[answer_output, graph_output, graph_errors_output])
        gr.Examples(examples=[["日本のAI技術の最新トレンドとその社会への影響について教えてください。"],["太陽光発電のメリット・デメリットを整理し、今後の展望を予測してください。"]], inputs=[question_input], label="質問例")
        gr.Markdown("---")
        gr.Markdown("© 2025 ユニークAIシステム構築道場 G9 (プロトタイプ - Gemini版). 回答生成には時間がかかることがあります。")
    return demo

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("CRITICAL: GEMINI_API_KEY environment variable not set. Please set it to run the application.")

    ai_system_instance = AISystem()
    gradio_app_interface = create_gradio_interface(ai_system_instance)
    gradio_app_interface.launch(server_name="localhost", server_port=7860, share=True)