# app_refactored.py

import os
import re
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # サーバー環境で必須
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, RetryError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import gradio as gr
import io
from PIL import Image
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# --- Configuration & Setup ---
load_dotenv()  # .envファイルからAPIキーを読み込み
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# .envファイルからAPIキーを取得する関数
def get_env_api_keys() -> Dict[str, Optional[str]]:
    """環境変数からAPIキーを取得"""
    return {
        'gemini_api_key': os.getenv('GEMINI_API_KEY'),
        'google_api_key': os.getenv('GOOGLE_API_KEY'),
        'google_cse_id': os.getenv('GOOGLE_CSE_ID')
    }


# --- Constants ---
class Config:
    MAX_LLM_RETRIES = 3
    NODE_SUMMARY_LENGTH = 25
    RETRY_DELAY_SECONDS = 5
    MAX_SEARCH_ITERATIONS = 5
    DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
    
    # デフォルトのシステムプロンプト
    DEFAULT_SYSTEM_PROMPT = """ユーザーの質問に対して、あなたの思考プロセスを段階的に説明しながら回答を生成してください。
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
また、各ステップは15ステップ以下で記述するようにしてください。

以下は出力形式の厳格な例です：
質問：日本の首都はどこですか？
<think>
ステップ1: 問題定義 内容：ユーザーは日本の首都について質問している。
ステップ2: 情報収集 内容：[検索クエリ：日本 首都]
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
また、あいまいな質問には詳細に指定するように返答してください。

ユーザーの質問：{user_question}"""


# --- Utility Functions ---
def setup_japanese_font_for_matplotlib() -> Optional[str]:
    """
    システムにインストールされている利用可能な日本語フォントを探し、Matplotlibに設定を試みます。
    Hugging Face Spaces環境を想定し、'Noto Sans CJK JP'を優先します。
    """
    font_candidates = [
        'Noto Sans CJK JP', 'IPAexGothic', 'Yu Gothic', 'Hiragino Sans', 'MS Gothic', 'Meiryo'
    ]
    
    try:
        if hasattr(fm, '_rebuild'):
            fm._rebuild()
    except Exception as e:
        logger.warning(f"Could not rebuild Matplotlib font cache. This may be normal. Error: {e}")

    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font_name in font_candidates:
        if font_name in available_fonts:
            try:
                plt.rcParams['font.family'] = font_name
                current_sans_serif = plt.rcParams.get('font.sans-serif', [])
                if 'DejaVu Sans' in current_sans_serif:
                    current_sans_serif.remove('DejaVu Sans')
                plt.rcParams['font.sans-serif'] = [font_name] + current_sans_serif
                logger.info(f"Found and set Japanese font for Matplotlib: '{font_name}'")
                return font_name
            except Exception as e:
                logger.debug(f"Font '{font_name}' found but failed to set: {e}")
    
    logger.warning("No suitable Japanese font found. Graph labels may not display correctly.")
    return None


INSTALLED_JAPANESE_FONT = setup_japanese_font_for_matplotlib()


# --- Data Classes ---
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


# --- Core Classes ---
class GeminiHandler:
    """Gemini APIとの対話を管理するクラス"""
    
    def __init__(self, api_key: str, model_name: str = Config.DEFAULT_MODEL_NAME, 
                 system_prompt: str = Config.DEFAULT_SYSTEM_PROMPT):
        if not api_key:
            raise ValueError("Gemini API key is not set.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"temperature": 0.3},
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.error_messages: List[str] = []

    def update_system_prompt(self, new_prompt: str):
        """システムプロンプトを更新"""
        self.system_prompt = new_prompt
        logger.info("System prompt updated")

    def generate_response(self, user_question: str, previous_steps: Optional[List[ParsedStep]] = None, 
                         search_results: Optional[str] = None):
        """LLMからの応答を生成"""
        self.error_messages = []
        previous_steps_str = self._format_previous_steps(previous_steps) if previous_steps else "なし"
        search_results_str = search_results if search_results else "なし"
        
        prompt = self.system_prompt.format(
            user_question=user_question,
            previous_steps_str=previous_steps_str,
            search_results_str=search_results_str
        )

        for attempt in range(Config.MAX_LLM_RETRIES):
            try:
                logger.info(f"Sending request to Gemini (Attempt {attempt + 1}/{Config.MAX_LLM_RETRIES}) Model: {self.model_name}")
                response = self.model.generate_content(prompt)
                response_text = "".join(
                    part.text for part in response.candidates[0].content.parts 
                    if hasattr(part, 'text')
                ) if response.candidates else ""
                
                if self._validate_response_format(response_text):
                    yield response_text
                    return
                
                error_msg = f"Geminiの出力形式エラー。再試行中({attempt + 1})。"
                self.error_messages.append(error_msg)
                logger.warning(f"Gemini response format error (Attempt {attempt + 1}). Retrying...")
                
                if attempt < Config.MAX_LLM_RETRIES - 1:
                    yield f"Geminiの出力形式エラー。再試行中です..."
                    time.sleep(Config.RETRY_DELAY_SECONDS)
                else:
                    yield "ERROR: LLM_FORMAT_ERROR_MAX_RETRIES"
                    return
                    
            except (GoogleAPIError, RetryError) as e:
                logger.error(f"Gemini API error (Attempt {attempt + 1}): {e}")
                self.error_messages.append(f"Gemini APIエラー({e})。再試行中...")
                
                if "API key not valid" in str(e):
                    yield "ERROR: API_KEY_INVALID"
                    return
                    
                if attempt < Config.MAX_LLM_RETRIES - 1:
                    yield f"Gemini APIエラー。再試行中です..."
                    time.sleep(Config.RETRY_DELAY_SECONDS * (attempt + 2))
                else:
                    yield "ERROR: API_ERROR_MAX_RETRIES"
                    return
                    
            except Exception as e:
                logger.error(f"Unexpected error during Gemini API call: {e}")
                self.error_messages.append("予期せぬエラー。再試行中...")
                
                if attempt < Config.MAX_LLM_RETRIES - 1:
                    yield "予期せぬエラー。再試行中です..."
                    time.sleep(Config.RETRY_DELAY_SECONDS)
                else:
                    yield "ERROR: UNEXPECTED_ERROR_MAX_RETRIES"
                    return
                    
        yield "ERROR: UNKNOWN_FAILURE_AFTER_RETRIES"

    def _format_previous_steps(self, steps: List[ParsedStep]) -> str:
        """前のステップを文字列形式にフォーマット"""
        if not steps:
            return "なし"
        
        return "\n".join([
            f"ステップ{s.id}: {s.type} 内容：{s.raw_content}" + 
            (f" (根拠: {', '.join(map(str, s.basis_ids))})" if s.basis_ids else "")
            for s in steps
        ])

    def _validate_response_format(self, response_text: str) -> bool:
        """レスポンス形式の検証"""
        if not response_text.strip():
            return False
        has_think_tags = "<think>" in response_text and "</think>" in response_text
        return has_think_tags

    def summarize_text(self, text_to_summarize: str, max_length: int = Config.NODE_SUMMARY_LENGTH) -> str:
        """テキストを要約"""
        if not text_to_summarize or not text_to_summarize.strip():
            return "（空の内容）"
        
        try:
            prompt = f"以下のテキストを日本語で{max_length}文字以内を目安に要約してください:\n\n\"{text_to_summarize}\""
            summary_model = genai.GenerativeModel(self.model_name)
            response = summary_model.generate_content(prompt)
            summary = "".join(
                part.text for part in response.candidates[0].content.parts 
                if hasattr(part, 'text')
            ).strip() if response.candidates else ""
            
            return summary if summary else text_to_summarize[:max_length] + "..."
        except Exception as e:
            raise e

    def get_error_messages(self) -> List[str]:
        """エラーメッセージを取得"""
        return self.error_messages


class GoogleSearchHandler:
    """Google検索APIの管理クラス"""
    
    def __init__(self, api_key: Optional[str], cse_id: Optional[str]):
        self.service = None
        self.is_enabled = False
        
        if not api_key or not cse_id:
            logger.warning("Google API key or CSE ID is not set. Search functionality will be disabled.")
            return
        
        try:
            self.service = build("customsearch", "v1", developerKey=api_key, cache_discovery=False)
            self.cse_id = cse_id
            self.is_enabled = True
            logger.info("Google Search Handler initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Google Search service: {e}")

    def search(self, query: str, num_results: int = 3) -> Optional[str]:
        """検索クエリを実行"""
        if not self.is_enabled or not self.service:
            return "検索機能が無効です。"
        
        try:
            res = self.service.cse().list(
                q=query, cx=self.cse_id, num=num_results, lr='lang_ja'
            ).execute()
            
            if 'items' not in res or not res['items']:
                return "検索結果なし。"
            
            search_results_text = "検索結果:\n" + "\n\n".join([
                f"{i+1}. {item.get('title', '')}\n  {item.get('snippet', '').replace(chr(10), ' ')}"
                for i, item in enumerate(res['items'])
            ])
            
            return search_results_text.strip()
        except HttpError as e:
            return f"検索APIエラー: {e._get_reason()}"
        except Exception as e:
            return f"検索中に予期せぬエラー: {e}"


class ThoughtParser:
    """LLM出力の解析クラス"""
    
    def parse_llm_output(self, raw_llm_output: str) -> LLMThoughtProcess:
        """LLMの出力を解析してステップに分解"""
        result = LLMThoughtProcess(raw_response=raw_llm_output)
        
        think_match = re.search(r"<think>(.*?)</think>", raw_llm_output, re.DOTALL)
        if not think_match:
            result.error_message = "LLMの応答に<think>タグが見つかりません。"
            result.final_answer = raw_llm_output.strip()
            return result
        
        think_content = think_match.group(1).strip()
        result.final_answer = raw_llm_output[think_match.end():].strip()
        
        step_pattern = re.compile(
            r"ステップ\s*(\d+)\s*:\s*([^内容]+?)\s*内容：(.*?)(?:\s*\(根拠:\s*(?:ステップ\s*\d+(?:,\s*ステップ\s*\d+)*)\s*\))?$",
            re.MULTILINE | re.IGNORECASE
        )
        
        for match in step_pattern.finditer(think_content):
            try:
                raw_content_full = match.group(3).strip()
                basis_match = re.search(r'\(根拠:\s*(.*?)\)', match.group(0), re.IGNORECASE)
                basis_ids = [
                    int(b_id) for b_id in re.findall(r'\d+', basis_match.group(1))
                ] if basis_match else []
                
                search_query_match = re.search(r"\[検索クエリ：(.*?)\]", raw_content_full, re.IGNORECASE)
                search_query = search_query_match.group(1).strip() if search_query_match else None
                
                result.steps.append(ParsedStep(
                    id=int(match.group(1)),
                    type=match.group(2).strip(),
                    raw_content=raw_content_full,
                    basis_ids=basis_ids,
                    search_query=search_query
                ))
            except Exception as e:
                result.error_message = f"ステップ解析エラー: {match.group(0)} ({e})"
        
        return result


class GraphGenerator:
    """グラフ生成クラス"""
    
    def __init__(self, llm_handler: GeminiHandler, installed_japanese_font: Optional[str]):
        self.llm_handler = llm_handler
        self.error_messages_for_graph: List[str] = []
        self.font_properties = fm.FontProperties(family=installed_japanese_font) if installed_japanese_font else None

    def _summarize_step_contents(self, steps: List[ParsedStep], progress_fn=None):
        """ステップ内容を要約"""
        if not steps:
            return
        
        total_steps = len(steps)
        for i, step in enumerate(steps):
            if progress_fn:
                progress_fn((i + 1) / total_steps, f"ステップ {step.id} 要約中 ({i+1}/{total_steps})...")
            
            if not step.summarized_content:
                try:
                    step.summarized_content = self.llm_handler.summarize_text(step.raw_content)
                except Exception as e:
                    self.error_messages_for_graph.append(f"S{step.id}要約エラー: {e}")
                    step.summarized_content = step.raw_content[:Config.NODE_SUMMARY_LENGTH] + "..."

    def _custom_hierarchical_layout(self, G, root_node=0, vertical_gap=0.8, horizontal_gap=1.5):
        """カスタム階層レイアウト"""
        if root_node not in G:
            return None
        
        levels, nodes_at_level = {root_node: 0}, {0: [root_node]}
        queue, visited = [root_node], {root_node}
        max_level = 0
        
        while queue:
            parent = queue.pop(0)
            children = sorted(list(G.successors(parent)))
            for child in children:
                if child not in visited:
                    visited.add(child)
                    levels[child] = levels[parent] + 1
                    if levels[child] not in nodes_at_level:
                        nodes_at_level[levels[child]] = []
                    nodes_at_level[levels[child]].append(child)
                    queue.append(child)
                    max_level = max(max_level, levels[child])
        
        pos = {}
        for level, nodes in nodes_at_level.items():
            level_width = (len(nodes) - 1) * horizontal_gap
            x_start = -level_width / 2
            for i, node in enumerate(sorted(nodes)):
                pos[node] = (x_start + i * horizontal_gap, -level * vertical_gap)
        
        return pos

    def create_thinking_graph(self, user_question: str, all_steps: List[ParsedStep], 
                            final_answer_text: str, progress_fn=None) -> Optional[plt.Figure]:
        """思考プロセスのグラフを生成"""
        self.error_messages_for_graph = []
        
        if self.llm_handler.get_error_messages():
            self.error_messages_for_graph.extend(self.llm_handler.get_error_messages())
        
        self._summarize_step_contents(all_steps, progress_fn)
        
        G, QUESTION_NODE_ID = nx.DiGraph(), 0
        
        # 質問ノードを追加
        try:
            question_summary = self.llm_handler.summarize_text(user_question, 40)
            G.add_node(QUESTION_NODE_ID, label=f"質問:\n{question_summary}", 
                      type="question", color="skyblue")
        except Exception as e:
            self.error_messages_for_graph.append(f"質問要約エラー: {e}")
            G.add_node(QUESTION_NODE_ID, label=f"質問:\n{user_question[:40]}...", 
                      type="question", color="skyblue")
        
        valid_ids = {QUESTION_NODE_ID}
        prev_id = QUESTION_NODE_ID
        max_id = max((s.id for s in all_steps), default=0)
        
        # ステップノードを追加
        for step in sorted(all_steps, key=lambda s: s.id):
            G.add_node(step.id, 
                      label=f"S{step.id}: {step.type}\n{step.summarized_content}",
                      type="ai_step", color="khaki")
            valid_ids.add(step.id)
            
            if G.has_node(prev_id):
                G.add_edge(prev_id, step.id, type="sequential")
            prev_id = step.id
        
        # 最終回答ノードを追加
        final_id = max_id + 1
        try:
            answer_summary = self.llm_handler.summarize_text(final_answer_text, 40)
            G.add_node(final_id, label=f"最終回答:\n{answer_summary}", 
                      type="final_answer", color="lightgreen")
        except Exception as e:
            self.error_messages_for_graph.append(f"回答要約エラー: {e}")
            G.add_node(final_id, label=f"最終回答:\n{final_answer_text[:40]}...", 
                      type="final_answer", color="lightgreen")
        
        if G.has_node(prev_id):
            G.add_edge(prev_id, final_id, type="sequential")
        
        # 根拠関係のエッジを追加
        for step in all_steps:
            for basis_id in step.basis_ids:
                if basis_id in valid_ids and step.id in valid_ids:
                    G.add_edge(basis_id, step.id, type="basis")

        if not G.nodes():
            self.error_messages_for_graph.append("グラフを生成するためのノードがありません。")
            return None
        
        # グラフの描画
        fig, ax = plt.subplots(figsize=(
            min(20, max(12, G.number_of_nodes() * 2.2)),
            min(15, max(8, G.number_of_nodes() * 1.2))
        ))
        
        pos = self._custom_hierarchical_layout(G) or nx.spring_layout(G, seed=42)
        
        # エッジの描画
        for u, v, data in G.edges(data=True):
            start_pos, end_pos = pos[u], pos[v]
            style = 'arc3,rad=0.15' if data.get('type') == 'basis' else 'arc3,rad=0.05'
            arrow = mpatches.FancyArrowPatch(
                start_pos, end_pos,
                arrowstyle='-|>',
                mutation_scale=20,
                shrinkA=30, shrinkB=30,
                color="purple" if data.get('type') == 'basis' else "black",
                linestyle="dashed" if data.get('type') == 'basis' else "solid",
                connectionstyle=style,
                zorder=1
            )
            ax.add_patch(arrow)

        # ノードの描画
        font_dict = {'family': INSTALLED_JAPANESE_FONT, 'size': 8, 'weight': 'bold'}
        for node_id, node_data in G.nodes(data=True):
            x, y = pos[node_id]
            ax.text(
                x, y,
                s=node_data['label'],
                fontdict=font_dict,
                ha='center', va='center',
                bbox=dict(
                    boxstyle='round,pad=0.6',
                    facecolor=node_data['color'],
                    edgecolor='black',
                    linewidth=1,
                    alpha=0.9
                ),
                zorder=2
            )

        # 凡例の描画
        handles = [
            mpatches.Patch(color='skyblue', label='質問'),
            mpatches.Patch(color='khaki', label='AI思考ステップ'),
            mpatches.Patch(color='lightgreen', label='最終回答'),
            mlines.Line2D([], [], color='black', label='時系列の流れ'),
            mlines.Line2D([], [], color='purple', ls='--', label='根拠・参照')
        ]
        ax.legend(handles=handles, loc='upper right', 
                 prop=self.font_properties, fontsize='small')
        
        ax.set_title("AI思考連鎖の可視化 (Gemini)", 
                    fontproperties=self.font_properties, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.axis('off')
        ax.margins(0.15)
        
        return fig

    def get_error_messages_html(self) -> str:
        """エラーメッセージをHTML形式で取得"""
        if not self.error_messages_for_graph:
            return ""
        
        error_list = ''.join(f'<li>{msg}</li>' for msg in self.error_messages_for_graph)
        return f"""<div style='color: red; border: 1px solid red; padding: 10px;'>
                   <strong>グラフ注意:</strong><ul>{error_list}</ul></div>"""


class AISystem:
    """AIシステムのメインクラス"""
    
    def __init__(self, gemini_api_key: str, google_api_key: Optional[str] = None, 
                 google_cse_id: Optional[str] = None, system_prompt: str = Config.DEFAULT_SYSTEM_PROMPT):
        if not gemini_api_key:
            raise ValueError("Gemini API key is required.")
        
        self.llm_handler = GeminiHandler(
            api_key=gemini_api_key, 
            system_prompt=system_prompt
        )
        self.search_handler = GoogleSearchHandler(
            api_key=google_api_key, 
            cse_id=google_cse_id
        )
        self.parser = ThoughtParser()
        self.graph_generator = GraphGenerator(
            llm_handler=self.llm_handler, 
            installed_japanese_font=INSTALLED_JAPANESE_FONT
        )

    def update_system_prompt(self, new_prompt: str):
        """システムプロンプトを更新"""
        self.llm_handler.update_system_prompt(new_prompt)

    def _append_and_renumber_steps(self, existing: List[ParsedStep], 
                                  new: List[ParsedStep]) -> List[ParsedStep]:
        """既存のステップに新しいステップを追加してリナンバー"""
        updated = list(existing)
        existing_sigs = {(s.type, s.raw_content) for s in existing}
        max_id = max((s.id for s in existing), default=0)
        
        for step in new:
            if (step.type, step.raw_content) not in existing_sigs:
                max_id += 1
                step.id = max_id
                updated.append(step)
                existing_sigs.add((step.type, step.raw_content))
        
        return updated

    def process_question_iterations(self, user_question: str, progress_fn):
        """質問処理の反復処理"""
        progress_fn(0, desc="思考開始...")
        steps, answer, raw_output, errors = [], "", "", []
        
        # 最初のLLM呼び出し
        for update in self.llm_handler.generate_response(user_question):
            if "ERROR:" in update:
                raw_output = update
                break
            if update.endswith("..."):
                progress_fn(0.1, desc=update)
            else:
                raw_output = update
        
        errors.extend(self.llm_handler.get_error_messages())
        
        if "ERROR:" in raw_output:
            yield (f"AI処理エラー: {raw_output}", None, f"<p style='color:red;'>{raw_output}</p>")
            return
        
        # 初回解析
        parsed = self.parser.parse_llm_output(raw_output)
        steps = self._append_and_renumber_steps(steps, parsed.steps)
        answer = parsed.final_answer
        
        if parsed.error_message:
            errors.append(f"パーサー通知: {parsed.error_message}")
        
        last_steps = parsed.steps
        
        # 検索反復処理
        for i in range(Config.MAX_SEARCH_ITERATIONS):
            search_step = next((
                s for s in last_steps 
                if s.type == "情報収集" and s.search_query
            ), None)
            
            if not search_step:
                break
            
            progress_fn(0.3 + i*0.1, desc=f"検索中({i+1})...")
            search_results = self.search_handler.search(search_step.search_query) or "情報取得失敗"
            
            progress_fn(0.4 + i*0.1, desc="再考中...")
            for update in self.llm_handler.generate_response(user_question, steps, search_results):
                if "ERROR:" in update:
                    raw_output = update
                    break
                if update.endswith("..."):
                    progress_fn(0.5 + i*0.1, desc=update)
                else:
                    raw_output = update
            
            errors.extend(self.llm_handler.get_error_messages())
            
            if "ERROR:" in raw_output:
                errors.append(f"AI処理エラー(検索後): {raw_output}")
                break
            
            parsed = self.parser.parse_llm_output(raw_output)
            newly_parsed_steps = parsed.steps
            steps = self._append_and_renumber_steps(steps, newly_parsed_steps)
            
            if parsed.final_answer.strip():
                answer = parsed.final_answer
            
            if parsed.error_message:
                errors.append(f"パーサー通知(検索後): {parsed.error_message}")
            
            last_steps = newly_parsed_steps
        
        if not answer.strip():
            answer = "最終回答の明示的出力はありませんでした。グラフを参照してください。"
        
        # グラフ生成
        progress_fn(0.8, desc="グラフ生成中...")
        summary_prog = lambda p, d: progress_fn(0.8 + p * 0.15, desc=d)
        fig = self.graph_generator.create_thinking_graph(user_question, steps, answer, summary_prog)
        
        graph_image = None
        if fig:
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                plt.close(fig)
                buf.seek(0)
                graph_image = Image.open(buf)
            except Exception as e:
                logger.error(f"Error converting graph to PIL Image: {e}")
                errors.append(f"グラフ画像の変換エラー: {e}")

        errors.extend(self.graph_generator.error_messages_for_graph)
        
        # エラーメッセージの生成
        if errors:
            unique_errors = sorted(list(set(errors)))
            error_list = ''.join(f"<li>{msg}</li>" for msg in unique_errors)
            error_html = f"""<div style='color: red; border: 1px solid red; padding: 10px; margin-top:10px;'>
                            <strong>処理に関する通知:</strong><ul>{error_list}</ul></div>"""
        else:
            error_html = ""
        
        progress_fn(1.0, desc="完了")
        yield (answer, graph_image, error_html)


def create_gradio_interface():
    """Gradioインターフェースを作成"""
    # 環境変数からAPIキーを取得
    env_keys = get_env_api_keys()
    has_all_keys = all(env_keys.values())
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        system_state = gr.State(None)
        
        gr.Markdown("# 🧠 AI思考連鎖可視化システム (Gemini版)")
        gr.Markdown("Gemini APIを利用して、AIの思考プロセスを段階的に可視化します。システムプロンプトをカスタマイズできます。")
        # カスタムCSS: グラフ画像を確実に中央に配置（flexレイアウトを列に適用）
        gr.HTML("""
        <style>
        /* グラフ列コンテナを中央寄せレイアウトに */
        .graph-column { 
            display: flex !important; 
            flex-direction: column; 
            align-items: center; 
            justify-content: flex-start; 
            gap: 0.5rem; 
        }
        /* 画像/キャンバス自体を中央 & 可変幅 */
        .graph-column img, .graph-column canvas { 
            display: block !important; 
            margin: 0 auto !important; 
            max-width: 100% !important; 
            height: auto !important;
        }
        /* Imageコンポーネント外枠をフル幅にしつつ内部を中央寄せ */
        #graph-image { 
            width: 100%; 
            text-align: center; 
        }
        #graph-image img { 
            max-width: 95%; 
        }
        /* モバイル向けの縮小余白 */
        @media (max-width: 780px) { 
            #graph-image img { max-width: 100%; }
        }
        </style>
        """)
        
        # .envにすべてのキーがある場合は設定画面を非表示にする
        with gr.Accordion("APIキー設定", open=not has_all_keys, visible=not has_all_keys) as api_accordion:
            with gr.Row():
                gemini_key_input = gr.Textbox(
                    label="Gemini API Key", 
                    type="password", 
                    placeholder="ここにGemini APIキーを入力"
                )
                google_key_input = gr.Textbox(
                    label="Google Search API Key (任意)", 
                    type="password", 
                    placeholder="検索機能を使う場合に入力"
                )
                google_cse_id_input = gr.Textbox(
                    label="Google CSE ID (任意)", 
                    type="text", 
                    placeholder="検索機能を使う場合に入力"
                )
            
            set_api_key_button = gr.Button("APIキーを設定", variant="primary")
            api_status_output = gr.Markdown()
        
        # .envからキーが読み込まれている場合の表示
        if has_all_keys:
            gr.Markdown("✅ APIキーが環境変数から自動読み込みされました")
        
        with gr.Accordion("システムプロンプト設定", open=False):
            system_prompt_input = gr.Textbox(
                label="システムプロンプト",
                value=Config.DEFAULT_SYSTEM_PROMPT,
                lines=30,
                max_lines=30,
                placeholder="AIの動作を制御するシステムプロンプトを入力してください",
                info="プロンプト内で {user_question}, {previous_steps_str}, {search_results_str} を使用できます"
            )
            update_prompt_button = gr.Button("システムプロンプトを更新", variant="secondary")
            prompt_status_output = gr.Markdown()
        
        # .envにキーがある場合はメインインターフェースを最初から表示
        with gr.Column(visible=has_all_keys) as main_interface:
            gr.Markdown("---")
            question_input = gr.Textbox(
                label="質問を入力してください:", 
                placeholder="例: 日本のAI技術の最新トレンドとその課題は何ですか？", 
                lines=10
            )
            submit_button = gr.Button("質問を送信する", variant="primary")
            
            gr.Markdown("## 🤖 AIの回答と思考プロセス")
            with gr.Row():
                with gr.Column(scale=1):
                    answer_output = gr.Markdown(label="AIの最終回答")
                    graph_errors_output = gr.HTML(label="処理に関する通知")
                with gr.Column(scale=2, elem_classes=["graph-column"]):
                    graph_output = gr.Image(
                        label="思考プロセスネットワーク図", 
                        type="pil", 
                        interactive=False, 
                        show_download_button=True,
                        elem_id="graph-image"
                    )
            
            gr.Examples(
                examples=[
                    ["日本のAI技術の最新トレンドとその社会への影響について教えてください。"],
                    ["太陽光発電のメリット・デメリットを整理し、今後の展望を予測してください。"]
                ], 
                inputs=[question_input]
            )

        def set_api_keys(gemini_key, google_key, cse_id, system_prompt):
            """APIキーを設定"""
            # 環境変数からキーを取得して、入力されたキーと統合
            env_keys = get_env_api_keys()
            
            final_gemini_key = gemini_key.strip() if gemini_key and gemini_key.strip() else env_keys['gemini_api_key']
            final_google_key = google_key.strip() if google_key and google_key.strip() else env_keys['google_api_key']
            final_cse_id = cse_id.strip() if cse_id and cse_id.strip() else env_keys['google_cse_id']
            
            if not final_gemini_key:
                return (
                    None, 
                    gr.update(value="<p style='color:red;'>Gemini APIキーは必須です。</p>"), 
                    gr.update(visible=False)
                )
            
            try:
                system = AISystem(
                    gemini_api_key=final_gemini_key, 
                    google_api_key=final_google_key, 
                    google_cse_id=final_cse_id,
                    system_prompt=system_prompt
                )
                search_status = "有効" if system.search_handler.is_enabled else "無効"
                key_source = "環境変数" if not gemini_key.strip() else "手動入力"
                status_md = f"""<p style='color:green;'>APIキー設定完了。質問を開始できます。</p>
                               <p><strong>Google検索</strong>: `{search_status}`</p>
                               <p><strong>APIキー取得元</strong>: `{key_source}`</p>"""
                
                return system, gr.update(value=status_md), gr.update(visible=True)
            except Exception as e:
                logger.error(f"Failed to initialize AI system: {e}")
                return (
                    None, 
                    gr.update(value=f"<p style='color:red;'>システム初期化エラー: {e}</p>"), 
                    gr.update(visible=False)
                )

        def update_system_prompt(system, new_prompt):
            """システムプロンプトを更新"""
            if system is None:
                return gr.update(value="<p style='color:red;'>先にAPIキーを設定してください。</p>")
            
            if not new_prompt.strip():
                return gr.update(value="<p style='color:orange;'>システムプロンプトが空です。</p>")
            
            try:
                system.update_system_prompt(new_prompt)
                return gr.update(value="<p style='color:green;'>システムプロンプトが更新されました。</p>")
            except Exception as e:
                logger.error(f"Failed to update system prompt: {e}")
                return gr.update(value=f"<p style='color:red;'>システムプロンプト更新エラー: {e}</p>")

        def submit_fn_wrapper(system, question, progress=gr.Progress(track_tqdm=True)):
            """質問送信のラッパー関数"""
            if system is None:
                return (
                    "APIキーが設定されていません。", 
                    None, 
                    "<p style='color:red;'>まずAPIキーを設定してください。</p>"
                )
            
            if not question.strip():
                return (
                    "質問が入力されていません。", 
                    None, 
                    "<p style='color:orange;'>質問を入力してください。</p>"
                )
            
            final_result = (None, None, None)
            for result in system.process_question_iterations(question, progress):
                final_result = result
            return final_result

        # イベントハンドラーの設定
        set_api_key_button.click(
            fn=set_api_keys,
            inputs=[gemini_key_input, google_key_input, google_cse_id_input, system_prompt_input],
            outputs=[system_state, api_status_output, main_interface]
        )
        
        update_prompt_button.click(
            fn=update_system_prompt,
            inputs=[system_state, system_prompt_input],
            outputs=[prompt_status_output]
        )
        
        submit_button.click(
            fn=submit_fn_wrapper,
            inputs=[system_state, question_input],
            outputs=[answer_output, graph_output, graph_errors_output]
        )
        
        # エンターキーで送信する機能を追加
        question_input.submit(
            fn=submit_fn_wrapper,
            inputs=[system_state, question_input],
            outputs=[answer_output, graph_output, graph_errors_output]
        )
        
        # .envにキーがある場合は起動時に自動設定
        if has_all_keys:
            def auto_init():
                system = AISystem(
                    gemini_api_key=env_keys['gemini_api_key'],
                    google_api_key=env_keys['google_api_key'], 
                    google_cse_id=env_keys['google_cse_id'],
                    system_prompt=Config.DEFAULT_SYSTEM_PROMPT
                )
                return system
            
            demo.load(
                fn=auto_init,
                inputs=[],
                outputs=[system_state]
            )
    
    return demo


if __name__ == "__main__":
    gradio_app = create_gradio_interface()
    gradio_app.launch()