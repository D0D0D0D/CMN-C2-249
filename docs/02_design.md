# テンプレート設計仕様 — CMN-C2-249（Cat 2）

> **テンプレート**: Deep Research Synthesis & Evidence Report Agent（ディープリサーチ統合・根拠レポート生成エージェント）
> **設計範囲**: 条件付き HITL とハイブリッド検索を備えた Cat 2 ワークフロー
> **カテゴリ**: **Cat 2**（多段ワークフローでユースケースを完遂）・**業種**: CMN（業種非依存。Cat 軸と Industry 軸は独立）
> **パターン**: 調査計画 → 人間確認（HITL）→ マルチソース調査 → 根拠レポート統合
>
> **ID/カテゴリ**: manifest と README は **`CMN-C2-249` / `Cat 2`** で整合する。

## AgentCore アーキテクチャ上の位置づけ

- **Agent クラス**: `DeepResearchAgent`
- **L1 Base**: **AgentBaseGraph**（固定 5 ノード backbone を外側グラフとして継承）。`main` スロットに
  **`GraphNode`** を置き、内部ドメインワークフロー（`src/graph/domain_workflow_graph.py`）をラップする
  ＝ Cat 2 の標準構造（外側 backbone は Cat 1 と同一、ドメインの複雑性は内部グラフに封じ込める）。
- **3 層分離**:
  - **State**: flat TypedDict 構成（Pydantic は msgpack 非互換のため不可）。`framework.schemas.agent_state.AgentState` を継承。
  - **Node**: L1 継承（Template Method: `execute(self, state: dict) -> dict` のみ override。`__call__()` は絶対に override しない）。
  - **Graph**: 合成（外側 `register_nodes()` で `pre_process` / `main`(GraphNode) / `post_process` を埋め、内部グラフが調査ワークフローを構成）。

### Cat 2 とする根拠

本テンプレートは単一の汎用能力（Cat 1）ではなく、**複数ステップでひとつのユースケース
「引用付き根拠レポートの作成」を完遂する多段ワークフロー**である:

> 調査計画の立案 → （必要時）人間による調査方針の確認 → 許可ソース群からのマルチソース調査
> → クロスソース統合 → 引用・信頼度・ギャップ付きの根拠レポート生成

この「parse →（計画）→ 確認 → 調査 → 統合 → レポート」という多段構成は、単一汎用能力の Cat 1 ではなく、
ユースケースを完遂する Cat 2 に該当する。
LLM が次アクションを自律決定する think→act→observe ループ（Cat 3）ではない — 計画は 1 パス、調査は
確認済みソースへの固定的取得であり、HITL は「自律」ではなく「人間統制の 1 ゲート」である。

## アーキテクチャ概要

### 外側 backbone ノード構成

| ノード | 責務 | 入力 State | 出力 State | 継承 / override |
|------|------|-------------|--------------|-------------------|
| initialize | schema_version・session_id・trust_level の初期化 | `user_input` | `session_id`, `schema_version`, `trust_level` | `InitializeNode`（デフォルト、super 注入） |
| pre_process | 質問解析。ユーザー指定（参照ソース・引用品質・深さ）があれば取り込み、無ければ候補プランを生成。`plan_confirmed` を判定 | `user_input`, `input_context` | `research_question`, `user_source_spec`, `research_plan`, `plan_confirmed`, `quality_bar` | `PreProcessNode(FunctionNode)` — `execute()` を override |
| **main（GraphNode）** | 内部ドメインワークフローをラップ・実行（計画確認 HITL → マルチソース調査 → 統合）。`get_subgraph()` で内部グラフを返す | `research_plan`, `plan_confirmed`, `quality_bar` | `source_whitelist`, `retrieved_evidence`, `synthesized_findings`, `identified_gaps` | `ResearchWorkflowNode(GraphNode)` — 内部グラフを合成 |
| post_process | 根拠レポート整形（ソースティア付き引用、信頼度、ギャップ）。引用整合性の強制（S-3 preservation variant） | `synthesized_findings`, `retrieved_evidence`, `identified_gaps` | `evidence_report`, `formatted_output` | `PostProcessNode(FunctionNode)` — `execute()` を override |
| finalize | response_metadata・total_time_ms | `formatted_output` | `response_metadata`, `total_time_ms` | `FinalizeNode`（デフォルト、super 注入） |

### 内部ドメインワークフローグラフ（`src/graph/domain_workflow_graph.py`）

`main` の `GraphNode` がラップする内部グラフ。3 ノード:

| 内部ノード | 責務 | HITL / セキュリティ |
|------|------|------|
| `plan_node` | `research_plan`（候補ソース ＋ 引用品質ティア ＋ 深さ）を確定。`plan_confirmed == false` のとき **D6 `interrupt()` で計画をユーザーに提示し Go 待ち**（査読済みのみ／新聞可／内部のみ 等）。再開時にユーザー修正を反映し確定 | HITL D6（条件付き）。S-1: 不要 |
| `retrieve_node` | 確定ソート集合 = **（デプロイ許可カタログ）∩（ユーザー/プラン選択）** の各ソースから `quality_bar` を守って取得。内部 KB は index 済み検索、外部ソースはライブコネクタ。重複排除・集約 | S-1: `required_trust_level = VERIFIED_EXTERNAL`。S-4: ソースごとに `emit_trace_event("source_retrieval", …)` |
| `synthesize_node` | クロスソース統合。claim ごとに出典バインドと信頼度スコア、エビデンス不足を `identified_gaps` に集約 | S-4: 統合 LLM 呼び出しごとに `emit_trace_event("llm_synthesis", …)` |

### データフロー

```
START → initialize → pre_process → main(GraphNode) → {route} → post_process → finalize → END
                                       │                          ↑ (RETRY, 最大 3 — 一時的なソース/LLM エラー)
                                       │  ┌───────── 内部ワークフローグラフ ─────────┐
                                       └─▶│ plan_node → [HITL: 計画提示→Go] → retrieve_node → synthesize_node │
                                          │              ↑ plan_confirmed==false のみ発火（指定明確なら skip） │
                                          └──────────────────────────────────────────────┘
```

- ハード早期終了: `pre_process` が空/不正な質問に `state["error"]` をセット → backbone は `main` 手前で停止
  （早期終了トリガーは `error_code` 単独でなく `state["error"]`）。
- HITL スキップ条件: ユーザーが入力時に参照ソース・引用品質・深さを**明確に指定済み**なら
  `plan_confirmed = true` とし、`plan_node` の `interrupt()` を飛ばす。

### State 定義

flat TypedDict、`AgentState` を継承。エージェント固有フィールドのみ（共有フィールド `user_input`,
`status`, `session_id`, `node_history`, `error_log`, `error` は継承）:

| フィールド | 型 | 目的 | 必須 |
|-------|------|---------|------|
| `research_question` | `str` | 検証・正規化済みのリサーチ質問 | はい |
| `user_source_spec` | `dict` | ユーザーが入力時に指定した参照ソース/品質/深さ（任意。未指定なら空） | いいえ |
| `research_plan` | `dict` | `{proposed_sources: list[str], quality_bar: str, depth: str}` — 確定/提案された調査プラン | はい |
| `plan_confirmed` | `bool` | 調査プランが確定済みか（true なら HITL skip） | はい |
| `quality_bar` | `str` | 引用品質ティア（`peer_reviewed_only` / `news_acceptable` / `internal_only` 等） | はい |
| `source_whitelist` | `list[str]` | 実効ソース集合 ＝（許可カタログ）∩（ユーザー/プラン選択） | はい |
| `retrieved_evidence` | `list[dict]` | 文献別ヒット: `{evidence_id, source_id, source_type, source_tier, title, snippet, url, published_date, score}`。`evidence_id`（`{source_id}#{n}`・1 始まり）は **`retrieve_node` がサーバ側で採番**する文献単位ハンドル | はい |
| `synthesized_findings` | `list[dict]` | `{claim, supporting_source_ids: list[str], confidence: float}`。`supporting_source_ids` は **`evidence_id` の集合**（＝文献 ID。コネクタ ID ではない）、`confidence` は有限かつ 0..1 | はい |
| `identified_gaps` | `list[str]` | 十分なエビデンスが無かった問い/サブトピック | はい |
| `evidence_report` | `dict` | 最終構造化レポート: `{sections, citations, confidence_summary, gaps}` | はい |
| `formatted_output` | `str` | 呼び出し元へ返す描画済み 2〜5 ページレポート（markdown） | はい |
| `hitl_*` | （schema 1.4 の許可 7 フィールド） | `interrupt()` が扱う HITL 状態 | HITL 時 |

**State 制約（必須）:**
- flat TypedDict のみ（プリミティブ ＋ JSON シリアライズ可能型）。
- State に JWT・API キー・認証情報を入れない（checkpoint DB 漏洩）。ソース/LLM の API キーは
  `ctx.secrets.require()` 経由で取得し State には書かない。
- `InvocationContext` は `config["configurable"]` 経由のみ。
- Pydantic モデル・dataclass・任意 Python オブジェクトは不可（msgpack 非互換）。

## フレームワーク活用

### 利用する共有コンポーネント
- [x] InvocationContext（correlation_id, session_id, permissions, credential handle） — `config["configurable"]` 経由
- [x] AuditSink（`emit_trace_event`） — 外部検索ごと・LLM 統合呼び出しごとに 1 ドメインイベント（S-4）
- [x] ConnectionPolicy — ソースごとの検索タイムアウト、一時失敗は backbone RETRY
- [x] SecurityViolationError — S-2 不許可入力、S-3 引用整合性違反で raise
- [x] **HITL（D6 `interrupt()`）** — `plan_node` で計画確認。`config.yaml` の `hitl.enabled` は `true`
- [x] `_security_gate_input()`（PII、`@final`。ドメインは `_extra_security_gate_input()`）
- [x] `_security_gate_output()`（コンテンツ安全、`@final`。引用整合性は `_extra_security_gate_output()`）

### LLM 注入シーム（`synthesize_node`）

統合生成は `src/services/llm_client.py` の `LLMClient`
（`synthesize(question, evidence, quality_bar) -> {"findings": [...], "gaps": [...]}`）が境界。
**受け口は `config["llm"]` ただ 1 つ**（`configurable.llm` ネストも可）。

| 渡されたもの | 扱い |
|---|---|
| framework の `BaseLLM`（`AnthropicClient` 等） | `BaseLLMSynthesisAdapter` でラップし `complete()` へ橋渡し |
| 当テンプレートの `synthesize()` プロトコル | そのまま使用 |
| いずれでもない | `LLMSynthesisError` で fail-loud |

判定順序は **`isinstance(BaseLLM)` が duck typing より先**。SDK の具象クライアントは
未知属性を内部 LangChain クライアントへ委譲しうるため、`getattr(x, "synthesize")` が
別シグネチャの無関係な callable を拾ってシームを誤結線する。

- **独自コンストラクタ引数は持たない**。`BaseGraph.__init__(self, config=None)` が
  フレームワークの唯一の構築契約であり、`llm=` のような独自キーワードは
  `src/api/server.py` を含む契約準拠の呼び出し元からは永久に渡らない。
- **未注入時**は `synthesize_node.execute()` が呼び出しごとに
  `ctx.secrets.require("LLM_API_KEY")` で鍵を解決し、`config/config.yaml` の
  `llm_provider` / `llm_model` でプロバイダクライアントを構築する。
  構築したクライアントは `self` にキャッシュしない（ノードインスタンスは全 invocation 共有）。
  未注入経路が構築できるのは **API キー 1 本で足りるプロバイダのみ**（`anthropic` / `openai`）。
  `BedrockClient` は `aws_access_key_id` / `aws_secret_access_key` / `aws_region` を要求するため
  この builder では構築できず、**意図的に候補から外している**。Bedrock 配備は
  `config["llm"]` に自前クライアントを注入する（当テンプレートはそのまま受け取る）。
- **モデル ID は `src/` に書かない**。`config/config.yaml` の `llm_model` か、未設定なら
  クライアント既定。監査イベント `llm_synthesis` の `model` は
  **canonical response の `model`** を採る（config の文字列は実際の選択を決めないため）。

#### 出所（provenance）の所有

モデルが書くのは **散文（`claim`）と参照 ID** のみ。

- **参照 ID の粒度は「文献」であって「コネクタ」ではない。** `retrieve_node` が受理した
  ヒットごとに `evidence_id`（`{source_id}#{n}`）を**サーバ側で採番**し、プロンプトはその
  ID だけを提示する。コネクタ名（`pubmed`）は citable な ID として提示しない — 提示すると
  「そのコネクタが返した全文献」を指すことになり、claim がどの文献に支えられているかを
  言えなくなる。
- コネクタが payload に `evidence_id` を入れてきても**無条件に上書き**する（サーバ発行トークン。
  他文献のハンドルへの成りすまし・衝突を防ぐ）。
- `supporting_source_ids` は取得済み `retrieved_evidence` の `evidence_id` に対して
  `synthesize_node` が **必ず解決**する。存在しない ID（モデルが発明したもの・素のコネクタ名）は
  破棄し `identified_gaps` に記録する。
- 引用メタデータ（title / URL / tier / published_date）は `post_process_node` が
  **取得レコードから読む**。モデルの出力からは一切採らない。索引キーも `evidence_id` なので、
  claim は自分が名指しした文献だけを引用する。
- findings も gaps も空になった場合は「根拠から支持できる知見が出なかった」旨の gap を
  必ず立てる。「知見なし・ギャップなし」は「不足が無かった」と読めてしまうため。

#### LLM 失敗時の縮退方針 — **縮退しない（`status=error`）**

統合そのものが本 AG の中核能力であり、決定論的な代替が存在しない。もっともらしい空レポートを
`success` で返すことは、調査結果として誤りである。理由として State に載せるのは
**例外の型名のみ**。`BaseNode.__call__` の汎用ハンドラは `str(exc)` とトレースバックを
`error_log` に入れるが、本テンプレートは `merge_output()` → `get_output()` で
`error_log` を呼び出し元へ返すため、`execute()` 内で捕捉して封じ込める。

**S-3 認証情報の姿勢 = BLOCK（入れ子は本テンプレート側で走査）。**
取得スニペットや LLM 応答に認証情報パターンが含まれる場合、実行は `status=error` で終わる
（下流ノードでのマスクは到達しない）。これが正しい end-to-end 挙動である。

ただし **framework の `@final` S-3 ゲートが入れ子を再帰するかは wheel の版に依存する**
（`framework/nodes/function_node.py`。トップレベルの `str` しか走査しない版が存在する）。
本テンプレートはその差に依存しない。
本テンプレートは payload を入れ子で返す（`retrieved_evidence` /
`synthesized_findings` / `identified_gaps` / `evidence_report`）ため、
そのままでは **平坦化された `formatted_output` で実行はブロックされるのに、
入れ子側の複製は `get_output()` 経由で呼び出し元へ返ってしまう**。

そこで `src/security_gates.py` の `assert_no_nested_credentials()` を
`retrieve_node` / `synthesize_node` / `post_process_node` の
`_extra_security_gate_output()` に配線し、**その値を最初に返すノードが所有する**形で
入れ子を再帰走査する（Redaction Ownership）。検出器そのものは
substrate の `detect_credentials` を**再利用**しており、独自実装はしていない。
送出するメッセージには**一致した値を含めない**（含めれば塞いだ穴を再び開ける）。

### セキュリティモデル対応（S-1 … S-5）

| レイヤ | 本テンプレートの責務 |
|-------|--------------------------------|
| **S-1 Trust Gate** | `retrieve_node`（内部 KB ＋ 外部コネクタにアクセス）に `required_trust_level = VERIFIED_EXTERNAL` を宣言。各コネクタが触れる権限は per-source で管理。`__call__()` は override しない。 |
| **S-2 Input Gate** | `PreProcessNode._extra_security_gate_input()` — 空/過長の質問、認証情報パターンを含む入力、**許可カタログ外のソース指定**を拒否。PII スキャンは `@final` が自動実行。 |
| **S-3 Output Gate** | `PostProcessNode._extra_security_gate_output()` — **preservation variant**: claim が `supporting_source_id` を 1 つも持たない、または finding が明示的信頼度なしで出力されるレポートを拒否（no-hallucination-as-fact ゲート）。認証情報スキャンは `@final` が実行。 |
| **S-4 Audit** | `retrieve_node`/`synthesize_node` の `execute()` 内で `emit_trace_event` を発行。`node_start/complete/error` はフレームワーク発行のため重複させない。 |
| **S-5 Credential / サプライチェーン** | ソース ＋ LLM キーは `ctx.secrets.require("KEY")` 経由、`agent.yaml` の `requires.secrets` に宣言。`os.environ` 不使用。`[project].dependencies` は全て `==` で厳密ピン。 |

### 入力契約（ユーザー駆動 ＋ 統制カタログ）

| 何を | 誰が | 必須 |
|---|---|---|
| リサーチ質問 | ユーザー | **必須** |
| 参照ソース・引用品質（査読済みのみ／新聞可／内部のみ 等）・深さ | **ユーザー指定、または AG 提案 → HITL 確認** | 任意（未指定なら AG 提案＋HITL） |

- **統制境界**: ユーザーが選べるのは**デプロイ時に接続を許可したソースカタログの中から**のみ。実行時に
  任意の新規外部エンドポイントへ AG を向けることは不可（S-1 権限・外部 ToS 保護、SSRF 回避）。新規コネクタ
  追加はデプロイ作業。これにより**自律的な任意ソース探索（Cat 3）に陥らず Cat 2 の固定ワークフローを保つ**。

### 合成パターン

- **パターン**: GraphNode（`main` スロットが内部 BaseGraph をラップ）。
- **合成ターゲット**: `src/graph/domain_workflow_graph.py`（plan → retrieve → synthesize）。
- **エラー伝播戦略**: **handle ＋ retry** — 一時的なソース別失敗は内部ノードで捕捉・`error_log` 記録し
  `status = ERROR` で backbone RETRY（最大 3）。枯渇後は到達不能ソースを `identified_gaps` に降格し、
  取得済み分の引用整合性を保ってレポートを返す（全体ハード失敗にしない）。

## インポート分離の確認
- [x] テンプレートは agenticstar-platform SDK（Level 0）を import しない
- [x] import 先は `framework/`（AgentBaseGraph, GraphNode, FunctionNode, AgentState, AgentStatus, TrustLevel, InvocationContext, SecurityViolationError）と `src.*` のみ — `agents/base/` 継承なし（2026-05-18 モデル: **Level 2 base → L1 direct inheritance**）

### 出力スキーマの変更履歴

**2026-08-28 — 引用を文献単位へ（破壊的変更）**

| 対象 | 変更前 | 変更後 |
|---|---|---|
| `retrieved_evidence[]` | `evidence_id` なし | `evidence_id: "{source_id}#{n}"` を追加（`retrieve_node` が採番・1 始まり・コネクタごとに振り直し） |
| `synthesized_findings[].supporting_source_ids` | **コネクタ ID** の集合（`["pubmed"]`） | **文献 ID** の集合（`["pubmed#1"]`）。素のコネクタ名はもう解決しない |
| `evidence_report.citations[]` | `{source_id, source_tier, title, url, published_date}` | `evidence_id` を追加した 6 キー。`source_id` は引き続きコネクタ名 |
| `formatted_output` の Citations 行 | `- [tier] title — url` | `- [evidence_id] [tier] title — url` |
| LLM プロンプト | `Allowed source_ids: [...]` | `Allowed evidence_ids: [...]`（文献単位・「claim を支持する文献だけを引用せよ」を明示） |

既存の呼び出し元が `supporting_source_ids` をコネクタ名として読んでいた場合は影響を受ける。
`evidence_id` を持たないレコード（手組みの単体テスト等）は `source_id` にフォールバックするため、
参照解決自体は壊れない（フォールバック値もレコード由来であってモデル由来ではない）。

## 設計判断記録（Design Decision Record）

### 2026-09-01 — Marketplace 動作確認用コーパス

`ResearchService` の fail-loud な既定 seam は維持し、`research_corpus_path` が設定された場合だけ
`CorpusResearchService` を構築する。検索は英数字 token と日本語 CJK 2-gram の語彙重なりで決定論的に
行う。サンプル文書は `config/research_corpus.sample.json` に隔離し、title・snippet・URL・doc_id の全てで
サンプルであることを明示する。provenance の `source_id` / `source_tier` / `evidence_id` は従来どおり
`RetrieveNode` が所有し、コーパス実装からは返さない。

### 2026-09-01 — AGENTIC STAR Marketplace entry adapter（`cli.py`）

`src/api/server.py`（FastAPI）に加えて 2 本目の入口として `cli.py` を追加する。ノードとグラフは無改造で、adapter が行うのは配線と Marketplace 固有の入口硬化のみ:

- **LLM 配線** — `config["llm"]` に遅延解決型の Azure `BaseLLM` を注入する。注入は`super().__init__()` の**前**（`register_nodes()` が `self.config` を内側 GraphNode に渡すため）、かつ`setdefault` ではなく `is None` ガード（`config.yaml` が `llm: null` を持つ場合の無言 no-op を避ける）。接続値はインスタンスに保持せず、呼び出しのたびに `current_secrets()` から解決する。
- **HITL 抑止** — runner は `AWAITING_HUMAN` を失敗として扱うため `ctx.hitl_allowed=False` を立てる。`plan_confirmed=true` の注入という代替案は**採らない**：人間が計画を承認したという偽の記録になるため。抑止先の `PlanNode._finalize()` は state delta を返すだけで永続化も外部書込も行わない（実測確認済み）。
- **`input_context` の組み直し** — runner の `conversation_history` は S-2 入力ゲートの走査対象（`user_input` / `validated_input` / `llm_response`）**外**の自由文なので、マージせず破棄してadapter 所有のキーだけで組み直す（現状は空 dict）。
- **起動時 fail closed** — `requires.secrets` は Marketplace 経路では強制されない（`require_at_compile()` に wheel 内の呼び出し元が無い）ため、adapter の `provision_secrets()` がAzure 3 値のクリーンさとコーパスファイルの存在を検査する唯一の防御になる。

| 判断 | 選択肢 A | 選択肢 B | 採用 | 根拠 |
|----------|----------|----------|--------|-----------|
| **引用の粒度（2026-08-28 変更）** | コネクタ単位（`source_id` で索引） | **文献単位（`evidence_id` で索引）** | **文献単位** | 旧設計では 1 コネクタが複数文献を返すと、そのコネクタを引用した claim に**全文献**が Citations として載った。実 LLM の実データ検証（GLP-1 の心血管アウトカム）で、無関係な土壌微生物論文が心血管の claim の引用として出力された。証跡レポート AG で出典が信用できないのは中核機能の破綻。ID はモデルではなく `retrieve_node` が採番するため「モデルは散文と参照 ID だけを書く」規律は維持される。 |
| L1 base 種別・構造 | **AgentBaseGraph（外側）＋ GraphNode（main）** | AutonomousBaseGraph（自律ループ） | **AgentBaseGraph ＋ GraphNode** | 計画は 1 パス、調査は確認済みソートへの固定的取得で、LLM が次アクションを自律決定しない。Cat 2 の標準構造（外側固定 backbone ＋ 内部ドメインワークフロー）が適合。HITL は人間統制の 1 ゲートであり自律ではない。 |
| カテゴリ | Cat 1（単一汎用能力） | **Cat 2（ユースケース完遂の多段ワークフロー）** | **Cat 2** | 「計画→確認→調査→統合→レポート」の多段で 1 ユースケースを完遂するため。manifest も Cat 2 で整合。 |
| 参照ソースの決定 | config 固定 | **ユーザー駆動（カタログ内選択）＋ AG 提案→HITL** | **ユーザー駆動＋HITL** | 調査内容で必要ソース・引用品質が変わるためユーザー指定が妥当。ただし選択は**許可カタログ内**に限定し、実行時の任意エンドポイント注入は禁止（S-1/ToS）。 |
| 調査計画の確認 | 常に自動実行 | **HITL 計画ゲート（条件付きスキップ）** | **HITL（条件付き）** | レポート種別で調査方法が変わるため、未指定時は計画をユーザーに提示し Go を得てから調査開始。入力時に明確指定があれば skip。 |
| 検索方式 | 純粋 VectorRAG | **ハイブリッド（内部 index ＋ 外部ライブ）** | **ハイブリッド** | 内部 KB は index 済み検索、外部（査読論文/ニュース等）はライブコネクタ取得。事前 index 済み単一 KB 前提の純 VectorRAG では要件を満たさない。 |
| マルチソース失敗時 | 全体ハード失敗 | **ギャップへ降格** | **ギャップへ降格** | RETRY 後も到達不能なソースは `identified_gaps` に記録、取得済み分の整合性を保つ。 |

---

*実装対象は src/、tests/、および config の hitl.enabled。*
