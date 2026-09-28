# テスト仕様 — CMN-C2-249 Deep Research Synthesis & Evidence Report Agent

## テスト戦略

- 構成: Unit（各ノード単体）/ Integration（5 ノード backbone + 内部ワークフロー）/ Proof-of-Boundary（S-1/S-2/S-3 と静的境界）
- 実行: `python -m pytest tests/ -v` および `python -m pytest tests/proof_of_boundary/ -v`（CI と同一）
- 依存: 本番 framework パッケージ未公開のため `tests/conftest.py` が `framework.*` を `sys.modules` にスタブ登録（DocGenerationAgent・ChatAgent と同じ方式）。スタブは L1 契約（S-1 自動強制・部分 dict マージ・5 ノード backbone + RETRY・GraphNode）を忠実に再現する。
- 外部依存（ソースコネクタ・LLM）は `config["configurable"]` 経由で注入し、テストは `tests/fakes.py` の `FakeResearchService` / `FakeLLMClient` を差し込む。
- 現状: **33 件 PASS / 0 FAIL**（unit 22・integration 4・proof_of_boundary 7）。

## フレームワーク準拠テスト

| TC-ID | テスト | 期待結果 | 対応テスト |
|-------|------|----------|-----------|
| TC-01 | State は flat TypedDict（`AgentState` 継承、Pydantic/dataclass 不可） | 静的スキャン違反 0 | `proof_of_boundary/test_state_safety.py` |
| TC-02 | ノードは `execute(self, state, config=None)` のみ override（`_invoke_impl` 不可） | 契約準拠 | 各 `unit/test_*_node.py` |
| TC-03 | State に認証情報フィールドを持たない | credential-like 名 0 | `proof_of_boundary/test_state_safety.py` |
| TC-04 | 依存は config 経由で注入（LLM は `config["llm"]`、service は `config["research_service"]`。`configurable:` ネストも可） | service/llm 注入で動作・独自 ctor 引数なし | `integration/test_workflow.py` / `integration/test_server_llm_injection.py` / `unit/test_llm_wiring.py` |
| TC-05 | S-4 監査イベント（source_retrieval / llm_synthesis）を発行 | retrieve/synthesize で emit | `retrieve_node.py` / `synthesize_node.py`（emit_trace_event） |
| TC-06 | S-2 入力ゲート（空 / 過長 / credential / 許可外ソース）を実行 | 拒否され早期終了 | `unit/test_pre_process_node.py` |
| TC-07 | S-3 出力ゲート（出典なし claim / 信頼度なし finding）を実行 | SecurityViolationError | `unit/test_post_process_node.py` |
| TC-08 | `required_trust_level`（retrieve=VERIFIED_EXTERNAL）を強制し、拒否時に `execute()` が呼ばれなかったことを証明 | 不足 trust は S-1 で拒否、`execute()` 未到達 | `proof_of_boundary/test_pb_invoke_order.py::test_s1_denial_refuses_execution_before_execute` / `proof_of_boundary/test_security_boundaries.py` |

## Proof-of-Boundary テスト

| PB-ID | 境界 | テスト | 期待結果 | 対応テスト |
|-------|------|------|----------|-----------|
| PB-1（S-1） | Trust gate | ANONYMOUS で retrieve 到達 → 拒否 / VERIFIED_EXTERNAL → 許可 | error に "S-1"、レポート無し / レポート生成 | `test_security_boundaries.py::TestS1TrustBoundary` |
| PB-2（S-2） | Input gate | 許可カタログ外ソース指定・credential 入力 | status error・早期終了 | `test_security_boundaries.py::TestS2InputBoundary` |
| PB-3（S-3） | Output gate | 出典なし claim を含むレポート生成 | SecurityViolationError | `test_security_boundaries.py::TestS3OutputBoundary` |
| PB-4 | Import 分離 | src/ が Level 0（agenticstar/platform）を import しない | AST スキャン違反 0 | `test_import_isolation.py` |
| PB-5 | State 安全性 | invoke 後 State がプリミティブのみ・認証情報なし | 静的スキャン違反 0 | `test_state_safety.py` |
| PB-6 | invoke 実行順 | initialize→pre_process→main(GraphNode)→route→post_process→finalize | node_history で順序確認 | `integration/test_workflow.py::test_happy_path_produces_report` |

## ビジネスロジックテスト

| TC-ID | テスト | 入力 | 期待結果 | 対応テスト |
|-------|------|-------|----------|-----------|
| BL-01 | 正常系（計画確定・複数ソース） | confirmed plan + VERIFIED_EXTERNAL | 引用・信頼度付きレポート生成 | `integration::test_happy_path_produces_report` |
| BL-02 | HITL 発火（計画未確定） | plan_confirmed=false | `hitl_required=true`・検索前に一時停止・レポート無し | `integration::test_hitl_pause_when_plan_unconfirmed` |
| BL-03 | HITL skip（明示指定） | sources+quality 明示 | 計画確定として検索へ進行 | `unit/test_plan_node.py::test_confirmed_plan_skips_hitl` |
| BL-04 | 空質問の早期終了 | 空白入力 | error・main 未実行・レポート無し | `integration::test_empty_question_early_terminates` |
| BL-05 | ソース部分失敗の gap 降格 | 1 ソース失敗 | 取得済み分でレポート生成・失敗ソースを gap 記録 | `integration::test_partial_source_failure_degrades_to_gaps` |
| BL-06 | 引用品質フィルタ | peer_reviewed_only | news ソース除外 | `unit/test_retrieve_node.py::test_quality_bar_filters_low_tier_sources` |
| BL-07 | no-hallucination（合成段） | 出典なし claim を LLM が返す | gap へ降格しレポートに混入させない | `unit/test_synthesize_node.py::test_unsourced_claim_dropped_to_gap` |
| BL-08 | 全ソース無収穫の RETRY | 全ソース empty | status error（retryable）→ backbone RETRY | `unit/test_retrieve_node.py::test_no_evidence_is_retryable_error` |
| BL-09 | `config["llm"]` が本番経路でノードへ届く | server の config 形＋`config["llm"]` | 実 LLM クライアントが呼ばれレポート生成 | `integration/test_server_llm_injection.py::test_config_llm_over_the_server_config_reaches_the_node` |
| BL-10 | `BaseLLM` 判定が duck typing より先 | `complete()` と `synthesize()` 両方を持つ型 | `BaseLLMSynthesisAdapter` 経由（誤結線しない） | `unit/test_llm_wiring.py::test_isinstance_wins_over_duck_typed_synthesize` |
| BL-11 | 発明された出典 ID がレポートに到達しない | 未取得 ID を引用する finding | finding 破棄・gap に記録 | `unit/test_llm_wiring.py::TestProvenanceResolution` |
| BL-12 | プロバイダ例外が State へ漏れない | 鍵と依頼文を含む例外 | 例外**型名のみ**を記録・トレースバックなし | `unit/test_llm_wiring.py::test_provider_exception_message_never_enters_state` |
| BL-13 | LLM 失敗は縮退せず `status=error` | LLM が送出 | もっともらしい空レポートを success で返さない | `unit/test_llm_wiring.py::test_failure_is_error_not_a_plausible_empty_report` |
| BL-14 | 空レポートは「空である」ことを述べる | findings も gaps も空の応答 | 「支持できる知見なし」を gap に記録 | `unit/test_llm_wiring.py::test_empty_result_still_records_why` |
| BL-15 | 実 LLM の応答揺れ（```json フェンス・散文混じり・null content） | 各形 | フェンス優先で厳密に解析。null は拒否（"None" を回答にしない） | `unit/test_llm_wiring.py::TestBaseLLMSynthesisAdapter` |
| BL-16 | 独自コンストラクタ引数を持たない | 内部 Graph の署名 | `(self, config)` のみ | `unit/test_llm_wiring.py::test_graph_takes_no_bespoke_llm_keyword` |
| BL-17 | 入れ子の認証情報を S-3 でブロック | claim / gap / 取得スニペット / 引用メタデータ内の鍵 | `status=error`・呼び出し元へ到達しない | `unit/test_llm_wiring.py::TestNestedOutputCredentialGate` / `::TestPostProcessNestedGate` |
| BL-18 | 走査不能な値は「安全」ではなく**ブロック** | `set` / `__str__` のみが秘密を持つ型 / 深さ超過 | fail-closed で拒否 | `unit/test_llm_wiring.py::TestNestedScanRobustness` |
| BL-19 | ブロックメッセージが一致値・キー名を復唱しない | 鍵を dict キーに置く | メッセージに秘密が出ない | `unit/test_llm_wiring.py::test_the_message_never_names_the_offending_key` |
| BL-20 | 共有参照・自己参照でも走査が停止する | 分岐する循環構造 | 2 秒以内に完了 | `unit/test_llm_wiring.py::test_a_shared_reference_graph_does_not_explode` |
| BL-21 | confidence は有限かつ 0..1 | `7` / `-0.5` / `NaN` / `10**400` | gap へ降格（クラッシュしない） | `unit/test_llm_wiring.py::TestConfidenceValidation` / `::TestConfidenceOverflow` |
| BL-22 | 引用は文献単位（コネクタ単位でない） | 1 コネクタ 2 文献、claim は 1 件目だけを引用 | 2 件目は Citations にもレポート本文にも出ない | `unit/test_llm_wiring.py::TestCitationGranularity` / `::TestEndToEndCitationGranularity` |
| BL-23 | 素のコネクタ名は解決しない | `supporting_source_ids: ["pubmed"]` | finding 破棄・gap 記録 | `unit/test_llm_wiring.py::TestConnectorWideCitationIsRejectedAtSynthesis` |
| BL-24 | `evidence_id` はサーバ採番・コネクタ値を信用しない | コネクタが `evidence_id: "arxiv#1"` を偽装 | `pubmed#1` に上書き | `unit/test_retrieve_node.py::test_connector_supplied_evidence_id_is_overwritten` |
| BL-25 | `evidence_id` の採番はコネクタごとに振り直す | arxiv 2 件 + pubmed 2 件 | `arxiv#1, arxiv#2, pubmed#1, pubmed#2` | `unit/test_retrieve_node.py::test_numbering_restarts_per_connector` |
| BL-26 | 同一コネクタの重複指定で ID が衝突しない | `sources: ["pubmed","pubmed"]` | コネクタ呼び出しは 1 回・ID は一意 | `unit/test_llm_wiring.py::TestEvidenceIdUniqueness` |
| BL-27 | ID 衝突は fail-closed | 採番関数が同じ ID を返す | `SecurityViolationError` | `unit/test_llm_wiring.py::test_a_duplicate_handle_is_blocked_not_silently_resolved` |
| BL-28 | provenance はカタログと実呼出コネクタが権威 | コネクタが `source_id`/`source_tier` を偽装 | 上書きされる | `unit/test_llm_wiring.py::test_provenance_fields_are_authoritative_not_connector_supplied` |
| BL-29 | 未採番レコードは citable でない（`source_id` へフォールバックしない） | 採番済みと未採番の混在 | コネクタ名は allowed list に出ず、引用しても 0 件 | `unit/test_llm_wiring.py::TestNoConnectorFallback` |
| BL-30 | レンダリング済み引用行が文献ハンドルを示す | claim が `pubmed#2` を引用 | `- [pubmed#2] [tier] ...`・`pubmed#1` は出ない | `unit/test_llm_wiring.py::TestRenderedCitationIdentifiesTheDocument` |

## テスト実行サマリ

- 実行コマンド: `python -m pytest tests/ -v` / `python -m pytest tests/proof_of_boundary/ -v`
- Total: 154 / Pass: 154 / Fail: 0 / Skip: 0（2026-08-28・`agenticstar-agentcore==1.0.0` 実 wheel）
- 備考: SDK v1.0 移行済み。`framework.*` のスタブは廃止済みで、実 wheel に対して実行している。

## 実 LLM 1 回転検証

モックだけで固めた実装は実 LLM を繋いだ瞬間に動かないことがあるため、注入シームへ実 LLM クライアントを刺して 1 回 invoke し、**中身のある出力が出ること**を確認している（`gitproject-dev-flow.md` §14-1-bis）。

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-08-28 |
| 使用 LLM | Azure OpenAI のモデルを `BaseLLM` 実装として注入 |
| 注入シーム | `config[llm]` |
| 実 LLM 呼び出し | 1 回 |
| 出力 | ['<text output: 372字>'] が非空。縮退理由なし |

本番の LLM は注入されたクライアントで、これは検証用シーム。テンプレの `src/` は変更していない（一時テストを生成→実行→削除する方式）。CI には入れていない（外部モデルの鍵が要るため）。

> **2026-08-28 以前の同記録について**: 当時も注入シームは `config[llm]` と記録されていたが、
> 実際にそれが届いていたのは `tests/conftest.py` が `ResearchWorkflowGraph.__init__` を
> monkeypatch する**テスト側だけのブリッジ**があったためで、本番経路には
> `config["llm"]` から `synthesize_node` へ至る道が存在しなかった。ブリッジは
> `src/services/llm_client.py` へ移し、conftest から削除済み。

> ⚠️ **この節は `work/scripts/record_live_llm_in_docs03.py` が上書きする。**
> 同スクリプトを再実行すると、この見出し以降に手で足した内容は消える（2026-08-28 に実際に消した）。
> 実行後は必ず本節以降が残っているか確認すること。

### 実入力（業務中核）での 1 回転

汎用ペイロード（`FakeResearchService` の短いスニペット）は invoke 経路の生存しか示さない。
本 AG の中核はクロスソース統合と claim→出典バインドなので、実データ形状で別途 1 回転した。

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-08-28（引用粒度の修正後に再実行） |
| 入力 | GLP-1 受容体作動薬の心血管アウトカム（3 文献: arxiv メタ解析 1 / pubmed RCT 1 / **無関係な土壌微生物論文 1**。pubmed が 2 件返す形にしてある） |
| 質問 | MACE 低減の有無 ＋ **全死因死亡への影響**（後者はコーパスが答えを持たない） |
| 結果 | `status=success`。`retrieve` が `arxiv#1` / `pubmed#1` / `pubmed#2` を採番。finding が arxiv・pubmed 双方の**文献**を引用（クロスソース統合が成立）。全死因死亡は gap として明示。引用メタデータは全て取得レコード由来。**発明された出典 ID は 0 件** |
| 受入条件 | **無関係な土壌微生物論文（`pubmed#2`）は取得はされるが Citations に載らない** → `cited: ['arxiv#1', 'pubmed#1']`。レンダリング済みレポートにも出現しない ✅ |

#### 修正済み — 引用の粒度（2026-08-28・ユーザー裁定によりこの MR で対応）

`post_process_node._build_citations()` は `source_id`（＝コネクタ識別子）で引用を索引していた。
1 つのコネクタが複数文献を返すと、その `source_id` を引用した claim に対して
**そのコネクタの全文献**が Citations に載った。修正前の実入力検証では pubmed が 2 件返したため、
心血管の claim の引用に土壌微生物論文が混入した（LLM 抜きでも決定論的に再現する）。

**修正**: 文献単位ハンドル `evidence_id`（`{source_id}#{n}`）を `retrieve_node` が
サーバ側で採番し、プロンプト・`supporting_source_ids`・引用索引の全てをその粒度へ移した。
スキーマ変更点は `docs/02_design.md`「出力スキーマの変更履歴」を参照。

**既存テストが検出できなかった理由と対処**: `FakeResearchService` が 1 ソース 1 件しか
返さないため、`source_id` と「その文献」が構造的に区別できなかった。**既定で 1 コネクタ
2 文献（2 件目は意図的に無関係な内容）を返すように作り直し**、テストを組み直した。
コネクタ単位の索引に戻す変異でテストが赤くなることを確認済み。

**粒度変更にあたって同時に塞いだ穴**（敵対レビュー第 3 巡で検出・いずれも実測再現済み）:

| 穴 | 再現 | 対処 |
|---|---|---|
| 同一コネクタを重複指定すると採番が振り出しに戻り、**別文献に同じ ID** が振られる | `sources: ["pubmed","pubmed"]` で `pubmed#1` が 2 文献に発行され、引用が 2 件目を指した | `effective_sources()` が順序保持で重複排除。加えて採番後に**一意性を検査し重複なら fail-closed** |
| `evidence_id` 無しのレコードを `source_id` にフォールバックさせていたため、**コネクタ名が再び citable** になる | 採番済み 1 件＋未採番 2 件の混在で allowed list が `["pubmed#1","pubmed"]` になり、`pubmed` を引用すると未採番の最後の 1 件が引用された | フォールバックを**廃止**。未採番レコードは citable でない（互換のつもりの経路が、直したはずの欠陥をそのまま復活させていた） |
| `source_id`/`source_type`/`source_tier` が `setdefault` だったため、**コネクタが出所とティアを偽装**できる | pubmed コネクタが `source_id: "arxiv", source_tier: "internal_kb"` を返すと引用がそのまま表示された | カタログと実際に呼んだコネクタを権威として**代入で上書き** |

#### 既知の未修正欠陥 — `input_context` が S-2 で走査されない（本改版のスコープ外）

`pre_process_node` の S-2 は `user_input`（質問文）だけを `detect_credentials` に掛け、
`input_context.source_spec` は走査しない。`source_spec.depth` は値の制約がないまま
`research_plan` に入り、応答・checkpoint・HITL payload へ echo される。

- `pre_process_node.py` / `plan_node.py` は develop から**未変更**で、develop でも同一に再現する。
- 入口（S-2）側の話であり、本改版の LLM 配線・出力側（S-3）とは別チャネル。
- 想定される修正は `depth` 等の業務キーを **allowlist** で受けること（denylist ではなく）。
  受理値の定義はドメイン判断なので人間の判断を待つ。

#### 既知の設計論点 — 既定ソース集合に `internal_kb` が含まれる（本改版のスコープ外）

ソース未指定時、`effective_sources()` はカタログ全件（`internal_kb` を含む）を選ぶ。
呼び出し元は `input_context.plan_confirmed=true` を自分で立てられるため、
HITL 計画ゲートは既定選択に対する保護になっていない。結果として VERIFIED_EXTERNAL の
呼び出しで内部 KB 本文が外部 LLM プロバイダのプロンプトへ載りうる。

- エビデンスを LLM で統合すること自体は本 AG の機能そのものだが、
  **既定集合の広さ**は別問題。`internal_kb` を明示選択（または INTERNAL trust）必須にする案がある。
- `policy.py` は develop から未変更。データ分類ポリシーの決定は人間の判断を待つ。

