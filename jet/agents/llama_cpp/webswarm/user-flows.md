### WebSwarm User Flows

#### Flow 1: Atomic Fact Lookup (Atom Mode)

_Triggered when a node’s local objective is a narrow, well-defined fact query._

1.  Parent node delegates subtask with `verb="atom"` and local objective $q_v$.
2.  AtomAgent receives the objective and enters a ReAct loop.
3.  Agent calls `search(query)` → retrieves top-5 snippets via Serper.
4.  Agent calls `fetch_url(url)` → retrieves summarized page content via Jina.
5.  Agent reasons over evidence; if insufficient, generates new search query and repeats from step 3.
6.  When satisfied, agent returns structured result $r_v$ to parent node.
7.  Parent updates its evidence set $R_v \leftarrow R_v \cup \{r_v\}$ and decides whether to continue or aggregate.

#### Flow 2: Deep Iterative Search & Verification (Deep Mode)

_Triggered when a node must identify an unknown entity through multi-constraint reasoning._

1.  Parent node delegates subtask with `verb="deep"` and constraint-rich objective.
2.  DeepAgent initializes serial searcher-verifier loop.
3.  **Searcher Phase**: Searcher agent proposes candidate(s) based on current clues using atom-style search.
4.  **Verifier Phase**: Independent verifier agent checks each candidate against _all_ constraints using separate search/browsing.
5.  If verifier rejects all candidates, searcher receives rejection feedback and shifts exploration perspective; return to step 3.
6.  If verifier accepts a candidate, DeepAgent returns verified result $r_v$ to parent.
7.  Parent evaluates result; may request deeper verification or accept and aggregate.

#### Flow 3: Wide Parallel Collection (Wide Mode)

_Triggered when a node must collect structured information across multiple entities/dimensions._

1.  Parent node delegates subtask with `verb="wide"` and collection objective.
2.  **Web-Probing Sub-flow** (if enabled):
    - WebProbingAgent performs lightweight search/read on objective $q_v$.
    - Returns structure hint $h_v$ (e.g., "info organized by brand," "concentrated in hub page").
    - WideAgent uses $h_v$ to determine expansion dimension and granularity.
3.  WideAgent generates child delegations $\{(q_i, m_i)\}$ along the chosen dimension.
4.  **Experience Transfer Sub-flow** (if enabled):
    - Execute first $k$ scout children (typically 2).
    - Extract process experience $k_v$ (reliable sources, query templates, failure paths) from scout trajectories.
    - Inject $k_v$ into remaining sibling children's context.
5.  Dispatch all children **in parallel**; children may be atom, deep, entity_collect, or nested wide nodes.
6.  Collect all child results $R_v = \bigcup \{r_i\}$.
7.  WideAgent aggregates results into structured output (e.g., table) and returns to parent.

#### Flow 4: Open-Set Entity Enumeration (Entity_Collect Mode)

_Triggered when the target set boundary is unknown and completeness matters._

1.  Parent node delegates subtask with `verb="entity_collect"` and enumeration objective.
2.  EntityCollectAgent dispatches multiple atom searchers **in parallel**, each using a different recall strategy/perspective (e.g., different query formulations, source types).
3.  Merge all recalled candidates; deduplicate.
4.  For low-confidence candidates, spawn verification sub-tasks (atom or deep) to confirm membership.
5.  Return verified entity set $r_v$ to parent.
6.  Parent may request additional recall paths if coverage seems insufficient.

#### Flow 5: Root Task Completion & Answer Submission

_The top-level flow that terminates the recursive tree._

1.  RootAgent receives original user query $q_0$.
2.  RootAgent analyzes task type and initiates first delegation (choosing appropriate verb).
3.  Recursive delegation tree unfolds via Flows 1–4 as intermediate evidence reveals new subtasks.
4.  At each level, parent nodes evaluate returned evidence and decide: expand further, revise direction, or terminate branch.
5.  When root determines sufficient evidence has been collected, it synthesizes final answer $a = \text{Aggregate}(q_0, R_{v_0})$.
6.  Root calls `submit_answer(a)` through ToolEnv.
7.  TaskManager evaluates answer against benchmark ground truth; logs trajectory, reward, and tool statistics.
