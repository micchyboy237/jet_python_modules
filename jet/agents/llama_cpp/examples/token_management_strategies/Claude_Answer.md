Here's a rundown of the main ways AI agents deal with running out of context window space, in plain terms:

**1. Sliding window / truncation**
The simplest fix — just drop the oldest messages and keep only the most recent N turns, with core instructions "pinned" at the top so they never get dropped. Sliding window approaches simply manage memory limits: they drop the oldest messages, making room for the newest ones, with only core instructions being "locked" at the top of the context. It's cheap and fast, but the agent literally forgets anything older than the window — like a coworker with 10-minute memory.

**2. Memory message consolidation / summarization**
Instead of deleting old messages, an LLM call condenses them into a shorter running summary that keeps the "gist" while freeing up space. This approach maintains a running summary of the entire conversation — after each exchange, the history (including the previous summary and the new messages) is sent to an LLM, which generates an updated, consolidated summary. The tradeoff: it incurs an LLM call for summarization at every step, adding latency and cost, and summaries can lose nuance. Some frameworks blend this with a small raw buffer — LangChain's ConversationSummaryBufferMemory provides a hybrid solution that combines a raw buffer of recent messages with a progressively updated summary of older ones, triggered once a token threshold is hit. Research systems like Generative Agents take this further with layered summaries — early works like Generative Agents utilize a hierarchical summarization strategy, recursively summarizing daily activities from low-level details to high-level reflections.

**3. Recursive / hierarchical summarization**
A step up from basic summarization — instead of re-summarizing everything each time, it periodically compresses chunks of old history into layered summaries (like repeatedly re-compressing a JPEG). Instead of removing the distant past as sliding windows would do, recursive summarization consists of periodically compressing old messages into a summary. This keeps a long-term "gist" alive without re-processing the whole history every turn.

**4. Structured state management**
Rather than treating memory as a blob of chat messages, the agent keeps an explicit structured state object (task status, variables, file list, etc.) that gets updated instead of endlessly appended to. This avoids re-deriving facts from scratch each turn and keeps token usage more predictable.

**5. Ephemeral context via retrieval (RAG-style memory)**
Old context isn't kept in the live prompt at all — it's stored externally (often in a vector database) and pulled back in only when relevant, via semantic search. A VectorStoreRetrieverMemory providing access to the full, long-term history or related documents via semantic search is a common example. This scales well for huge histories, but has a real weakness: if the retrieval step misses something important, the agent never even knows it existed ("retrieval blind spots").

**6. Dynamic context routing**
The agent (or a controller around it) decides on the fly what type of memory access a given step needs — recent turns, a summary, a retrieved doc, or nothing — instead of using one blanket strategy for everything.

**7. Observation masking**
Specific to tool-using/coding agents: instead of summarizing, you hide or collapse old tool outputs (e.g., a full file listing from 20 steps ago) that are no longer needed, while keeping the reasoning text intact. This hybrid approach combines the strengths of both observation masking and LLM summarization — cheap for simple tasks, only paying for real summarization when the task is genuinely complex.

**8. Tiered / OS-style virtual memory**
Some systems mimic how computer operating systems manage RAM vs. disk. MemGPT addresses the context limit by implementing a hierarchical virtual context management system similar to an operating system, paging information in and out of the active context as needed, with a separate long-term store underneath.

**9. Dependency-aware / selective memory construction**
Rather than keeping things by recency, this approach tracks _which_ earlier steps a current step actually depends on and only pulls in those. ContextWeaver supports dependency-based construction and traversal that link each step to the earlier steps it relies on, plus compact dependency summarization that condenses root-to-step reasoning paths into reusable units — useful in coding agents where a runtime error might depend on something from many steps back, not just the last few turns.

**10. Agent-controlled memory operations**
The newest direction: instead of a fixed rule engine deciding when to summarize or forget, the agent itself is given memory as a "tool" it can call. AgeMem exposes memory operations as tool-based actions, enabling the LLM agent to autonomously decide what and when to store, retrieve, update, summarize, or discard information.

A caveat worth knowing: summarization isn't automatically the best choice everywhere. In one small-model training experiment, sliding-window context performed best, while summary-based context actually underperformed — the model-generated summaries introduced noise. So the "best" strategy is genuinely task-dependent.

## Use case summary

| Use case                                           | Best-fit strategy                                    | Why                                                                 |
| -------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------- |
| Short Q&A / simple chatbot                         | Sliding window                                       | Simple, cheap, no info worth preserving long-term                   |
| Long customer support chat                         | Summarization (buffer + summary)                     | Keeps gist of issue without ballooning tokens                       |
| Multi-day/session personal assistant               | Hierarchical/recursive summarization                 | Preserves long-term patterns without re-processing everything       |
| Coding agent (small/medium tasks)                  | Observation masking                                  | Cuts noisy tool output cheaply, keeps reasoning intact              |
| Coding agent on large repos (e.g. SWE-bench style) | Dependency-aware memory                              | Old but relevant steps (e.g. an error 30 turns back) stay reachable |
| Agent needing huge knowledge base recall           | Retrieval-based (RAG) memory                         | Scales to unbounded history, pulls in only what's relevant          |
| Autonomous long-running agent (task planner)       | Structured state management                          | Predictable, explicit state beats messy chat logs                   |
| Personalized assistant with user preferences       | Tool-based agent-controlled memory (e.g. Mem0-style) | Agent decides what's worth remembering, adapts over time            |
| Reinforcement-learning / training pipelines        | Sliding window (per Agent-R1 finding)                | Summaries can add noise that hurts training signal                  |

## Common combined strategies

| Combination                               | What it looks like                                                                           | Used by / example                           |
| ----------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Recent buffer + running summary           | Keep last few raw turns, summarize everything older                                          | LangChain `ConversationSummaryBufferMemory` |
| Summary + retrieval                       | Short-term summary for flow, vector store for deep recall                                    | Typical production RAG-memory setup         |
| Observation masking + LLM summarization   | Mask irrelevant tool output cheaply, summarize only when task complexity demands it          | JetBrains hybrid coding-agent approach      |
| Tiered memory (hot/warm/cold)             | Active context (hot) + summarized mid-term (warm) + full archive (cold, retrieved on demand) | MemGPT-style virtual memory                 |
| Structured state + token-limited assembly | Explicit state object, then strict token cap applied before sending to LLM                   | General production agent design pattern     |
| Dependency graph + summarization          | Track step dependencies, summarize only the dependency path, not full history                | ContextWeaver                               |

If it'd help, I can sketch a quick diagram showing how these layers (hot/warm/cold memory) typically fit together in a real agent pipeline.
