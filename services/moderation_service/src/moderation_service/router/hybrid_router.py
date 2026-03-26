from embedding_service.engine import EmbeddingEngine
from langgraph.graph import END, StateGraph
from llm_service.llm_client import LLMClient
from llm_service.reasoning import build_prompt
from moderation_service.classifier.inference import ToxicityEngine
from moderation_service.router.schemas import ModerationResult, ModerationState


class HybridRouter:
    def __init__(
        self,
        model_path: str,
        embedding_index_path: str,
        llm_client: LLMClient,
        high_threshold: float = 0.8,
    ):
        self.ml_engine = ToxicityEngine(model_path)
        self.embedding_engine = EmbeddingEngine(embedding_index_path)
        self.llm = llm_client

        self.high_threshold = high_threshold

        self.graph = self._build_graph()

    def route(self, text: str) -> ModerationResult:
        state = self.graph.invoke({"text": text})

        return ModerationResult(
            label=state["final_label"],
            source=state["source"],
            probability=state.get("probability"),
            llm_confidence=state.get("llm_confidence"),
            reason=state.get("reason"),
        )

    def _build_graph(self):
        graph = StateGraph(ModerationState)

        graph.add_node("ml", self._ml_node)
        graph.add_node("faiss", self._faiss_node)
        graph.add_node("llm", self._llm_node)

        graph.add_node("high", self._high_conf_node)
        graph.add_node("llm_final", self._llm_final_node)

        graph.set_entry_point("ml")

        graph.add_conditional_edges(
            "ml",
            self._route_decision,
            {
                "high": "high",
                "uncertain": "faiss",
            },
        )

        graph.add_edge("faiss", "llm")
        graph.add_edge("llm", "llm_final")

        graph.add_edge("high", END)
        graph.add_edge("llm_final", END)

        return graph.compile()

    def _ml_node(self, state: ModerationState):
        result = self.ml_engine.predict(state["text"])

        return {
            "probability": result["probability"],
            "label": result["label"],
            "reason": result["reason"],
        }

    def _faiss_node(self, state: ModerationState):
        results = self.embedding_engine.search(state["text"], top_k=5)

        return {"similar_examples": [r.text for r in results]}

    def _llm_node(self, state: ModerationState):
        prompt = build_prompt(
            text=state["text"],
            examples=state.get("similar_examples", []),
        )

        response = self.llm.complete(prompt)

        return {
            "llm_label": response["label"],
            "llm_confidence": response["confidence"],
            "llm_reason": response.get("reason", "llm: no reason"),
        }

    def _high_conf_node(self, state: ModerationState):
        return {
            "final_label": "toxic",
            "source": "ml",
            "reason": state.get("reason"),
        }

    def _llm_final_node(self, state: ModerationState):
        return {
            "final_label": state["llm_label"],
            "source": "llm",
            "reason": state.get("llm_reason"),
        }

    def _route_decision(self, state: ModerationState) -> str:
        prob = state["probability"]
        print("PROB:", prob)

        if prob >= self.high_threshold:
            return "high"

        return "uncertain"
