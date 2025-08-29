import math
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from openevals.llm import create_llm_as_judge
from openevals.types import ChatCompletionMessage


class ConversationEvaluator:
    """
    A reusable evaluator for conversation quality assessment using LLM-as-a-judge.

    This class provides standardized evaluation methods using Answer Accuracy
    and Answer Relevance metrics based on Nvidia Answer Accuracy and Relevance metrics
    adapted from: https://github.com/explodinggradients/ragas/blob/40eafca1cbc9a8a20a161666c28091df8b03df9c/src/ragas/metrics/_nv_metrics.py.
    """

    def __init__(
        self,
        evaluator_client: Optional[BaseChatModel] = None,
        retry_count: int = 2,
    ):
        """
        Initialize the conversation evaluator.

        Args:
            evaluator_client: Optional LangChain chat model to use for evaluation (default: None)
            retry_count: Number of retries if evaluation fails (default: 2)
        """
        self.evaluator_client = evaluator_client
        self.retry_count = retry_count

        # Answer Accuracy templates based on RAGAS
        template_accuracy1 = (
            "Instruction: You are a world class state of the art assistant for rating "
            "a User Answer given a Question. The Question is completely answered by the Reference Answer.\n"
            "Say 4, if User Answer is full contained and equivalent to Reference Answer "
            "in all terms, topics, numbers, metrics, dates and units.\n"
            "Say 2, if User Answer is partially contained and almost equivalent to Reference Answer "
            "in all terms, topics, numbers, metrics, dates and units.\n"
            "Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, topics, "
            "numbers, metrics, dates and units or the User Answer do not answer the question.\n"
            "Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the instructions above.\n"
            "### Question: {query}\n"
            "### {answer0}: {sentence_inference}\n"
            "### {answer1}: {sentence_true}\n"
            "The rating is:\n"
        )

        template_accuracy2 = (
            "I will rate the User Answer in comparison to the Reference Answer for a given Question.\n"
            "A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, covering all aspects, topics, numbers, metrics, dates, and units.\n"
            "A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, with minor discrepancies in some areas.\n"
            "A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to the Reference Answer, or it fails to address the Question.\n"
            "I will provide the rating without any explanation or justification, adhering to the following scale: 0 (no match), 2 (partial match), 4 (exact match).\n"
            "Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.\n\n"
            "Question: {query}\n\n"
            "{answer0}: {sentence_inference}\n\n"
            "{answer1}: {sentence_true}\n\n"
            "Rating: "
        )

        # Answer Relevance templates - evaluate how relevant the response is to the user's question
        template_relevance1 = (
            "### Instructions\n\n"
            "You are a world class expert designed to evaluate how relevant an Answer is "
            "to the User's Question.\n"
            "Your task is to determine if the Answer properly addresses and is relevant to the Question.\n"
            "Do not rely on your previous knowledge about the Question.\n"
            "Use only what is written in the Answer and in the Question.\n"
            "Follow the instructions below:\n"
            "0. If the answer does not address or is not relevant to the question, say 0.\n"
            "1. If the answer partially addresses or is somewhat relevant to the question, say 1.\n"
            "2. If the answer fully addresses and is highly relevant to the question, say 2.\n"
            "You must provide the relevance score of 0, 1, or 2, nothing else.\nDo not explain.\n"
            "### Question: {user_question}\n\n"
            "### Answer: {assistant_answer}\n\n"
            "Do not try to explain.\n"
            "Analyzing Answer and Question, the Relevance score is "
        )

        template_relevance2 = (
            "As a specially designed expert to assess the relevance score of a given Answer in relation to a User's Question, "
            "my task is to determine the extent to which the Answer addresses and is relevant to the Question. "
            "I will rely solely on the information provided in the Answer and Question, and not on any prior knowledge.\n\n"
            "Here are the instructions I will follow:\n"
            "* If the Answer does not address or is not relevant to the Question, I will respond with a relevance score of 0.\n"
            "* If the Answer partially addresses or is somewhat relevant to the Question, I will respond with a relevance score of 1.\n"
            "* If the Answer fully addresses and is highly relevant to the Question, I will respond with a relevance score of 2.\n\n"
            "### Question: {user_question}\n\n"
            "### Answer: {assistant_answer}\n\n"
            "Do not try to explain.\n"
            "Based on the provided Question and Answer, the Relevance score is ["
        )

        self.accuracy_evaluator1 = create_llm_as_judge(
            prompt=template_accuracy1,
            choices=[0, 2, 4],
            judge=self.evaluator_client,
        )

        self.accuracy_evaluator2 = create_llm_as_judge(
            prompt=template_accuracy2,
            choices=[0, 2, 4],
            judge=self.evaluator_client,
        )

        self.relevance_evaluator1 = create_llm_as_judge(
            prompt=template_relevance1,
            choices=[0, 1, 2],
            judge=self.evaluator_client,
        )

        self.relevance_evaluator2 = create_llm_as_judge(
            prompt=template_relevance2,
            choices=[0, 1, 2],
            judge=self.evaluator_client,
        )

    async def evaluate_trajectory(
        self,
        trajectory: List[ChatCompletionMessage],
        expected_answers: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a conversation trajectory using Answer Accuracy and Answer Relevance metrics.

        Args:
            trajectory: List of conversation turns with role and content
            expected_answers: Optional list of expected answers for accuracy evaluation

        Returns:
            List of evaluation results with keys, scores, and comments
        """
        evaluator_results = []

        if not trajectory:
            return self._get_empty_trajectory_results()

        # Extract questions and responses from trajectory
        questions_and_responses = self._extract_qa_pairs(trajectory)

        if not questions_and_responses:
            return self._get_empty_trajectory_results()

        # Evaluate each question-answer pair for relevancy
        for question, response in questions_and_responses:
            relevance_result = await self._evaluate_answer_relevance(question, response)
            if relevance_result is not None:
                evaluator_results.append(relevance_result)

        # Answer Accuracy evaluation - evaluate first question against all responses, take max score
        if expected_answers and questions_and_responses:
            first_question = questions_and_responses[0][0]  # Get the first question
            expected_answer = expected_answers[0]  # Use the first expected answer

            accuracy_scores = []

            # Evaluate the first question against each response in the conversation
            for _, response in questions_and_responses:
                accuracy_result = await self._evaluate_answer_accuracy(
                    first_question, response, expected_answer
                )
                if accuracy_result is not None and "score" in accuracy_result:
                    accuracy_scores.append(accuracy_result["score"])

            # Take the maximum accuracy score across all responses
            if accuracy_scores:
                max_accuracy_score = max(accuracy_scores)
                evaluator_results.append(
                    {
                        "key": "answer_accuracy",
                        "score": max_accuracy_score,
                    }
                )
        print(evaluator_results)
        # If no evaluations were performed, return empty results
        if not evaluator_results:
            return self._get_empty_trajectory_results()

        return evaluator_results

    def _extract_qa_pairs(
        self, trajectory: List[ChatCompletionMessage]
    ) -> List[tuple[str, str]]:
        """
        Extract question-answer pairs from conversation trajectory.

        Args:
            trajectory: List of conversation turns

        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []
        current_question = None

        for turn in trajectory:
            role = turn.get("role", "")
            content = turn.get("content", "")

            if role == "user":
                current_question = content
            elif role == "assistant" and current_question:
                qa_pairs.append((current_question, content))
                current_question = None

        return qa_pairs

    async def _evaluate_answer_accuracy(
        self, question: str, user_answer: str, reference_answer: str
    ) -> Dict[str, Any] | None:
        """
        Evaluate answer accuracy using dual-template RAGAS approach.

        Args:
            question: The user's question
            user_answer: The model's response
            reference_answer: The expected/reference answer

        Returns:
            Evaluation result dict or None if evaluation fails
        """
        try:
            score_ref_gen = score_gen_ref = float("nan")

            # First template evaluation
            for _ in range(self.retry_count):
                result1 = self.accuracy_evaluator1(
                    query=question,
                    answer0="User Answer",
                    answer1="Reference Answer",
                    sentence_inference=user_answer,
                    sentence_true=reference_answer,
                    outputs={},
                )

                score_ref_gen = self._process_accuracy_score(result1)
                if not math.isnan(score_ref_gen):
                    break

            # Second template evaluation (swap roles)
            for _ in range(self.retry_count):
                result2 = self.accuracy_evaluator2(
                    query=question,
                    answer0="Reference Answer",
                    answer1="User Answer",
                    sentence_inference=reference_answer,
                    sentence_true=user_answer,
                    outputs={},
                )

                score_gen_ref = self._process_accuracy_score(result2)
                if not math.isnan(score_gen_ref):
                    break

            # Average the scores
            final_score = self._average_scores(score_ref_gen, score_gen_ref)

            return {
                "key": "answer_accuracy",
                "score": final_score,
            }

        except Exception as e:
            print(f"Warning: Answer accuracy evaluation failed: {e}")
            return None

    async def _evaluate_answer_relevance(
        self, user_question: str, assistant_answer: str
    ) -> Dict[str, Any] | None:
        """
        Evaluate answer relevance using dual-template approach.

        Args:
            user_question: The user's question
            assistant_answer: The assistant's response

        Returns:
            Evaluation result dict or None if evaluation fails
        """
        if not user_question.strip() or not assistant_answer.strip():
            return None

        try:
            score1 = score2 = float("nan")

            # First template evaluation
            for _ in range(self.retry_count):
                result1 = self.relevance_evaluator1(
                    user_question=user_question,
                    assistant_answer=assistant_answer,
                    outputs={},
                )

                score1 = self._process_relevance_score(result1)
                if not math.isnan(score1):
                    break

            # Second template evaluation
            for _ in range(self.retry_count):
                result2 = self.relevance_evaluator2(
                    user_question=user_question,
                    assistant_answer=assistant_answer,
                    outputs={},
                )

                score2 = self._process_relevance_score(result2)
                if not math.isnan(score2):
                    break

            # Average the scores
            final_score = self._average_scores(score1, score2)

            return {
                "key": "answer_relevance",
                "score": final_score,
            }

        except Exception as e:
            print(f"Warning: Answer relevance evaluation failed: {e}")
            return None

    def _process_accuracy_score(self, response: Any) -> float:
        """Process answer accuracy score from evaluator response."""
        if isinstance(response, dict):
            raw_score = response.get("score", 0)
        else:
            # Try to extract score from string response
            response_str = str(response)
            for score in [4, 2, 0]:
                if str(score) in response_str:
                    raw_score = score
                    break
            else:
                return float("nan")

        # Convert to 0-1 scale
        return raw_score / 4.0

    def _process_relevance_score(self, response: Any) -> float:
        """Process context relevance score from evaluator response."""
        if isinstance(response, dict):
            raw_score = response.get("score", 0)
        else:
            # Try to extract score from string response
            response_str = str(response)
            for score in [2, 1, 0]:
                if str(score) in response_str:
                    raw_score = score
                    break
            else:
                return float("nan")

        # Convert to 0-1 scale
        return raw_score / 2.0

    def _average_scores(self, score1: float, score2: float) -> float:
        """Average two scores, handling NaN values."""
        if not math.isnan(score1) and not math.isnan(score2):
            return (score1 + score2) / 2
        elif not math.isnan(score1):
            return score1
        elif not math.isnan(score2):
            return score2
        else:
            return float("nan")

    def _format_conversation(self, trajectory: List[ChatCompletionMessage]) -> str:
        """
        Convert trajectory to a readable conversation format.

        Args:
            trajectory: List of conversation turns

        Returns:
            Formatted conversation string
        """
        conversation_text = ""
        for turn in trajectory:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            conversation_text += f"{role.title()}: {content}\n"
        return conversation_text

    def _get_empty_trajectory_results(self) -> List[Dict[str, Any]]:
        """
        Get default results for empty trajectories.

        Returns:
            List of default evaluation results for empty conversations
        """
        return [
            {
                "key": "answer_accuracy",
                "score": 0.0,
            },
            {
                "key": "answer_relevance",
                "score": 0.0,
            },
        ]
