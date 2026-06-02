"""
Ragas 评估模块（最新版 API）
用于评估 RAG 系统的检索质量和生成质量
"""
from typing import List, Dict
import pandas as pd
from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from utils_tool.model_factory import llm , embedding

class RAGASEvaluator:
    """Ragas 评估器"""

    def __init__(self, llm, embeddings):
        print("[RAGASEvaluator] 初始化评估器...")
        self.llm = llm
        self.embeddings = embeddings
        # 新版：指标需要实例化并传入 llm
        self.metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ]
        print(f"[RAGASEvaluator] 初始化完成，共 {len(self.metrics)} 个评估指标")

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> pd.DataFrame:
        """同步执行评估"""
        print(f"[evaluate] 开始评估，共 {len(questions)} 个样本")
        samples = []
        for i, (q, a, c, gt) in enumerate(zip(questions, answers, contexts, ground_truths)):
            print(f"[evaluate] 构建样本 {i+1}: 问题={q[:50]}...")
            sample = SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=c,
                reference=gt
            )
            samples.append(sample)

        dataset = EvaluationDataset(samples=samples)
        print(f"[evaluate] 数据集构建完成，共 {len(samples)} 个样本")

        results = {}
        for metric in self.metrics:
            metric_name = metric.__class__.__name__.lower()
            print(f"[evaluate] 正在计算指标: {metric_name}")
            scores = []
            for i, sample in enumerate(samples):
                print(f"[evaluate]   {metric_name} 样本 {i+1}/{len(samples)}")
                score = metric.single_turn_score(sample)
                scores.append(score)
                print(f"[evaluate]   {metric_name} 样本 {i+1} 得分: {score}")
            results[metric_name] = scores
            print(f"[evaluate] 指标 {metric_name} 计算完成，平均分: {sum(scores)/len(scores):.4f}")

        print("[evaluate] 评估完成")
        return pd.DataFrame(results)


evaluator = RAGASEvaluator(llm, embedding)
# # 使用示例
# if __name__ == "__main__":
#     print("=" * 60)
#     print("Ragas 评估模块测试")
#     print("=" * 60)
#
#
#
#     print("[main] 初始化 LLM 和 Embeddings...")
#     print("[main] 初始化完成")
#
#     evaluator = RAGASEvaluator(llm, embedding)
#
#     print("[main] 开始评估...")
#     df = evaluator.evaluate(
#         questions=["恐龙是怎么被命名的？"],
#         answers=["1841年由理查德·欧文命名，意为恐怖的蜥蜴"],
#         contexts=[["1841年，英国科学家理查德·欧文创造了Dinosauria一词，意为恐怖的蜥蜴"]],
#         ground_truths=["1841年，理查德·欧文命名恐龙，意为恐怖的蜥蜴"]
#     )
#     print("[main] 评估结果:")
#     print(df)