"""
查询扩展模块

功能：
1. LLM 改写：生成多个语义等价的查询变体
2. 关键词提取：提取核心关键词
3. 返回带权重的查询列表

策略：
- 原始查询权重: 0.5
- 扩展查询总权重: 0.5 (LLM改写的多个变体平分这部分权重)
- 关键词查询作为扩展查询的一部分
"""

from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from utils_tool.config_handler import config


class QueryExpander:
    """查询扩展器"""

    def __init__(
            self,
            llm: ChatOpenAI = None,
            cfg=None
    ):
        print("[QueryExpander] 初始化...")
        self.config = cfg or config
        self.llm = llm or ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3.6-plus"
        )
        self._init_prompts()
        print("[QueryExpander] 初始化完成")

    def _init_prompts(self):
        """初始化 Prompt 模板"""
        print("[QueryExpander] 初始化 Prompt 模板...")
        rewrite_template = """请将用户的问题改写成 {num} 个语义等价但表达方式不同的查询。
要求：每行一个查询，不要编号。

原始问题：{query}

改写结果："""

        self.rewrite_prompt = PromptTemplate(
            input_variables=["query", "num"],
            template=rewrite_template
        )

        keyword_template = """请从用户问题中提取 {max_keywords} 个最核心的关键词。
要求：只提取实义词，用空格分隔。

问题：{query}

关键词："""

        self.keyword_prompt = PromptTemplate(
            input_variables=["query", "max_keywords"],
            template=keyword_template
        )
        print("[QueryExpander] Prompt 模板初始化完成")

    def expand(self, query: str) -> List[tuple]:
        """
        扩展查询，返回带权重的查询列表

        Returns:
            [(查询文本, 权重), ...]
        """
        print(f"[expand] 开始扩展查询: {query}")
        queries = [(query, 0.5)]
        expanded_queries = []

        print("[expand] 执行 LLM 改写...")
        rewrites = self._llm_rewrite(query)
        expanded_queries.extend(rewrites)
        print(f"[expand] LLM 改写生成 {len(rewrites)} 个变体")

        print("[expand] 提取关键词...")
        keywords = self._extract_keywords(query)
        if keywords:
            expanded_queries.append(keywords)
            print(f"[expand] 关键词提取: {keywords}")
        else:
            print("[expand] 未提取到关键词")

        if expanded_queries:
            per_weight = 0.5 / len(expanded_queries)
            print(f"[expand] 扩展查询共 {len(expanded_queries)} 个，每个权重: {per_weight:.4f}")
            for eq in expanded_queries:
                queries.append((eq, per_weight))
        else:
            print("[expand] 无扩展查询")

        print(f"[expand] 最终查询列表: {len(queries)} 个")
        for q, w in queries:
            print(f"  - 权重 {w}: {q[:50]}...")
        return queries

    def _llm_rewrite(self, query: str) -> List[str]:
        """LLM 生成查询变体"""
        print(f"[_llm_rewrite] 生成查询变体，原始查询: {query[:50]}...")
        try:
            chain = self.rewrite_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "query": query,
                "num": self.config.NUM_REWRITES
            })
            rewrites = [line.strip() for line in response.strip().split("\n") if line.strip()]
            print(f"[_llm_rewrite] 生成 {len(rewrites)} 个变体")
            return rewrites[:self.config.NUM_REWRITES]
        except Exception as e:
            print(f"[_llm_rewrite] LLM 改写失败: {e}")
            return []

    def _extract_keywords(self, query: str) -> str:
        """提取关键词"""
        print(f"[_extract_keywords] 提取关键词，查询: {query[:50]}...")
        try:
            chain = self.keyword_prompt | self.llm | StrOutputParser()
            keywords = chain.invoke({
                "query": query,
                "max_keywords": self.config.MAX_KEYWORDS
            })
            result = keywords.strip()
            print(f"[_extract_keywords] 提取结果: {result}")
            return result
        except Exception as e:
            print(f"[_extract_keywords] 关键词提取失败: {e}")
            return ""