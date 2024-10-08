from typing import List, Tuple, Iterator

import graphviz

from agent.chain import TaskChain, DatasetChain, ModelChain, PlanChain, PipelineChain
from agent import ResponseStatus


class MLChain:

    def __init__(self, llm, task_names: List, datasets: List, models: List, components: List):
        self.llm = llm
        self.task_chain = TaskChain(llm=self.llm, task_names=task_names)
        self.dataset_chain = DatasetChain(llm=self.llm, datasets=datasets)
        self.model_chain = ModelChain(llm=self.llm, models=models)
        self.plan_chain = PlanChain(llm=self.llm)
        self.pipeline_chain = PipelineChain(llm=self.llm, components=components)

    def call(self, query: str) -> Iterator[Tuple]:
        task = self.task_chain.call(query)
        yield ResponseStatus.STR, f"任务类型:{task}\n"

        dataset = self.dataset_chain.call(query, task)
        yield ResponseStatus.STR, f"数据集:{dataset}\n"

        model = self.model_chain.call(query, task)
        yield ResponseStatus.STR, f"模型:{model}\n"

        steps = self.plan_chain.call(task=task, model=model, dataset=dataset, query=query)
        yield ResponseStatus.STR, f"步骤:{str(steps)}\n"

        res_components = []
        for i, component in enumerate(self.pipeline_chain.call(task=task, model=model, dataset=dataset, steps=steps)):
            yield ResponseStatus.STR, f"{steps[i]} 组件：{component}\n"
            res_components.append(component)

        yield ResponseStatus.IMG, self.draw_flowchart(res_components)
        res = ' -> '.join(res_components)
        yield ResponseStatus.FIN, f"工作流已生成：\n{res}\n"

    @staticmethod
    def draw_flowchart(components: List):
        graph_name = 'flowchart'
        ext = 'png'
        graph_file_name = graph_name + '.' + ext
        graph = graphviz.Digraph(format=ext)
        for component in components:
            graph.node(component)
        for i in range(len(components) - 1):
            graph.edge(components[i], components[i + 1])
        graph.render(graph_name)
        return graph_file_name
