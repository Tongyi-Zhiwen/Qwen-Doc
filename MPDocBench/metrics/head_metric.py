# modify from https://github.com/icip-cas/READoc/blob/main/src/evaluation/utils.py#L129
import re
from apted import APTED, Config
from Levenshtein import distance


class TedsHeadNode:
    def __init__(self, content, depth=0, order=0, parent=None, children=None):
        self.content = content
        self.depth = depth
        self.parent = parent
        self.order = order
        self.children = children if children else []

    def __repr__(self):
        return f"TedsHeadNode(content={self.content!r}, depth={self.depth}, order={self.order})"

    def print_tree(self, indent=0):
        print("  " * indent + f"[{self.depth}] {self.content}")
        for child in self.children:
            child.print_tree(indent + 1)


class TedsHeadConfig(Config):

    def rename(self, node1, node2):
        """node1:pred, node2:label"""
        if node1.content or node2.content:
            max_len = max(len(node1.content), len(node2.content))
            return distance(node1.content, node2.content) / max_len
            '''
            if distance(node1.content, node2.content) / max_len < 0.15:
                return 0.0
            return 1.0
            '''
        return 0.0



class HeadTEDS(object):
    def __init__(self) -> None:
        pass

    def count_nodes(self, root):
        """统计树的节点总数（含虚拟根节点，保证至少为1，避免除零）"""
        count = 0
        stack = [root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)
        return count

    def evaluate(self, node_list_pred, node_list_gold):
        if isinstance(node_list_pred, list) and (len(node_list_pred) == 0 or isinstance(node_list_pred[0], str)):
            pred_tree = self.build_md_head_tree(node_list_pred)
        elif isinstance(node_list_pred, list) and len(node_list_pred) == 2 and isinstance(node_list_pred[0], list) and isinstance(node_list_pred[1], dict):
            pred_tree = self.build_json_head_tree(node_list_pred)
        else:
            print(node_list_pred)
            raise TypeError('Invalid node list')

        if isinstance(node_list_gold, list) and (len(node_list_gold) == 0 or isinstance(node_list_gold[0], str)):
            gold_tree = self.build_md_head_tree(node_list_gold)
        elif isinstance(node_list_gold, list) and len(node_list_gold) == 2 and isinstance(node_list_gold[0], list) and isinstance(node_list_gold[1], dict):
            gold_tree = self.build_json_head_tree(node_list_gold)
        else:
            print(node_list_gold)
            raise TypeError('Invalid node list')
            
        tree_distance = APTED(pred_tree, gold_tree, TedsHeadConfig()).compute_edit_distance()   
        pred_count = self.count_nodes(pred_tree)
        gold_count = self.count_nodes(gold_tree)
        teds = 1.0 - (float(tree_distance) / max(gold_count, pred_count))
        return teds, self.get_max_depth(gold_tree)

    def get_max_depth(self, node):
        """返回节点（root）下的最大深度（root 不计入深度）。
        如果 root 没有子节点，返回 0。
        """
        children = getattr(node, 'children', None)
        if not children:
            return 0
        # 对每个子节点递归求深度（子节点作为 depth=1 开始计数）
        def _depth(n):
            ch = getattr(n, 'children', None)
            if not ch:
                return 1
            return 1 + max(_depth(c) for c in ch)
        return max(_depth(c) for c in children)
    def build_md_head_tree(self, node_list):
        """
        根据 Markdown 格式的标题字符串列表构建标题树。

        Args:
            node_list (list[str]): 标题字符串列表，每个元素为一个 Markdown 标题行，支持以下两种格式：
                - 标准格式："## 标题文本"（# 与文本之间有空格，# 数量表示层级）
                - 非标准格式："##标题文本"（# 与文本之间无空格，# 数量同样表示层级）
                注意：若某行不含任何 #，则视为无效节点并跳过。

        Returns:
            TedsHeadNode: 构建好的标题树的根节点（content 为 'Root'，depth 为 0），
                          其 children 按照原始列表顺序排列的各层级标题节点。
        """
        node_stack = [TedsHeadNode('root', 0, 0)]
        
        for idx, node in enumerate(node_list):
            node = node.strip()
            if re.match(r'^#+\s', node):
                label, title = node.split(' ', 1)
                level = len(label)
                title = title.strip()
            else:
                level = node.count('#')
                title = node.replace('#', '').strip()
                if level == 0:
                    print(f"Invalid node idx {idx} has no level: {node}")
                    continue

            cur_node = TedsHeadNode(title, level, idx + 1)
            while level <= node_stack[-1].depth:
                node_stack.pop()
            cur_node.parent = node_stack[-1]
            node_stack[-1].children.append(cur_node)
            node_stack.append(cur_node)
            
        return node_stack[0]
    
    def build_json_head_tree(self, node_list):
        """
        根据 JSON 格式的标题节点列表构建标题树。

        Args:
            node_list (list): 包含两个元素的列表：
                - node_list[0] (list[dict]): 关系列表，每个元素描述一条父子关系，格式为：
                    {
                        "source_anno_id": str,  # 父节点的标注 ID
                        "target_anno_id": str   # 子节点的标注 ID
                    }
                - node_list[1] (dict): 节点信息字典，key 为节点的 anno_id，value 格式为：
                    {
                        "text": str,      # 节点的标题文本内容
                        "anno_id": str    # 节点的唯一标注 ID，用于排序
                    }

        Returns:
            TedsHeadNode: 构建好的标题树的根节点（content 为 'root'，depth 为 0），
                          其 children 为所有无父节点的顶层标题节点。
        """
        # [[{source_id: xxx, target_id: xxx}....], {}]
        relations, node_items = node_list[0], node_list[1]

        root = TedsHeadNode('root', 0, 0)
        node_map = {'root': root}

        for relation in relations:
            source_anno_id = relation['source_anno_id']
            target_anno_id = relation['target_anno_id']
            if source_anno_id not in node_map:
                node_map[source_anno_id] = TedsHeadNode(node_items[source_anno_id]["text"], depth=0, order=node_items[source_anno_id]["anno_id"])
            if target_anno_id not in node_map:
                node_map[target_anno_id] = TedsHeadNode(node_items[target_anno_id]["text"], depth=0, order=node_items[target_anno_id]["anno_id"])

 
        for relation in relations:
            source_id = relation['source_anno_id']
            target_id = relation['target_anno_id']
            parent_node = node_map[source_id]
            child_node  = node_map[target_id]
            parent_node.children.append(child_node)
            child_node.parent = parent_node

        for node_id, node in node_map.items():
            if node_id == 'root':
                continue
            # 没有父节点的节点 = 根节点的直接子节点
            if node.parent is None:
                root.children.append(node)
                node.parent = root
        
        for node in node_map.values():
            node.children.sort(key=lambda x: x.order) # Sort children by order

        return root