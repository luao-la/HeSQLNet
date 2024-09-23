#coding=utf-8

from my_lib.util.code_parser.code_parser import SitParser
import numpy as np

class MySitter(SitParser):
    @property
    def code_strings(self):
        # find the string nodes
        code_strs = []
        for node_id in self.ast_edges[0, :]:
            # print(self.ast_nodes[node_id])
            if self.ast_node_poses[node_id][-1] == -1 and self.ast_nodes[node_id][0] in ['"', "'", "'''", '"""']:
                parent_id = self.ast_edges[1, self.ast_edges[0, :].tolist().index(node_id)]
                if 'string' in self.ast_nodes[parent_id]:  # java里的字符类型在ast中为character_literal，不是string
                    if self.seg_attr:
                        str_edge_ids = np.argwhere(self.ast_edges[1, :] == parent_id)
                        code_strs.append([self.ast_nodes[self.ast_edges[0, idx[0]]] for idx in sorted(str_edge_ids)])
                    else:
                        # print(self.ast_nodes[node_id])
                        assert self.ast_nodes[node_id][-1] == self.ast_nodes[node_id][0]
                        code_strs.append(self.ast_nodes[node_id])
        return code_strs

def test():
    code = '''
    SELECT sum(bonus) FROM evaluation
    '''
    parser = MySitter(lan='Spider', lemmatize=False, lower=False, ast_intact=True, seg_attr=True, rev_dic=None,
                          user_words=None)  # 这里lower=False！！
    parser.parse(code)
    # a = parser.code_layout_sibling_edges
    print(parser._get_intact_ast_node_in_code_poses)
    # print('--------')
    # print(a)

if __name__=='__main__':
    test()

'''

# if __name__=='__main__':

    # nodes, edges, poses = java2ast_sitter(code,attr='all',seg_attr=True,lemmatize=True,lower=True,keep_punc=True,seg_var=True,)
    # print(list(zip(nodes,list(range(len(nodes))),poses,)))
    # print(edges)
    # print(poses)
    # print('***'*20)
    # nodes, edges, poses = java2ast(code, attr='all', seg_attr=True, lemmatize=True, lower=True, keep_punc=True,
    #                                       seg_var=True, )
    # print(list(zip(nodes, list(range(len(nodes))), poses, )))
    # print(edges)
    # print(poses)

    # def walk(node):  # 广度优先遍历所有功能节点
    #     """
    #     Recursively yield all descendant nodes in the tree starting at *node*
    #     (including *node* itself), in no specified order.  This is useful if you
    #     only want to modify nodes in place and don't care about the context.
    #     """
    #     from collections import deque
    #     todo = deque([node])
    #     while todo:
    #         node = todo.popleft()
    #         todo.extend(node.children)
    #         yield node
    # 
    # def is_func_node(node):    #如果有named子节点，说明是功能节点
    #     if isinstance(node,str):
    #         return False
    #     if node.type=='comment':
    #         return False
    #     if not node.is_named and not (node.parent.type.endswith('assignment') or node.parent.type.endswith('operator')):
    #         return False
    #     return True

    # from tree_sitter import Language, Parser

    from ._DFG import *

    # from .tree_sitter_repo import my


    # py_language = Language('tree_sitter_repo/my-languages.so', 'Spider')
    # parser = Parser()
    # parser.set_language(py_language)
    # bcode = bytes(code, 'utf8')
    # tree = parser.parse(bcode)
    # print(tree.root_node.children)

    # i = 0
    # print(tree.root_node)
    # 
    # for child in walk(tree.root_node):
    #     # if child.type=="=":
    #     print(child)
    #     print(child.type)
    #     print(str(child.text,encoding="utf-8"))
    #     print(child.is_named)
    #     print("***" * 20)
    #     # print(child.parent.type,str(child.parent.text,encoding="utf-8"))
    #     i += 1
    # print(i)
'''