"""
抽象验证器基类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging


class BaseVerifier(ABC):
    """验证器的抽象基类"""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def should_continue(self, results: List[Dict[str, Any]]) -> bool:
        """
        判断是否应该继续迭代
        
        Args:
            results: 历史结果列表，每个结果包含iteration, seed, feedback_score等
            
        Returns:
            bool: True表示应该继续迭代，False表示应该停止
        """
        pass
    
    @abstractmethod
    def get_stop_reason(self) -> str:
        """
        获取停止迭代的原因
        
        Returns:
            str: 停止原因的描述
        """
        pass
    
    def log_decision(self, results: List[Dict[str, Any]], should_continue: bool) -> None:
        """记录决策信息"""
        if results:
            latest_result = results[-1]
            self.logger.info(f"验证器决策: 继续={should_continue}, "
                           f"当前迭代={latest_result.get('iteration', 'N/A')}, "
                           f"当前评分={latest_result.get('feedback_score', 'N/A')}")
        else:
            self.logger.info(f"验证器决策: 继续={should_continue}, 无历史结果")