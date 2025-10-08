"""
基于阈值的验证器实现
"""

from typing import List, Dict, Any
from .base_verifier import BaseVerifier
import logging


class ThresholdVerifier(BaseVerifier):
    """基于阈值的验证器"""
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        
        # 配置参数（10分制）
        self.position_threshold = cfg.get('position_threshold', 7.0)  # 位置评分阈值
        self.joint_threshold = cfg.get('joint_threshold', 8.0)  # 关节评分阈值
        self.overall_threshold = cfg.get('overall_threshold', 7.5)  # 总分阈值
        self.max_iterations = cfg.get('max_iterations', 10)
        self.patience = cfg.get('patience', 3)  # 连续多少次没有改进就停止
        
        # 状态变量
        self.best_score = -1
        self.no_improvement_count = 0
        self.stop_reason = ""
        
        self.logger.info(f"初始化ThresholdVerifier（10分制）: position_threshold={self.position_threshold}/10, "
                        f"joint_threshold={self.joint_threshold}/10, overall_threshold={self.overall_threshold}/10")
    
    def should_continue(self, results: List[Dict[str, Any]]) -> bool:
        """判断是否应该继续迭代"""
        if not results:
            return True
        
        # 获取最新结果
        latest_result = results[-1]
        current_iteration = latest_result.get('iteration', 0)
        current_score = latest_result.get('feedback_score', -1)
        
        # 检查最大迭代次数
        if current_iteration >= self.max_iterations:
            self.stop_reason = f"达到最大迭代次数 {self.max_iterations}"
            self.log_decision(results, False)
            return False
        
        # 检查整体阈值
        if current_score >= self.overall_threshold:
            self.stop_reason = f"达到整体阈值 {self.overall_threshold}/10 (当前分数: {current_score:.1f}/10)"
            self.log_decision(results, False)
            return False
        
        # 检查分项阈值（如果有详细评分的话）
        position_score = latest_result.get('position_score', None)
        joint_score = latest_result.get('joint_score', None)
        
        if position_score is not None and joint_score is not None:
            if position_score >= self.position_threshold and joint_score >= self.joint_threshold:
                self.stop_reason = (f"位置和关节都达到阈值 "
                                  f"(位置: {position_score:.1f}>={self.position_threshold}/10, "
                                  f"关节: {joint_score:.1f}>={self.joint_threshold}/10)")
                self.log_decision(results, False)
                return False
        
        # 检查改进情况
        if current_score > self.best_score:
            self.best_score = current_score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # 检查耐心值
        if self.no_improvement_count >= self.patience:
            self.stop_reason = f"连续 {self.patience} 次迭代无改进 (最佳分数: {self.best_score:.1f}/10)"
            self.log_decision(results, False)
            return False
        
        # 继续迭代
        self.log_decision(results, True)
        return True
    
    def get_stop_reason(self) -> str:
        """获取停止原因"""
        return self.stop_reason or "未知原因"
    
    def reset(self):
        """重置验证器状态"""
        self.best_score = -1
        self.no_improvement_count = 0
        self.stop_reason = ""
        self.logger.info("验证器状态已重置")


class AdaptiveThresholdVerifier(ThresholdVerifier):
    """自适应阈值验证器"""
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        
        # 自适应参数
        self.initial_threshold = self.overall_threshold
        self.threshold_decay = cfg.get('threshold_decay', 0.95)
        self.min_threshold = cfg.get('min_threshold', 0.5)
        self.adaptation_interval = cfg.get('adaptation_interval', 3)
        
        self.logger.info(f"初始化AdaptiveThresholdVerifier: threshold_decay={self.threshold_decay}, "
                        f"min_threshold={self.min_threshold}")
    
    def should_continue(self, results: List[Dict[str, Any]]) -> bool:
        """带自适应阈值的判断"""
        if not results:
            return True
        
        # 每隔一定迭代次数降低阈值
        current_iteration = results[-1].get('iteration', 0)
        if current_iteration > 0 and current_iteration % self.adaptation_interval == 0:
            new_threshold = max(self.overall_threshold * self.threshold_decay, self.min_threshold)
            if new_threshold != self.overall_threshold:
                self.logger.info(f"自适应调整阈值: {self.overall_threshold:.3f} -> {new_threshold:.3f}")
                self.overall_threshold = new_threshold
        
        return super().should_continue(results)