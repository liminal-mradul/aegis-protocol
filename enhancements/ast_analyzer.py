import ast
from typing import Dict, List, Any

class AdvancedASTAnalyzer:
    def __init__(self):
        self.pattern_library = self._initialize_pattern_library()
        self.complexity_metrics = {}
        
    def _initialize_pattern_library(self) -> Dict:
        # Use plain function names (no parentheses) and dotted names where appropriate
        return {
            'security_risks': ['eval', 'exec', 'pickle.loads', 'os.system', 'subprocess.call', '__import__', 'compile', 'input'],
            'performance_issues': ['double_loop', 'deep_recursion', 'large_allocations'],
            'best_practices': ['type_hints', 'docstrings', 'error_handling'],
            'anti_patterns': ['god_object', 'spaghetti_code', 'magic_numbers']
        }
    
    def analyze_code_semantics(self, code: str) -> Dict:
        try:
            tree = ast.parse(code)
            
            analysis = {
                'complexity': self._compute_cyclomatic_complexity(tree),
                'security_risks': self._detect_security_risks(tree),
                'performance_issues': self._detect_performance_issues(tree),
                'maintainability': self._compute_maintainability_index(tree),
                'patterns_detected': self._detect_code_patterns(tree),
                'suggestions': self._generate_improvement_suggestions(tree)
            }
            
            return analysis
            
        except SyntaxError:
            return {
                'complexity': 1,
                'security_risks': [],
                'performance_issues': [],
                'maintainability': 0.7,
                'patterns_detected': [],
                'suggestions': ['Code syntax could not be parsed']
            }
    
    def _compute_cyclomatic_complexity(self, tree: ast.AST) -> int:
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
        
        return complexity
    
    def _detect_security_risks(self, tree: ast.AST) -> List[str]:
        risks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if any(risk in func_name for risk in self.pattern_library['security_risks']):
                    risks.append(f"Potential security risk: {func_name}")
        
        return risks
    
    def _detect_performance_issues(self, tree: ast.AST) -> List[str]:
        issues = []
        
        loop_depth = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_depth += 1
                if loop_depth > 2:
                    issues.append("Deeply nested loops detected")
            elif isinstance(node, ast.FunctionDef):
                loop_depth = 0
        
        return issues
    
    def _compute_maintainability_index(self, tree: ast.AST) -> float:
        complexity = self._compute_cyclomatic_complexity(tree)
        lines = len(ast.unparse(tree).split('\n')) if hasattr(ast, 'unparse') else 100
        
        maintainability = max(0, 1 - (complexity / lines * 0.1))
        return maintainability
    
    def _detect_code_patterns(self, tree: ast.AST) -> List[str]:
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not ast.get_docstring(node):
                    patterns.append("Function missing docstring")
        
        return patterns
    
    def _generate_improvement_suggestions(self, tree: ast.AST) -> List[str]:
        suggestions = []
        
        complexity = self._compute_cyclomatic_complexity(tree)
        if complexity > 10:
            suggestions.append(f"High cyclomatic complexity ({complexity}), consider refactoring")
        
        maintainability = self._compute_maintainability_index(tree)
        if maintainability < 0.7:
            suggestions.append("Low maintainability score, add comments and simplify logic")
        
        return suggestions
    
    def _get_function_name(self, node: ast.Call) -> str:
        # Return a dotted function name for attribute chains (e.g., os.path.join -> 'os.path.join')
        def _attr_to_dotted(attr_node: ast.AST) -> str:
            parts = []
            current = attr_node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))

        try:
            if isinstance(node.func, ast.Name):
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return _attr_to_dotted(node.func)
        except Exception:
            pass
        return "unknown"
