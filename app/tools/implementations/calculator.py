"""Calculator tool for mathematical operations."""

import ast
import operator
import time
from typing import Any, Dict, Optional
from ..base import BaseTool, ToolResult, ToolExecutionError, ToolExecutionStatus
from ..context import ToolExecutionContext


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations safely."""
    
    # Supported operators for safe evaluation
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Supported functions
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
    }
    
    def __init__(self):
        parameters_schema = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                },
                "precision": {
                    "type": "integer",
                    "description": "Number of decimal places for the result (default: 10)",
                    "minimum": 0,
                    "maximum": 15,
                    "default": 10
                }
            },
            "required": ["expression"]
        }
        
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations safely. Supports basic arithmetic operations (+, -, *, /, **, %), functions (abs, round, min, max, sum, pow), and parentheses.",
            parameters_schema=parameters_schema,
            timeout=5.0,  # Quick execution expected
            required_permissions=[]  # No special permissions needed
        )
    
    def _safe_eval(self, node: ast.AST) -> float:
        """Safely evaluate an AST node.
        
        Args:
            node: AST node to evaluate
            
        Returns:
            Numeric result of evaluation
            
        Raises:
            ToolExecutionError: If evaluation fails or uses unsupported operations
        """
        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return float(node.value)
            else:
                raise ToolExecutionError(
                    f"Unsupported constant type: {type(node.value)}",
                    error_code="UNSUPPORTED_CONSTANT"
                )
        elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
            return float(node.n)
        elif isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            op_type = type(node.op)
            
            if op_type not in self.OPERATORS:
                raise ToolExecutionError(
                    f"Unsupported operator: {op_type.__name__}",
                    error_code="UNSUPPORTED_OPERATOR"
                )
            
            try:
                if op_type == ast.Div and right == 0:
                    raise ToolExecutionError(
                        "Division by zero",
                        error_code="DIVISION_BY_ZERO"
                    )
                return self.OPERATORS[op_type](left, right)
            except (ZeroDivisionError, OverflowError, ValueError) as e:
                raise ToolExecutionError(
                    f"Mathematical error: {str(e)}",
                    error_code="MATH_ERROR"
                )
        elif isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand)
            op_type = type(node.op)
            
            if op_type not in self.OPERATORS:
                raise ToolExecutionError(
                    f"Unsupported unary operator: {op_type.__name__}",
                    error_code="UNSUPPORTED_OPERATOR"
                )
            
            return self.OPERATORS[op_type](operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            
            if func_name not in self.FUNCTIONS:
                raise ToolExecutionError(
                    f"Unsupported function: {func_name}",
                    error_code="UNSUPPORTED_FUNCTION"
                )
            
            args = [self._safe_eval(arg) for arg in node.args]
            
            try:
                return float(self.FUNCTIONS[func_name](*args))
            except (TypeError, ValueError) as e:
                raise ToolExecutionError(
                    f"Function error: {str(e)}",
                    error_code="FUNCTION_ERROR"
                )
        else:
            raise ToolExecutionError(
                f"Unsupported expression type: {type(node).__name__}",
                error_code="UNSUPPORTED_EXPRESSION"
            )
    
    async def execute(self, parameters: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> ToolResult:
        """Execute the calculator tool.
        
        Args:
            parameters: Tool parameters containing 'expression' and optional 'precision'
            context: Execution context (not used for calculator)
            
        Returns:
            ToolResult with calculation result
        """
        start_time = time.time()
        
        try:
            expression = parameters["expression"].strip()
            precision = parameters.get("precision", 10)
            
            if not expression:
                raise ToolExecutionError(
                    "Expression cannot be empty",
                    error_code="EMPTY_EXPRESSION"
                )
            
            # Parse the expression into an AST
            try:
                tree = ast.parse(expression, mode='eval')
            except SyntaxError as e:
                raise ToolExecutionError(
                    f"Invalid mathematical expression: {str(e)}",
                    error_code="SYNTAX_ERROR",
                    details={"expression": expression}
                )
            
            # Safely evaluate the AST
            result = self._safe_eval(tree.body)
            
            # Round to specified precision
            if precision >= 0:
                result = round(result, precision)
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                status=ToolExecutionStatus.SUCCESS,
                result={
                    "expression": expression,
                    "result": result,
                    "precision": precision
                },
                execution_time=execution_time,
                metadata={
                    "tool_version": "1.0",
                    "expression_length": len(expression)
                }
            )
            
        except ToolExecutionError:
            # Re-raise tool execution errors
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                error_message=f"Unexpected error: {str(e)}",
                execution_time=execution_time,
                metadata={"exception_type": type(e).__name__}
            )